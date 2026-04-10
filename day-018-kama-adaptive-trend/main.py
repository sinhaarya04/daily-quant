"""Day 018 — KAMA (Kaufman Adaptive Moving Average) trend filter.

Self-contained script:
- downloads daily data with yfinance
- implements KAMA
- backtests: buy&hold, SMA filter, KAMA filter

Usage:
  python day-018-kama-adaptive-trend/main.py --ticker SPY --start 2005-01-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def kama(price: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Compute Kaufman's Adaptive Moving Average (KAMA).

    References (common formulation):
      ER = |P_t - P_{t-n}| / sum_{i=1..n} |P_{t-i} - P_{t-i-1}|
      SC = (ER*(fastSC-slowSC) + slowSC)^2
      KAMA_t = KAMA_{t-1} + SC*(P_t - KAMA_{t-1})

    fast and slow are EMA-equivalent periods.
    """
    p = price.astype(float)

    change = (p - p.shift(er_window)).abs()
    volatility = (p.diff().abs()).rolling(er_window).sum()
    er = change / volatility
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = pd.Series(index=p.index, dtype=float)

    # initialize with first available price (after ER window) to avoid long NaN runs
    first_idx = sc.index[0]
    if sc.notna().any():
        first_idx = sc[sc.notna()].index[0]
    out.loc[:first_idx] = np.nan
    out.loc[first_idx] = p.loc[first_idx]

    for t_prev, t in zip(out.index[out.index.get_loc(first_idx) : -1], out.index[out.index.get_loc(first_idx) + 1 :]):
        prev = out.loc[t_prev]
        out.loc[t] = prev + sc.loc[t] * (p.loc[t] - prev)

    return out


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


@dataclass
class Perf:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    turnover: float


def perf_stats(returns: pd.Series, positions: pd.Series | None = None, periods_per_year: int = 252) -> Perf:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return Perf(np.nan, np.nan, np.nan, np.nan, np.nan)

    equity = (1.0 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / vol if vol > 0 else np.nan
    mdd = max_drawdown(equity)

    turnover = np.nan
    if positions is not None:
        # proportion of days with position changes (rough turnover proxy)
        pos = positions.reindex(r.index).fillna(0.0)
        turnover = float((pos.diff().abs() > 1e-12).mean())

    return Perf(float(cagr), float(vol), float(sharpe), float(mdd), float(turnover))


def backtest_filter(
    close: pd.Series,
    signal_long: pd.Series,
    cost_bps: float = 1.0,
) -> Dict[str, pd.Series]:
    """Backtest long/cash strategy with costs on position changes."""
    close = close.astype(float)
    ret = close.pct_change()

    pos = signal_long.astype(float).clip(0.0, 1.0).shift(1).fillna(0.0)

    # cost applied when changing position (enter/exit)
    trade = pos.diff().abs().fillna(0.0)
    cost = trade * (cost_bps / 1e4)

    strat_ret = pos * ret - cost

    return {
        "returns": strat_ret,
        "position": pos,
        "equity": (1.0 + strat_ret.fillna(0.0)).cumprod(),
        "cost": cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--er_window", type=int, default=10)
    ap.add_argument("--fast", type=int, default=2)
    ap.add_argument("--slow", type=int, default=30)
    ap.add_argument("--sma_window", type=int, default=200)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--out", default="day-018-kama-adaptive-trend/output.png")
    args = ap.parse_args()

    df = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit("No data returned. Check ticker/start/end.")

    # yfinance may return either flat columns or a MultiIndex (e.g., when grouping by ticker)
    if isinstance(df.columns, pd.MultiIndex):
        close_df = df.xs("Close", level=0, axis=1)
        # for single-ticker downloads, this will be a 1-col DataFrame
        close = close_df.iloc[:, 0]
    else:
        close = df["Close"]
    close = close.dropna().astype(float)
    close.name = "close"

    k = kama(close, er_window=args.er_window, fast=args.fast, slow=args.slow)
    sma = close.rolling(args.sma_window).mean()

    bh_ret = close.pct_change()
    bh_equity = (1.0 + bh_ret.fillna(0.0)).cumprod()

    kama_bt = backtest_filter(close, signal_long=(close > k), cost_bps=args.cost_bps)
    sma_bt = backtest_filter(close, signal_long=(close > sma), cost_bps=args.cost_bps)

    stats = pd.DataFrame(
        {
            "Buy&Hold": asdict(perf_stats(bh_ret)),
            f"SMA{args.sma_window}": asdict(perf_stats(sma_bt["returns"], sma_bt["position"])),
            f"KAMA({args.er_window},{args.fast},{args.slow})": asdict(
                perf_stats(kama_bt["returns"], kama_bt["position"])
            ),
        }
    ).T

    def fmt(x: float, pct: bool = False) -> str:
        if x is None or np.isnan(x):
            return "nan"
        return f"{x*100:6.2f}%" if pct else f"{x:6.2f}"

    pretty = pd.DataFrame(
        {
            "CAGR": [fmt(v, pct=True) for v in stats["cagr"].values],
            "Vol": [fmt(v, pct=True) for v in stats["vol"].values],
            "Sharpe": [fmt(v, pct=False) for v in stats["sharpe"].values],
            "MaxDD": [fmt(v, pct=True) for v in stats["max_dd"].values],
            "Turnover": [fmt(v, pct=True) for v in stats["turnover"].values],
        },
        index=stats.index,
    )

    print("\nPerformance (daily data, simple backtest)\n")
    print(pretty.to_string())

    # Plot: price + filters + equity curves
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(close.index, close.values, label=f"{args.ticker} (Adj Close)", linewidth=1.2)
    ax.plot(k.index, k.values, label="KAMA", linewidth=1.2)
    ax.plot(sma.index, sma.values, label=f"SMA{args.sma_window}", linewidth=1.0, alpha=0.9)
    ax.set_title("Price + Trend Filters")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax2 = axes[1]
    ax2.plot(bh_equity.index, bh_equity.values, label="Buy&Hold", linewidth=1.2)
    ax2.plot(sma_bt["equity"].index, sma_bt["equity"].values, label=f"SMA{args.sma_window} filter", linewidth=1.2)
    ax2.plot(kama_bt["equity"].index, kama_bt["equity"].values, label="KAMA filter", linewidth=1.2)
    ax2.set_title(f"Equity Curves (cost={args.cost_bps:.1f} bps per position change)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nSaved plot → {args.out}")


if __name__ == "__main__":
    main()
