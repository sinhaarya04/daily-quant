"""Day 017 — Sector rotation via relative momentum (monthly).

Strategy (monthly):
- Compute trailing lookback-month return for each sector ETF
- Select top K
- Hold equal-weight for next month

Outputs are written to ./output.

Run:
  python main.py --start 2005-01-01 --lookback 6 --topk 3

Dependencies: numpy, pandas, yfinance, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


SECTOR_ETFS = [
    "XLB",  # Materials
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Staples
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Discretionary
    "XLRE",  # Real Estate
]


@dataclass(frozen=True)
class Perf:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float


def download_adj_close(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a multiindex if multiple tickers, single series otherwise.
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].rename(columns={"Close": tickers[0]}).copy()

    px = px.dropna(how="all")
    # forward-fill small gaps (holidays, etc.), but do not fill leading NAs
    px = px.ffill()
    return px


def month_end_prices(px: pd.DataFrame) -> pd.DataFrame:
    # Use last available price each calendar month.
    return px.resample("M").last()


def perf_stats(rets_m: pd.Series) -> Perf:
    rets_m = rets_m.dropna()
    if len(rets_m) < 3:
        return Perf(np.nan, np.nan, np.nan, np.nan)

    ann_factor = 12.0
    eq = (1.0 + rets_m).cumprod()

    n_years = len(rets_m) / ann_factor
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else np.nan

    vol = float(rets_m.std(ddof=1) * np.sqrt(ann_factor))
    sharpe = float((rets_m.mean() * ann_factor) / vol) if vol and vol > 0 else np.nan

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())

    return Perf(cagr=cagr, vol=vol, sharpe=sharpe, max_dd=max_dd)


def backtest_relative_momentum(
    sector_px_m: pd.DataFrame,
    spy_px_m: pd.Series,
    lookback: int = 6,
    topk: int = 3,
    tcost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (equity_curves, weights, monthly_returns).

    Transaction costs model:
      cost_t = turnover_t * (tcost_bps/1e4)
    where turnover is sum(abs(w_t - w_{t-1})).
    """

    # monthly returns
    sec_rets = sector_px_m.pct_change()
    spy_rets = spy_px_m.pct_change().rename("SPY")

    # momentum signal: trailing lookback return excluding current month
    # (i.e., at month t, use months t-lookback..t-1)
    mom = (1.0 + sec_rets).rolling(lookback).apply(np.prod, raw=True) - 1.0

    # weights at month t (applied to returns at month t+1):
    # use mom as of month t, shift by 1 to avoid look-ahead.
    mom_lag = mom.shift(1)

    w = pd.DataFrame(0.0, index=sec_rets.index, columns=sec_rets.columns)
    for dt in w.index:
        scores = mom_lag.loc[dt].dropna()
        if len(scores) == 0:
            continue
        top = scores.sort_values(ascending=False).head(topk).index
        w.loc[dt, top] = 1.0 / len(top)

    # portfolio returns (weights at t-1 applied to returns at t)
    w_prev = w.shift(1)
    port_rets = (w_prev * sec_rets).sum(axis=1)

    # transaction costs based on turnover at rebalance time (t-1 -> t)
    turnover = (w_prev - w_prev.shift(1)).abs().sum(axis=1)
    cost = turnover * (tcost_bps / 1e4)
    port_rets_net = (port_rets - cost).rename("SectorMom")

    monthly = pd.concat([port_rets_net, spy_rets], axis=1).dropna()

    equity = (1.0 + monthly).cumprod()

    weights = w.loc[monthly.index].copy()

    diag = pd.DataFrame(
        {
            "turnover": turnover.loc[monthly.index],
            "tcost": cost.loc[monthly.index],
        }
    )

    return equity, weights, monthly, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2005-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tcost-bps", type=float, default=0.0)
    ap.add_argument(
        "--tickers",
        type=str,
        default=" ".join(SECTOR_ETFS),
        help="Space-separated sector ETF tickers",
    )
    args = ap.parse_args()

    tickers = args.tickers.split()
    if args.topk < 1 or args.topk > len(tickers):
        raise SystemExit("--topk must be between 1 and number of tickers")
    if args.lookback < 1:
        raise SystemExit("--lookback must be >= 1")

    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    # Download daily prices then convert to month-end
    sector_px = download_adj_close(tickers, start=args.start, end=args.end)
    spy_px = download_adj_close(["SPY"], start=args.start, end=args.end)["SPY"]

    sector_px_m = month_end_prices(sector_px)
    spy_px_m = month_end_prices(spy_px.to_frame("SPY"))["SPY"]

    equity, weights, monthly, diag = backtest_relative_momentum(
        sector_px_m=sector_px_m,
        spy_px_m=spy_px_m,
        lookback=args.lookback,
        topk=args.topk,
        tcost_bps=args.tcost_bps,
    )

    # Stats
    stats = {
        col: perf_stats(monthly[col]) for col in monthly.columns
    }
    stats_df = pd.DataFrame(
        {
            "CAGR": {k: v.cagr for k, v in stats.items()},
            "Vol": {k: v.vol for k, v in stats.items()},
            "Sharpe": {k: v.sharpe for k, v in stats.items()},
            "MaxDD": {k: v.max_dd for k, v in stats.items()},
        }
    )

    print("\nPerformance (monthly, annualized):")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(stats_df)

    # Save artifacts
    equity.to_csv(outdir / "equity_curves.csv", index=True)
    weights.to_csv(outdir / "weights.csv", index=True)
    monthly.to_csv(outdir / "monthly_returns.csv", index=True)
    diag.to_csv(outdir / "diagnostics.csv", index=True)

    # Plot
    import matplotlib.pyplot as plt

    ax = equity.plot(figsize=(10, 5), title="Equity Curves: Sector Momentum vs SPY")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "equity_curves.png", dpi=150)

    # Show last weights snapshot
    last_dt = weights.dropna(how="all").index.max()
    if pd.notna(last_dt):
        last_w = weights.loc[last_dt]
        held = last_w[last_w > 0].sort_values(ascending=False)
        held_str = ", ".join([f"{k} ({v:.0%})" for k, v in held.items()])
        print(f"\nLast rebalance ({last_dt.date()}): {held_str}")

    print(f"\nWrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
