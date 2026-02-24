"""Day 009 — Trend + Vol Targeting on SPY

Educational mini-backtest:
- Trend filter: only take risk when price > long MA
- Vol targeting: scale exposure to a target annualized vol using rolling realized vol
- Optional leverage cap and simple linear transaction costs

Outputs:
- Console summary
- Equity curve plot saved to out_equity.png

Data: yfinance (SPY adjusted close)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


TRADING_DAYS = 252


@dataclass
class Stats:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float


def compute_stats(equity: pd.Series) -> Stats:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return Stats(np.nan, np.nan, np.nan, np.nan)

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol and vol > 0 else np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    return Stats(float(cagr), float(vol), float(sharpe), float(max_dd))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--ma", type=int, default=200, help="trend MA window (trading days)")
    ap.add_argument("--vol_window", type=int, default=63, help="realized vol window (trading days)")
    ap.add_argument("--target_vol", type=float, default=0.10, help="target annualized vol (e.g. 0.10 = 10%%)")
    ap.add_argument("--max_leverage", type=float, default=2.0)
    ap.add_argument(
        "--tc_bps",
        type=float,
        default=2.0,
        help="transaction cost in bps applied to |delta weight| each day",
    )
    args = ap.parse_args()

    px = yf.download(args.ticker, start=args.start, auto_adjust=True, progress=False)["Close"].rename("px")
    if px.empty:
        raise SystemExit("No data downloaded. Check ticker/start date.")

    df = pd.DataFrame({"px": px})
    df["ret"] = df["px"].pct_change()

    # Trend filter
    df["ma"] = df["px"].rolling(args.ma).mean()
    df["trend_on"] = (df["px"] > df["ma"]).astype(float)

    # Realized vol (annualized)
    df["realized_vol"] = df["ret"].rolling(args.vol_window).std() * np.sqrt(TRADING_DAYS)

    # Vol target weight before caps and trend filter
    w_raw = args.target_vol / df["realized_vol"]
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan)

    # Apply caps and trend filter
    df["w"] = (w_raw.clip(lower=0.0, upper=args.max_leverage) * df["trend_on"]).fillna(0.0)

    # Transaction costs on weight changes
    df["dw"] = df["w"].diff().abs().fillna(0.0)
    tc = (args.tc_bps / 1e4) * df["dw"]

    # Strategy returns: yesterday's weight applied to today's return
    df["strat_ret_gross"] = df["w"].shift(1).fillna(0.0) * df["ret"].fillna(0.0)
    df["strat_ret"] = df["strat_ret_gross"] - tc

    df["eq_bh"] = (1.0 + df["ret"].fillna(0.0)).cumprod()
    df["eq_strat"] = (1.0 + df["strat_ret"].fillna(0.0)).cumprod()

    s_bh = compute_stats(df["eq_bh"])
    s_st = compute_stats(df["eq_strat"])

    print("=== Inputs ===")
    print(f"Ticker: {args.ticker} | start={args.start}")
    print(
        f"MA={args.ma}d | vol_window={args.vol_window}d | target_vol={args.target_vol:.2%} | max_lev={args.max_leverage:.2f} | tc={args.tc_bps:.1f} bps"
    )
    print()

    def fmt(stats: Stats) -> str:
        return f"CAGR={stats.cagr:>7.2%} | Vol={stats.vol:>6.2%} | Sharpe={stats.sharpe:>5.2f} | MaxDD={stats.max_dd:>7.2%}"

    print("=== Results ===")
    print("Buy & Hold:", fmt(s_bh))
    print("Trend+VolTarget:", fmt(s_st))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["eq_bh"], label="Buy & Hold")
    ax.plot(df.index, df["eq_strat"], label="Trend+VolTarget")
    ax.set_title("Day 009 — SPY: Trend filter + Vol targeting")
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("out_equity.png", dpi=150)
    print("\nSaved plot: out_equity.png")


if __name__ == "__main__":
    main()
