"""Day 010 — Equity/Bond Correlation Regimes (SPY vs TLT) + 10Y Yield

Goal: visualize how the rolling correlation between equity and long-duration bonds
changes over time, and how it lines up with the 10Y yield level.

Data sources:
- yfinance: SPY, TLT adjusted close
- FRED: DGS10 (10-year Treasury constant maturity rate)

Outputs:
- out_corr_regimes.png
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr


TRADING_DAYS = 252


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all")
    px.columns = [c.upper() for c in px.columns]
    return px


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--window", type=int, default=63, help="rolling correlation window (trading days)")
    ap.add_argument("--equity", default="SPY")
    ap.add_argument("--bond", default="TLT")
    args = ap.parse_args()

    tickers = [args.equity.upper(), args.bond.upper()]
    px = download_prices(tickers, args.start)
    if px.empty:
        raise SystemExit("No price data downloaded.")

    rets = px.pct_change().dropna()

    # Rolling correlation
    corr = rets[tickers[0]].rolling(args.window).corr(rets[tickers[1]])

    # FRED 10Y yield
    dgs10 = pdr.DataReader("DGS10", "fred", start=args.start)["DGS10"].rename("DGS10")
    dgs10 = dgs10.ffill()  # FRED has missing values on weekends/holidays

    # Align everything
    df = pd.concat(
        {
            "px_equity": px[tickers[0]],
            "px_bond": px[tickers[1]],
            "corr": corr,
            "dgs10": dgs10,
        },
        axis=1,
    ).dropna(subset=["px_equity", "px_bond"])

    # Quick summaries
    corr_valid = df["corr"].dropna()
    neg_share = float((corr_valid < 0).mean()) if len(corr_valid) else np.nan
    pos_share = float((corr_valid > 0).mean()) if len(corr_valid) else np.nan

    # Annualized vols for context
    vol_eq = float(rets[tickers[0]].std() * np.sqrt(TRADING_DAYS))
    vol_bd = float(rets[tickers[1]].std() * np.sqrt(TRADING_DAYS))

    print("=== Day 010: Equity/Bond correlation regimes ===")
    print(f"Period start: {args.start} | window={args.window}d")
    print(f"Tickers: {tickers[0]} vs {tickers[1]}")
    print(f"Ann vol: {tickers[0]}={vol_eq:.2%} | {tickers[1]}={vol_bd:.2%}")
    print(f"Rolling corr < 0 share: {neg_share:.1%} | > 0 share: {pos_share:.1%}")

    # Build equity curves (normalized)
    eq_equity = (1.0 + rets[tickers[0]]).cumprod().reindex(df.index).ffill()
    eq_bond = (1.0 + rets[tickers[1]]).cumprod().reindex(df.index).ffill()

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 0.8]})

    ax0, ax1, ax2 = axes

    ax0.plot(df.index, eq_equity, label=f"{tickers[0]} (equity curve)")
    ax0.plot(df.index, eq_bond, label=f"{tickers[1]} (equity curve)")
    ax0.set_title("Day 010 — Equity/Bond Corr Regimes + 10Y Yield")
    ax0.set_ylabel("Equity (start=1.0)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left")

    ax1.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax1.plot(df.index, df["corr"], color="#2c7fb8", label=f"Rolling corr ({args.window}d)")
    ax1.set_ylabel("Corr")
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(df.index, df["dgs10"], color="#d95f0e", label="10Y yield (DGS10)")
    ax2.set_ylabel("Yield (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig("out_corr_regimes.png", dpi=150)
    print("Saved plot: out_corr_regimes.png")


if __name__ == "__main__":
    main()
