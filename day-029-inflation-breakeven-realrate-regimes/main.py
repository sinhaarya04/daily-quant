"""Day 029: Inflation expectations & real-rate regimes vs SPY returns.

Mini-project goals:
- Pull free macro series from FRED (no API key):
  - T5YIE: 5-Year Breakeven Inflation Rate (proxy for inflation expectations)
  - DFII10: 10-Year Treasury Inflation-Indexed Security, Constant Maturity (real rate)
- Pull SPY prices from yfinance.
- Create simple regime labels and evaluate subsequent 1M forward returns.

This is deliberately simple and self-contained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def load_fred_series(series: str) -> pd.Series:
    """Load a FRED series via the public fredgraph CSV endpoint."""
    url = FRED_CSV.format(series=series)
    df = pd.read_csv(url)
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True).dt.tz_convert(None)
    s = df.set_index("DATE")[series]
    s = pd.to_numeric(s, errors="coerce")
    s.name = series
    return s


def load_spy(start: str, end: str) -> pd.Series:
    px = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index)
    px.name = "SPY"
    return px


@dataclass
class RegimeConfig:
    realrate_threshold: float = 0.0
    breakeven_lookback_days: int = 63  # ~3 months


def compute_dataset(start: str, end: str, cfg: RegimeConfig) -> pd.DataFrame:
    spy = load_spy(start, end)
    t5yie = load_fred_series("T5YIE")
    dfii10 = load_fred_series("DFII10")

    # Align on business days via join on dates. FRED is often daily but with gaps.
    df = pd.concat([spy, t5yie, dfii10], axis=1).sort_index()
    df = df.loc[start:end].dropna(subset=["SPY"]).copy()

    # Fill macro series forward (macro doesn't need daily trading calendar granularity).
    df[["T5YIE", "DFII10"]] = df[["T5YIE", "DFII10"]].ffill()

    # Returns
    df["ret_1d"] = df["SPY"].pct_change()
    fwd_h = 21  # ~1 trading month
    df["fwd_ret_1m"] = df["SPY"].pct_change(fwd_h).shift(-fwd_h)

    # Regime features
    df["realrate_low"] = df["DFII10"] < cfg.realrate_threshold
    df["breakeven_mom"] = df["T5YIE"].diff(cfg.breakeven_lookback_days)
    df["breakeven_rising"] = df["breakeven_mom"] > 0

    # 4-state regime
    df["regime"] = (
        df["realrate_low"].astype(int) * 2 + df["breakeven_rising"].astype(int)
    )
    df["regime"] = df["regime"].map(
        {
            0: "Real>=0 & BE falling",
            1: "Real>=0 & BE rising",
            2: "Real<0 & BE falling",
            3: "Real<0 & BE rising",
        }
    )

    return df


def summarize_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["fwd_ret_1m", "regime"]).copy()
    out = (
        x.groupby("regime")["fwd_ret_1m"]
        .agg(n="count", mean="mean", median="median", std="std")
        .sort_values("mean", ascending=False)
    )
    out["sharpe_like"] = out["mean"] / out["std"]
    return out


def toy_strategy_equity_curve(df: pd.DataFrame, preferred_regime: str) -> pd.Series:
    """A toy timing strategy: hold SPY only when in preferred regime, else flat (0%)."""
    x = df.dropna(subset=["ret_1d", "regime"]).copy()
    pos = (x["regime"] == preferred_regime).astype(float)
    strat_ret = pos.shift(1).fillna(0.0) * x["ret_1d"]  # trade next day to avoid look-ahead
    eq = (1.0 + strat_ret).cumprod()
    eq.name = "strategy_equity"
    return eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2003-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default: today")
    ap.add_argument("--realrate-threshold", type=float, default=0.0)
    ap.add_argument("--be-lookback", type=int, default=63)
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()

    cfg = RegimeConfig(
        realrate_threshold=args.realrate_threshold,
        breakeven_lookback_days=args.be_lookback,
    )

    df = compute_dataset(args.start, args.end, cfg)
    summ = summarize_forward_returns(df)

    outdir = args.out
    import os

    os.makedirs(outdir, exist_ok=True)

    # Save summary
    summ.to_csv(os.path.join(outdir, "forward_return_summary.csv"))

    # Choose best regime by mean forward return
    best_regime = summ.index[0]
    eq = toy_strategy_equity_curve(df, best_regime)

    # Plot macro + SPY
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    df["SPY"].dropna().plot(ax=axes[0], color="black", lw=1)
    axes[0].set_title("SPY (adj close)")

    df["T5YIE"].plot(ax=axes[1], color="tab:blue", lw=1)
    axes[1].set_title("T5YIE: 5Y Breakeven Inflation (%)")

    df["DFII10"].plot(ax=axes[2], color="tab:red", lw=1)
    axes[2].axhline(cfg.realrate_threshold, color="gray", ls="--", lw=1)
    axes[2].set_title("DFII10: 10Y TIPS Real Rate (%)")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "macro_and_spy.png"), dpi=150)
    plt.close(fig)

    # Plot regime forward return boxplot
    x = df.dropna(subset=["fwd_ret_1m", "regime"]).copy()
    order = list(summ.index)
    fig, ax = plt.subplots(figsize=(11, 4))
    x.boxplot(column="fwd_ret_1m", by="regime", ax=ax, grid=False)
    ax.set_title("1M forward SPY returns by regime")
    ax.set_ylabel("Forward return")
    plt.suptitle("")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "forward_return_boxplot.png"), dpi=150)
    plt.close(fig)

    # Plot toy equity curve vs buy&hold
    buy_hold = (1.0 + df["ret_1d"].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4))
    buy_hold.plot(ax=ax, label="Buy & Hold", color="gray", lw=1)
    eq.plot(ax=ax, label=f"Toy: hold only in '{best_regime}'", color="tab:green", lw=1.5)
    ax.set_yscale("log")
    ax.set_title("Toy timing strategy (log scale, no transaction costs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "toy_strategy_equity.png"), dpi=150)
    plt.close(fig)

    # Print console summary
    print("Forward 1M return summary (by regime):")
    print(summ.round(4).to_string())
    print() 
    print(f"Best regime by mean forward return: {best_regime}")
    print(f"Wrote outputs to: {outdir}/")


if __name__ == "__main__":
    main()
