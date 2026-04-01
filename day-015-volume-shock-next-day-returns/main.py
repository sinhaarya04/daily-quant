"""Day 015 — Volume Shock vs Next-Day Returns (SPY)

Idea
----
Compute a rolling z-score of daily volume, then compare next-day returns
conditional on volume being unusually high/low.

Outputs
-------
Writes tables + plots to day-015-volume-shock-next-day-returns/outputs/

Usage
-----
python day-015-volume-shock-next-day-returns/main.py --ticker SPY --start 2000-01-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


OUTDIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class BucketResult:
    name: str
    n: int
    mean: float
    median: float
    std: float
    sharpe_daily: float
    win_rate: float


def download_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    # yfinance column variants: sometimes 'Adj Close' exists, sometimes not.
    df = df.rename(columns={"Adj Close": "AdjClose"})
    if "AdjClose" not in df.columns:
        df["AdjClose"] = df["Close"]

    needed = {"Open", "High", "Low", "Close", "AdjClose", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns from yfinance download: {sorted(missing)}")

    df = df.dropna(subset=["AdjClose", "Volume"]).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()

    out["ret"] = out["AdjClose"].pct_change()
    out["next_ret"] = out["ret"].shift(-1)

    vol = out["Volume"].astype(float)
    roll_mean = vol.rolling(window).mean()
    roll_std = vol.rolling(window).std(ddof=0)

    out["vol_z"] = (vol - roll_mean) / roll_std

    # Keep rows where z-score and next-day return exist
    out = out.dropna(subset=["vol_z", "next_ret"]).copy()
    return out


def summarize_bucket(s: pd.Series, name: str) -> BucketResult:
    s = s.dropna()
    n = int(s.shape[0])
    mean = float(s.mean())
    median = float(s.median())
    std = float(s.std(ddof=0))
    sharpe_daily = float(mean / std) if std > 0 else np.nan
    win_rate = float((s > 0).mean())
    return BucketResult(
        name=name,
        n=n,
        mean=mean,
        median=median,
        std=std,
        sharpe_daily=sharpe_daily,
        win_rate=win_rate,
    )


def to_frame(results: list[BucketResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results]).set_index("name")
    return df


def plot_distributions(df: pd.DataFrame, z_lo: float, z_hi: float, ticker: str) -> None:
    # Histogram overlay
    bins = 60
    plt.figure(figsize=(10, 6))

    buckets = {
        f"Low vol (z <= {z_lo:.1f})": df.loc[df["vol_z"] <= z_lo, "next_ret"],
        "Normal": df.loc[(df["vol_z"] > z_lo) & (df["vol_z"] < z_hi), "next_ret"],
        f"High vol (z >= {z_hi:.1f})": df.loc[df["vol_z"] >= z_hi, "next_ret"],
    }

    for name, s in buckets.items():
        s = s.dropna()
        if len(s) == 0:
            continue
        plt.hist(s.values, bins=bins, alpha=0.45, density=True, label=f"{name} (n={len(s)})")

    plt.axvline(0.0, color="black", lw=1)
    plt.title(f"{ticker}: Next-day returns conditioned on volume z-score")
    plt.xlabel("Next-day return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "next_day_return_hist.png", dpi=150)
    plt.close()


def plot_mean_by_volz(df: pd.DataFrame, ticker: str, n_bins: int) -> None:
    # Bin vol_z and compute mean next_ret by bin
    z = df["vol_z"].clip(lower=df["vol_z"].quantile(0.01), upper=df["vol_z"].quantile(0.99))
    binned = pd.qcut(z, q=n_bins, duplicates="drop")
    g = df.groupby(binned, observed=True)["next_ret"].agg(["mean", "count"])

    plt.figure(figsize=(10, 6))
    x = np.arange(len(g))
    plt.bar(x, g["mean"].values)
    plt.axhline(0.0, color="black", lw=1)
    plt.title(f"{ticker}: Mean next-day return by volume-z quantile bin")
    plt.xlabel("Volume-z quantile bin")
    plt.ylabel("Mean next-day return")
    plt.xticks(ticks=x, labels=[str(i) for i in range(1, len(g) + 1)], rotation=0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "mean_next_day_return_by_volz_bin.png", dpi=150)
    plt.close()

    g.to_csv(OUTDIR / "mean_next_day_return_by_volz_bin.csv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY", help="Ticker to analyze (default: SPY)")
    p.add_argument("--start", type=str, default="2000-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--window", type=int, default=60, help="Rolling window for volume z-score")
    p.add_argument("--z_lo", type=float, default=-1.5, help="Low volume-z threshold")
    p.add_argument("--z_hi", type=float, default=1.5, help="High volume-z threshold")
    p.add_argument("--bins", type=int, default=12, help="Number of quantile bins for vol-z plot")
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    raw = download_ohlcv(args.ticker, args.start)
    df = compute_features(raw, window=args.window)

    low = df.loc[df["vol_z"] <= args.z_lo, "next_ret"]
    normal = df.loc[(df["vol_z"] > args.z_lo) & (df["vol_z"] < args.z_hi), "next_ret"]
    high = df.loc[df["vol_z"] >= args.z_hi, "next_ret"]

    results = [
        summarize_bucket(low, f"Low vol (z <= {args.z_lo})"),
        summarize_bucket(normal, "Normal"),
        summarize_bucket(high, f"High vol (z >= {args.z_hi})"),
    ]
    summary = to_frame(results)

    # Pretty + raw outputs
    summary_rounded = summary.copy()
    for c in ["mean", "median", "std", "sharpe_daily", "win_rate"]:
        summary_rounded[c] = summary_rounded[c].astype(float).round(6)
    summary_rounded.to_csv(OUTDIR / "bucket_summary.csv")

    # Extra diagnostic table: average next-day returns by vol_z decile
    plot_distributions(df, args.z_lo, args.z_hi, args.ticker)
    plot_mean_by_volz(df, args.ticker, n_bins=args.bins)

    # Save the feature dataframe (small-ish; but keep it optional-ish)
    df[["AdjClose", "Volume", "ret", "next_ret", "vol_z"]].to_csv(OUTDIR / "features.csv")

    print("\nBucket summary (next-day returns):")
    print(summary_rounded)
    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
