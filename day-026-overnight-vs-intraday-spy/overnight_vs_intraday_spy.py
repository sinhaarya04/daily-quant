"""Day 026 — Overnight vs Intraday Returns (SPY)

Self-contained script:
- Fetch SPY OHLCV from yfinance.
- Decompose total daily returns into overnight and intraday components.
- Summarize statistics and write a CSV + SVG plot.

Run:
  python day-026-overnight-vs-intraday-spy/overnight_vs_intraday_spy.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


@dataclass
class Config:
    ticker: str = "SPY"
    start: str = "2000-01-01"
    end: str | None = None
    out_dir: str = os.path.join(os.path.dirname(__file__), "output")


def download_ohlc(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def safe_stats(x: pd.Series) -> dict[str, float]:
    x = x.dropna()
    if len(x) < 5:
        return {
            "n": float(len(x)),
            "mean": np.nan,
            "vol": np.nan,
            "sharpe_like": np.nan,
            "hit_rate": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
        }
    mu = x.mean()
    vol = x.std(ddof=1)
    sharpe_like = (mu / vol) * np.sqrt(252.0) if vol > 0 else np.nan
    return {
        "n": float(len(x)),
        "mean": float(mu),
        "vol": float(vol),
        "sharpe_like": float(sharpe_like),
        "hit_rate": float((x > 0).mean()),
        "skew": float(x.skew()),
        "kurtosis": float(x.kurt()),
    }


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    ohlc = download_ohlc(cfg.ticker, cfg.start, cfg.end)

    # Returns decomposition
    df = ohlc.copy()
    df["PrevClose"] = df["Close"].shift(1)

    df["r_overnight"] = df["Open"] / df["PrevClose"] - 1.0
    df["r_intraday"] = df["Close"] / df["Open"] - 1.0
    df["r_total"] = df["Close"] / df["PrevClose"] - 1.0

    # Clean: require finite returns and positive prices
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_overnight", "r_intraday", "r_total"])
    df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["PrevClose"] > 0)]

    # Cumulative return paths (log-additive)
    df["cum_total"] = np.exp(np.log1p(df["r_total"]).cumsum())
    df["cum_overnight"] = np.exp(np.log1p(df["r_overnight"]).cumsum())
    df["cum_intraday"] = np.exp(np.log1p(df["r_intraday"]).cumsum())

    out_csv = os.path.join(cfg.out_dir, "spy_overnight_intraday.csv")
    df[["Open", "Close", "Volume", "r_overnight", "r_intraday", "r_total"]].to_csv(out_csv, index=True)

    # Plot to SVG (tracked; PNGs are gitignored in this repo)
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df.index, df["cum_total"], label="Total (Close→Close)", color="black", lw=1.5)
    ax.plot(df.index, df["cum_overnight"], label="Overnight (PrevClose→Open)", lw=1.2)
    ax.plot(df.index, df["cum_intraday"], label="Intraday (Open→Close)", lw=1.2)
    ax.set_title(f"{cfg.ticker}: cumulative returns decomposition")
    ax.set_ylabel("Growth of $1 (log-compounded)")
    ax.legend(loc="best")
    fig.tight_layout()

    out_svg = os.path.join(cfg.out_dir, "cum_returns.svg")
    fig.savefig(out_svg)
    plt.close(fig)

    # Console summary
    stats = pd.DataFrame(
        {
            "overnight": safe_stats(df["r_overnight"]),
            "intraday": safe_stats(df["r_intraday"]),
            "total": safe_stats(df["r_total"]),
        }
    )

    # Annualize mean (approx) for interpretability
    ann_mean = pd.Series(
        {
            "overnight": df["r_overnight"].mean() * 252.0,
            "intraday": df["r_intraday"].mean() * 252.0,
            "total": df["r_total"].mean() * 252.0,
        },
        name="ann_mean_simple",
    )

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    print("\n=== Summary stats (daily returns) ===")
    print(stats)
    print("\n=== Approx annualized simple mean (daily mean * 252) ===")
    print(ann_mean)
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_svg}")


if __name__ == "__main__":
    main()
