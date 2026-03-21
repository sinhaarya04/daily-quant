"""Day 13 — VIX vs Realized Vol (VRP proxy)

Fetch:
- VIXCLS from FRED (daily close)
- SPY from yfinance (daily close)

Compute:
- Realized volatility proxy = rolling std of daily log returns over `window` days,
  annualized by sqrt(252), expressed in *vol points* (e.g., 20 = 20%).
- VRP proxy = VIX - RealizedVol

This is intentionally lightweight and self-contained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


@dataclass
class Config:
    start: str
    end: str | None
    window: int
    out_png: str


def fetch_vix_fred() -> pd.Series:
    df = pd.read_csv(FRED_VIX_URL)

    # FRED CSVs are typically: observation_date,<SERIES_ID>
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # FRED uses '.' for missing values sometimes
    vix = pd.to_numeric(df["VIXCLS"], errors="coerce")
    vix.name = "VIX"
    return vix


def fetch_spy(start: str, end: str | None) -> pd.Series:
    df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No SPY data returned from yfinance.")

    close = df["Close"]
    # Depending on yfinance/pandas versions, this can be a Series or a 1-col DataFrame
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.copy()
    close.name = "SPY"
    return close


def realized_vol_points(close: pd.Series, window: int) -> pd.Series:
    logret = np.log(close).diff()
    rv = logret.rolling(window).std() * np.sqrt(252) * 100.0
    rv.name = f"RV_{window}d"
    return rv


def summarize(vix: pd.Series, rv: pd.Series, vrp: pd.Series) -> str:
    df = pd.concat([vix, rv, vrp], axis=1).dropna()
    out = []
    out.append(f"Sample: {df.index.min().date()} → {df.index.max().date()} ({len(df):,} obs)")
    out.append("")

    def line(name: str, s: pd.Series) -> str:
        return (
            f"{name:>10}: mean={s.mean():6.2f}  med={s.median():6.2f}  "
            f"p10={s.quantile(0.10):6.2f}  p90={s.quantile(0.90):6.2f}"
        )

    out.append(line("VIX", df["VIX"]))
    out.append(line(rv.name, df[rv.name]))
    out.append(line("VRP", df["VRP"]))
    out.append("")

    corr = df[["VIX", rv.name]].corr().iloc[0, 1]
    out.append(f"Corr(VIX, {rv.name}) = {corr:.3f}")

    # A simple diagnostic: how often is VIX above realized vol?
    pct_pos = (df["VRP"] > 0).mean() * 100.0
    out.append(f"P(VIX > realized) = {pct_pos:.1f}%")

    return "\n".join(out)


def plot(vix: pd.Series, rv: pd.Series, vrp: pd.Series, out_png: str):
    df = pd.concat([vix, rv, vrp], axis=1).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df.index, df["VIX"], label="VIX (FRED)", lw=1.2)
    ax.plot(df.index, df[rv.name], label=f"Realized vol ({rv.name.replace('RV_', '').replace('d', 'd')})", lw=1.2)
    ax.set_ylabel("Vol points")
    ax.set_title("VIX vs realized volatility (proxy)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df.index, df["VRP"], color="purple", lw=1.0)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("VRP (VIX - RV)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2004-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD), optional")
    ap.add_argument("--window", type=int, default=21, help="Rolling window (trading days) for realized vol")
    ap.add_argument(
        "--out",
        default="day-013-vix-vs-realized-vol-vrp/vrp_timeseries.png",
        help="Output PNG path",
    )
    args = ap.parse_args()

    cfg = Config(start=args.start, end=args.end, window=args.window, out_png=args.out)

    vix = fetch_vix_fred()
    spy = fetch_spy(cfg.start, cfg.end)

    # Align to business days where both series exist.
    vix = vix.loc[cfg.start : cfg.end]
    rv = realized_vol_points(spy.loc[cfg.start : cfg.end], cfg.window)
    rv_col = rv.name or f"RV_{cfg.window}d"

    df = pd.DataFrame({"VIX": vix, rv_col: rv}).dropna()
    df["VRP"] = df["VIX"] - df[rv_col]

    print(summarize(df["VIX"], df[rv_col], df["VRP"]))
    plot(df["VIX"], df[rv_col], df["VRP"], cfg.out_png)
    print(f"\nWrote: {cfg.out_png}")


if __name__ == "__main__":
    main()
