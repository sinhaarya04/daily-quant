"""Day 020 — Drawdowns & recovery time stats.

Self-contained script:
- downloads daily prices via yfinance
- computes drawdown series
- segments drawdown episodes (peak -> trough -> recovery)
- prints summary stats and saves plots/CSV

Run:
  python day-020-drawdown-recovery-stats/analyze.py --ticker SPY --start 1993-01-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Episode:
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    max_drawdown: float
    days_peak_to_trough: int
    days_trough_to_recovery: int | None
    days_peak_to_recovery: int | None


def download_prices(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    px = df[col].dropna().copy()
    px.name = ticker
    return px


def compute_drawdown(px: pd.Series) -> pd.DataFrame:
    running_max = px.cummax()
    dd = px / running_max - 1.0
    out = pd.DataFrame({"price": px, "running_max": running_max, "drawdown": dd})
    return out


def segment_episodes(dd: pd.Series) -> list[Episode]:
    """Segment drawdown episodes.

    Episode definition:
    - peak_date: last date at which drawdown == 0 before going negative
    - trough_date: date of minimum drawdown within the episode
    - recovery_date: first date after peak where drawdown returns to 0 (new high). None if unrecovered.

    We treat exact zeros (within float tolerance) as "at high".
    """

    s = dd.copy()
    s = s.fillna(0.0)

    at_high = np.isclose(s.values, 0.0, atol=1e-12)

    episodes: list[Episode] = []

    i = 0
    n = len(s)
    idx = s.index

    while i < n:
        # Find next start: transition from at_high -> drawdown < 0
        while i < n and at_high[i]:
            i += 1
        if i >= n:
            break

        # We are inside a drawdown; peak is previous index (or same if starts immediately)
        peak_i = max(i - 1, 0)
        peak_date = idx[peak_i]

        # Move until recovery (back to at_high) or end
        j = i
        trough_i = i
        trough_dd = s.iloc[i]

        while j < n and not at_high[j]:
            if s.iloc[j] < trough_dd:
                trough_dd = s.iloc[j]
                trough_i = j
            j += 1

        trough_date = idx[trough_i]
        max_dd = float(trough_dd)

        recovery_date = None
        if j < n and at_high[j]:
            recovery_date = idx[j]

        days_peak_to_trough = int((trough_date - peak_date).days)
        days_trough_to_recovery = None
        days_peak_to_recovery = None
        if recovery_date is not None:
            days_trough_to_recovery = int((recovery_date - trough_date).days)
            days_peak_to_recovery = int((recovery_date - peak_date).days)

        episodes.append(
            Episode(
                peak_date=peak_date,
                trough_date=trough_date,
                recovery_date=recovery_date,
                max_drawdown=max_dd,
                days_peak_to_trough=days_peak_to_trough,
                days_trough_to_recovery=days_trough_to_recovery,
                days_peak_to_recovery=days_peak_to_recovery,
            )
        )

        i = j + 1  # continue search after recovery day

    return episodes


def episodes_to_frame(episodes: list[Episode]) -> pd.DataFrame:
    rows = []
    for e in episodes:
        rows.append(
            {
                "peak_date": e.peak_date,
                "trough_date": e.trough_date,
                "recovery_date": e.recovery_date,
                "max_drawdown": e.max_drawdown,
                "days_peak_to_trough": e.days_peak_to_trough,
                "days_trough_to_recovery": e.days_trough_to_recovery,
                "days_peak_to_recovery": e.days_peak_to_recovery,
                "recovered": e.recovery_date is not None,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("max_drawdown")  # most negative first
        df.reset_index(drop=True, inplace=True)
    return df


def save_plots(df: pd.DataFrame, episodes_df: pd.DataFrame, ticker: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: price + running max
    fig, ax = plt.subplots(figsize=(11, 5))
    df["price"].plot(ax=ax, lw=1.2, label="price")
    df["running_max"].plot(ax=ax, lw=1.0, alpha=0.8, label="running max")
    ax.set_title(f"{ticker} — Price and running max")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "price_running_max.png", dpi=160)
    plt.close(fig)

    # Plot 2: underwater drawdown chart
    fig, ax = plt.subplots(figsize=(11, 4))
    (df["drawdown"] * 100).plot(ax=ax, lw=1.0, color="tab:red")
    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"{ticker} — Drawdown (underwater chart)")
    ax.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    fig.savefig(outdir / "drawdown_underwater.png", dpi=160)
    plt.close(fig)

    # Plot 3: histogram of recovery times (peak -> recovery)
    rec = episodes_df.loc[episodes_df["recovered"], "days_peak_to_recovery"].dropna()
    if len(rec) >= 3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(rec.values, bins=20, color="tab:blue", alpha=0.85)
        ax.set_title(f"{ticker} — Recovery time distribution (peak → recovery)")
        ax.set_xlabel("Days")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(outdir / "recovery_time_hist.png", dpi=160)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, default="1993-01-01")
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Output directory for plots and CSV.",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)

    px = download_prices(args.ticker, args.start)
    df = compute_drawdown(px)

    episodes = segment_episodes(df["drawdown"])
    episodes_df = episodes_to_frame(episodes)

    # Basic stats
    print(f"Ticker: {args.ticker}")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()} ({len(df):,} bars)")

    if episodes_df.empty:
        print("No drawdown episodes detected (unexpected).")
        return

    worst = episodes_df.iloc[0]
    print("\nWorst drawdown episode:")
    print(
        f"  peak={worst['peak_date'].date()} trough={worst['trough_date'].date()} "
        f"max_dd={worst['max_drawdown']:.2%} recovered={bool(worst['recovered'])}"
    )
    if bool(worst["recovered"]):
        print(f"  recovery={worst['recovery_date'].date()}  days_peak_to_recovery={int(worst['days_peak_to_recovery'])}")

    rec = episodes_df.loc[episodes_df["recovered"], "days_peak_to_recovery"].dropna()
    if len(rec) > 0:
        print("\nRecovery time (peak → recovery) summary (recovered episodes only):")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            print(f"  p{int(q*100):02d}: {np.quantile(rec.values, q):.0f} days")
        print(f"  mean: {rec.mean():.0f} days")

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    episodes_df.to_csv(outdir / f"{args.ticker}_drawdown_episodes.csv", index=False)

    print("\nTop 10 drawdowns by depth:")
    display = episodes_df.head(10).copy()
    display["max_drawdown"] = (display["max_drawdown"] * 100).round(2)
    print(display[["peak_date", "trough_date", "recovery_date", "max_drawdown", "days_peak_to_recovery", "recovered"]].to_string(index=False))

    save_plots(df, episodes_df, args.ticker, outdir)
    print(f"\nWrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
