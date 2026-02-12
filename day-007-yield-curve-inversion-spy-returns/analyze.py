"""Day 007 — Yield Curve Inversion vs Forward SPY Returns

Pulls the 10Y-2Y term spread from FRED and SPY prices from Yahoo Finance,
then compares forward 12-month SPY returns conditional on the spread being
negative (inversion) vs non-negative.

Run:
  python day-007-yield-curve-inversion-spy-returns/analyze.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


FRED_T10Y2Y_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"


@dataclass
class Config:
    start: str = "1990-01-01"
    horizon_months: int = 12
    ticker: str = "SPY"
    out_dir: str = os.path.join(
        "day-007-yield-curve-inversion-spy-returns", "outputs"
    )


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_t10y2y(start: str) -> pd.Series:
    """Load 10Y-2Y spread from FRED CSV endpoint (daily, percent points)."""
    df = pd.read_csv(FRED_T10Y2Y_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()

    # FRED uses '.' for missing
    s = pd.to_numeric(df["T10Y2Y"], errors="coerce").dropna()
    s = s.loc[pd.to_datetime(start) :]
    s.name = "t10y2y"
    return s


def load_spy_adj_close(start: str, ticker: str) -> pd.Series:
    """Load adjusted close prices via yfinance."""
    px = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if px.empty:
        raise RuntimeError("No price data returned from yfinance.")
    s = px["Adj Close"].dropna().copy()
    s.name = "spy_adj_close" if ticker.upper() == "SPY" else f"{ticker}_adj_close"
    return s


def to_month_end_last(s: pd.Series) -> pd.Series:
    """Resample to calendar month-end, taking last observation in the month."""
    s = s.sort_index()
    m = s.resample("M").last().dropna()
    m.index = m.index.to_period("M").to_timestamp("M")
    return m


def forward_return(prices: pd.Series, horizon_months: int) -> pd.Series:
    """Forward total return over horizon_months, using month-end prices."""
    fwd = prices.shift(-horizon_months) / prices - 1.0
    fwd.name = f"fwd_{horizon_months}m_ret"
    return fwd


def summarize_by_regime(df: pd.DataFrame, regime_col: str, ret_col: str) -> pd.DataFrame:
    g = df.dropna(subset=[regime_col, ret_col]).groupby(regime_col)[ret_col]
    out = pd.DataFrame(
        {
            "count": g.count(),
            "mean": g.mean(),
            "median": g.median(),
            "std": g.std(),
            "p10": g.quantile(0.10),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
            "p90": g.quantile(0.90),
        }
    )
    return out


def main() -> None:
    cfg = Config()
    ensure_out_dir(cfg.out_dir)

    spread_d = load_t10y2y(cfg.start)
    spy_d = load_spy_adj_close(cfg.start, cfg.ticker)

    spread_m = to_month_end_last(spread_d)
    spy_m = to_month_end_last(spy_d)

    fwd = forward_return(spy_m, cfg.horizon_months)

    df = pd.concat([spread_m, spy_m, fwd], axis=1)
    df["inversion"] = df["t10y2y"] < 0

    # Persist dataset
    csv_path = os.path.join(cfg.out_dir, "monthly_dataset.csv")
    df.to_csv(csv_path, index_label="date")

    # Summary
    summary = summarize_by_regime(df, "inversion", fwd.name)
    summary_path = os.path.join(cfg.out_dir, "summary.csv")
    summary.to_csv(summary_path)

    print("== Yield curve inversion vs forward returns ==")
    print(f"Monthly observations: {df.dropna().shape[0]}")
    print(f"Horizon: {cfg.horizon_months} months")
    print("\nSummary of forward returns (decimal):")
    print(summary.round(4))

    # Plot 1: Spread over time
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df["t10y2y"], lw=1.2, color="black")
    plt.axhline(0, color="red", lw=1, alpha=0.8)
    plt.title("FRED T10Y2Y (10Y–2Y term spread)")
    plt.ylabel("percentage points")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "t10y2y_timeseries.png"), dpi=160)
    plt.close()

    # Plot 2: Conditional distributions (hist)
    data_inv = df.loc[df["inversion"], fwd.name].dropna()
    data_norm = df.loc[~df["inversion"], fwd.name].dropna()

    plt.figure(figsize=(11, 4))
    bins = np.linspace(-0.6, 0.8, 40)
    plt.hist(data_norm, bins=bins, alpha=0.55, label="Spread >= 0", color="#4C78A8", density=True)
    plt.hist(data_inv, bins=bins, alpha=0.55, label="Spread < 0 (Inversion)", color="#F58518", density=True)
    plt.axvline(0, color="black", lw=1)
    plt.title(f"Forward {cfg.horizon_months}M SPY returns (monthly, conditional)")
    plt.xlabel("forward return")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "forward_return_hist.png"), dpi=160)
    plt.close()

    # Plot 3: Mean forward return by regime
    means = (
        df.dropna(subset=[fwd.name])
        .groupby("inversion")[fwd.name]
        .mean()
        .rename(index={False: "Spread >= 0", True: "Spread < 0"})
    )

    plt.figure(figsize=(6, 4))
    plt.bar(means.index, means.values, color=["#4C78A8", "#F58518"])
    plt.axhline(0, color="black", lw=1)
    plt.title(f"Mean forward {cfg.horizon_months}M return")
    plt.ylabel("mean return")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "mean_forward_return.png"), dpi=160)
    plt.close()

    print(f"\nWrote outputs to: {cfg.out_dir}")
    print(f"- {os.path.basename(csv_path)}")
    print(f"- {os.path.basename(summary_path)}")
    print("- t10y2y_timeseries.png")
    print("- forward_return_hist.png")
    print("- mean_forward_return.png")


if __name__ == "__main__":
    main()
