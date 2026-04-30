"""Day 024 — Real Rate Regimes (DGS10 − T10YIE) vs SPY

Self-contained script:
- Fetch DGS10 and T10YIE from FRED (no key) via fredgraph CSV.
- Fetch SPY from yfinance.
- Build a 10Y real-rate proxy and compare conditional SPY forward returns.

Run:
  python day-024-real-rate-regimes-spy/real_rate_regimes_spy.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@dataclass
class Config:
    start: str = "2003-01-01"  # SPY liquid era; also avoids early breakeven sparsity
    end: str | None = None
    fwd_days: int = 21  # ~1 month
    out_dir: str = os.path.join(os.path.dirname(__file__), "output")


def fetch_fred_series(series_id: str, start: str | None = None) -> pd.Series:
    """Fetch a FRED series (date,value) from fredgraph.csv.

    Returns a float Series indexed by Timestamp.
    Missing observations (".") are parsed as NaN.
    """
    url = FRED_CSV.format(series_id=series_id)
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or series_id.lower() not in df.columns:
        raise ValueError(f"Unexpected FRED CSV columns for {series_id}: {df.columns.tolist()}")

    s = df.set_index("date")[series_id.lower()]
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce")
    if start is not None:
        s = s.loc[pd.to_datetime(start) :]
    s.name = series_id
    return s


def fetch_spy(start: str, end: str | None) -> pd.Series:
    px = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.name = "SPY"
    return px


def fwd_return(px: pd.Series, days: int) -> pd.Series:
    return px.shift(-days) / px - 1.0


def annualized_vol(daily_rets: pd.Series, window: int = 21) -> pd.Series:
    return daily_rets.rolling(window).std() * np.sqrt(252.0)


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Rates (percent units)
    dgs10 = fetch_fred_series("DGS10", start=cfg.start)
    t10yie = fetch_fred_series("T10YIE", start=cfg.start)

    # Prices
    spy = fetch_spy(cfg.start, cfg.end)
    spy_ret = spy.pct_change()

    # Merge on dates: use business-day index from SPY, forward-fill rates (they can be missing on holidays)
    df = pd.DataFrame({"SPY": spy}).dropna()
    df["DGS10"] = dgs10.reindex(df.index).ffill()
    df["T10YIE"] = t10yie.reindex(df.index).ffill()

    # Real-rate proxy (percent)
    df["REAL10Y"] = df["DGS10"] - df["T10YIE"]

    # Targets: forward returns and realized vol
    df["FWD_RET_1M"] = fwd_return(df["SPY"], cfg.fwd_days)
    df["VOL_1M"] = annualized_vol(spy_ret.reindex(df.index), window=cfg.fwd_days)

    # Drop tail where forward return is NaN
    df = df.dropna(subset=["FWD_RET_1M", "REAL10Y", "VOL_1M"]).copy()

    # --- Regimes
    med = df["REAL10Y"].median()
    df["REG_MED"] = np.where(df["REAL10Y"] >= med, "High real rate", "Low real rate")

    # Quartiles (use qcut; drop duplicates if constant segments)
    try:
        df["REAL10Y_Q"] = pd.qcut(df["REAL10Y"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    except ValueError:
        # fallback: fewer bins
        df["REAL10Y_Q"] = pd.qcut(df["REAL10Y"].rank(method="average"), 4, labels=["Q1", "Q2", "Q3", "Q4"])

    # --- Summary stats
    def summarize(group_col: str) -> pd.DataFrame:
        g = df.groupby(group_col)
        out = pd.DataFrame(
            {
                "n": g.size(),
                "mean_fwd_ret": g["FWD_RET_1M"].mean(),
                "median_fwd_ret": g["FWD_RET_1M"].median(),
                "hit_rate": g["FWD_RET_1M"].apply(lambda x: (x > 0).mean()),
                "mean_vol": g["VOL_1M"].mean(),
            }
        )
        return out.sort_index()

    med_stats = summarize("REG_MED")
    q_stats = summarize("REAL10Y_Q")

    # Save stats to CSV
    med_stats.to_csv(os.path.join(cfg.out_dir, "stats_median_regime.csv"))
    q_stats.to_csv(os.path.join(cfg.out_dir, "stats_quartiles.csv"))

    # --- Plots
    plt.style.use("seaborn-v0_8")

    # 1) Time series of real-rate proxy and SPY
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(df.index, df["SPY"], color="black", lw=1.2)
    ax[0].set_title("SPY (auto-adjusted) and 10Y Real-Rate Proxy")
    ax[0].set_ylabel("SPY")

    ax[1].plot(df.index, df["REAL10Y"], color="tab:blue", lw=1.0)
    ax[1].axhline(med, color="tab:red", ls="--", lw=1.0, label=f"Median = {med:.2f}%")
    ax[1].set_ylabel("DGS10 − T10YIE (%)")
    ax[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "timeseries_spy_real10y.png"), dpi=160)
    plt.close(fig)

    # 2) Boxplot of forward returns by median regime
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = ["Low real rate", "High real rate"]
    data = [df.loc[df["REG_MED"] == k, "FWD_RET_1M"].values for k in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.axhline(0, color="gray", lw=1)
    ax.set_title(f"SPY forward {cfg.fwd_days}d returns by real-rate regime")
    ax.set_ylabel("Forward return")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "box_fwdret_by_regime.png"), dpi=160)
    plt.close(fig)

    # 3) Bar chart of mean/median by quartile
    q = q_stats.copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(q))
    ax.bar(x - 0.2, q["mean_fwd_ret"].values, width=0.4, label="Mean")
    ax.bar(x + 0.2, q["median_fwd_ret"].values, width=0.4, label="Median")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in q.index], rotation=0)
    ax.set_title(f"SPY forward {cfg.fwd_days}d returns by REAL10Y quartile")
    ax.set_ylabel("Forward return")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "bars_fwdret_by_quartile.png"), dpi=160)
    plt.close(fig)

    # Console summary
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    print("\n=== Median regime summary ===")
    print(med_stats)
    print("\n=== Quartile summary ===")
    print(q_stats)
    print(f"\nWrote outputs to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
