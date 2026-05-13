"""Day 030 — Rolling Hurst Exponent (DFA) on SPY.

Self-contained script:
- pulls SPY daily close with yfinance
- estimates rolling Hurst exponent via DFA on log-prices
- checks whether next-day returns differ across H-regimes

Run:
  pip install -r requirements.txt
  python day-030-rolling-hurst-dfa/main.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


@dataclass
class DFAConfig:
    window: int = 252  # ~1Y
    min_scale: int = 8
    max_scale: int = 64
    n_scales: int = 10
    poly_deg: int = 1


def dfa_hurst(x: np.ndarray, *, min_scale: int, max_scale: int, n_scales: int, poly_deg: int = 1) -> float:
    """Estimate H via detrended fluctuation analysis.

    Steps:
    - integrate demeaned series
    - for each scale s: break into segments, detrend each segment, compute RMS fluctuation
    - regress log(F(s)) ~ log(s); slope is H

    Notes:
    - Uses non-overlapping segments (fast + simple)
    - Returns nan if insufficient data
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < max_scale * 2:
        return float("nan")

    y = np.cumsum(x - np.mean(x))

    # log-spaced integer scales
    scales = np.unique(
        np.round(np.exp(np.linspace(math.log(min_scale), math.log(max_scale), n_scales))).astype(int)
    )
    scales = scales[(scales >= 2) & (scales <= n // 2)]
    if scales.size < 2:
        return float("nan")

    Fs = []
    Ss = []
    for s in scales:
        k = n // s
        if k < 2:
            continue
        y_use = y[: k * s]
        segs = y_use.reshape(k, s)

        # detrend each segment with polynomial fit
        t = np.arange(s)
        rms = []
        for seg in segs:
            coef = np.polyfit(t, seg, deg=poly_deg)
            trend = np.polyval(coef, t)
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))

        F = float(np.sqrt(np.mean(np.square(rms))))
        if np.isfinite(F) and F > 0:
            Fs.append(F)
            Ss.append(s)

    if len(Ss) < 2:
        return float("nan")

    logS = np.log(np.asarray(Ss))
    logF = np.log(np.asarray(Fs))

    # slope via OLS
    slope = float(np.polyfit(logS, logF, deg=1)[0])
    return slope


def rolling_dfa(series: pd.Series, cfg: DFAConfig) -> pd.Series:
    xs = series.to_numpy(dtype=float)
    idx = series.index

    out = np.full(xs.shape, np.nan, dtype=float)
    for i in range(cfg.window - 1, len(xs)):
        window_x = xs[i - cfg.window + 1 : i + 1]
        out[i] = dfa_hurst(
            window_x,
            min_scale=cfg.min_scale,
            max_scale=cfg.max_scale,
            n_scales=cfg.n_scales,
            poly_deg=cfg.poly_deg,
        )

    return pd.Series(out, index=idx, name="hurst")


def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    cfg = DFAConfig()
    out_dir = ensure_out_dir()

    df = yf.download("SPY", start="2000-01-01", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance.")

    # yfinance sometimes returns MultiIndex columns: (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        px = df[("Close", "SPY")].dropna()
    else:
        px = df["Close"].dropna()
    px = px.rename("close")
    log_px = np.log(px)
    ret1 = px.pct_change().rename("ret1")

    hurst = rolling_dfa(log_px, cfg=cfg)

    data = pd.concat([px, ret1, hurst], axis=1).dropna()

    # regime check: bucket by H quantiles
    q = data["hurst"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()

    def bucket(h: float) -> str:
        if h <= q[0]:
            return "H lowest 20%"
        if h <= q[1]:
            return "H 20-40%"
        if h <= q[2]:
            return "H 40-60%"
        if h <= q[3]:
            return "H 60-80%"
        return "H top 20%"

    data["bucket"] = data["hurst"].map(bucket)

    # next-day returns by today's H bucket
    data["ret1_fwd"] = data["ret1"].shift(-1)
    summary = (
        data.dropna(subset=["ret1_fwd"])
        .groupby("bucket")["ret1_fwd"]
        .agg(n="count", mean="mean", vol="std")
        .sort_index()
    )
    summary["sharpe_like"] = summary["mean"] / summary["vol"]

    print("\nRolling DFA Hurst (SPY, log-price)\n")
    print(f"Window={cfg.window} days | scales=[{cfg.min_scale}, {cfg.max_scale}] | poly_deg={cfg.poly_deg}")
    print("\nNext-day returns conditioned on today's H bucket:")
    print(summary.to_string(float_format=lambda x: f"{x: .6f}"))

    # plots
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data.index, data["hurst"], lw=1)
    ax.axhline(0.5, color="black", ls="--", lw=1, alpha=0.7)
    ax.set_title("SPY Rolling Hurst Exponent (DFA on log-price)")
    ax.set_ylabel("H")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rolling_hurst.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ordered = ["H lowest 20%", "H 20-40%", "H 40-60%", "H 60-80%", "H top 20%"]
    means = summary.loc[ordered, "mean"]
    ax.bar(range(len(ordered)), means.to_numpy())
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=20, ha="right")
    ax.set_title("Mean next-day return by H bucket")
    ax.set_ylabel("mean(ret(t+1))")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "next_day_return_by_bucket.png"), dpi=160)
    plt.close(fig)

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()
