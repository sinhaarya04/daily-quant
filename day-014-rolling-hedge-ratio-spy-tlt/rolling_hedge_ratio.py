"""Day 014 — Rolling Hedge Ratio (SPY ↔ TLT)

Self-contained script:
- Pulls SPY and TLT via yfinance
- Computes log returns
- Estimates rolling OLS beta (no statsmodels)
- Builds a hedged return series and compares performance

Run:
  python day-014-rolling-hedge-ratio-spy-tlt/rolling_hedge_ratio.py --start 2006-01-01 --window 60
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


@dataclass
class Perf:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float


def _to_dt(s: str | None) -> str:
    if s is None:
        return str(date.today())
    return s


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns different column layouts for single vs multi ticker.
    if len(tickers) == 1:
        px = df["Close"].to_frame(tickers[0])
    else:
        px = pd.concat({t: df[t]["Close"] for t in tickers}, axis=1)
        px.columns = tickers
    return px.dropna(how="any")


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna()


def rolling_beta_ols(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS slope with intercept.

    Model: y = a + b x + eps

    Returns an array aligned to y, with NaNs for the first (window-1) points.
    """
    n = len(y)
    betas = np.full(n, np.nan, dtype=float)
    if window < 5:
        raise ValueError("window too small; use at least 5")
    if n < window:
        raise ValueError("not enough data for the chosen window")

    for t in range(window - 1, n):
        ys = y[t - window + 1 : t + 1]
        xs = x[t - window + 1 : t + 1]
        X = np.column_stack([np.ones(window), xs])
        # Solve min ||Xb - y||
        b, *_ = np.linalg.lstsq(X, ys, rcond=None)
        betas[t] = b[1]
    return betas


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def perf_stats(r: pd.Series, periods_per_year: int = 252) -> Perf:
    r = r.dropna()
    mu = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = np.nan if vol == 0 else mu / vol
    curve = equity_curve(r)
    return Perf(
        ann_return=float(mu),
        ann_vol=float(vol),
        sharpe=float(sharpe),
        max_dd=max_drawdown(curve),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2006-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="day-014-rolling-hedge-ratio-spy-tlt/outputs")
    args = ap.parse_args()

    start, end = args.start, _to_dt(args.end)
    window = int(args.window)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tickers = ["SPY", "TLT"]
    px = download_prices(tickers, start=start, end=end)
    rets = log_returns(px)

    spy = rets["SPY"].astype(float)
    tlt = rets["TLT"].astype(float)

    # Rolling beta of SPY returns on TLT returns
    beta_roll = pd.Series(
        rolling_beta_ols(spy.values, tlt.values, window=window),
        index=rets.index,
        name="beta_roll",
    )

    # Hedged series: r_spy - beta_t * r_tlt
    hedged_roll = (spy - beta_roll * tlt).rename("hedged_roll")

    # Static hedge ratio from full sample (with intercept)
    X_full = np.column_stack([np.ones(len(spy)), tlt.values])
    b_full, *_ = np.linalg.lstsq(X_full, spy.values, rcond=None)
    beta_static = float(b_full[1])
    hedged_static = (spy - beta_static * tlt).rename("hedged_static")

    # Align (rolling has NaNs at start)
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt"), beta_roll, hedged_roll, hedged_static], axis=1)

    # Print stats on overlapping period where rolling hedge exists
    sample = df.dropna()
    stats_spy = perf_stats(sample["spy"].pipe(np.expm1))  # convert log-return to simple return
    stats_hedge_roll = perf_stats(sample["hedged_roll"].pipe(np.expm1))
    stats_hedge_static = perf_stats(sample["hedged_static"].pipe(np.expm1))

    def fmt(p: Perf) -> str:
        return (
            f"ann_ret={p.ann_return:6.2%}  ann_vol={p.ann_vol:6.2%}  "
            f"sharpe={p.sharpe:5.2f}  maxDD={p.max_dd:6.2%}"
        )

    print(f"Sample: {sample.index.min().date()} → {sample.index.max().date()} (n={len(sample)})")
    print(f"Static beta (SPY~TLT): {beta_static:.3f}")
    print(f"SPY           : {fmt(stats_spy)}")
    print(f"Hedged (roll) : {fmt(stats_hedge_roll)}")
    print(f"Hedged (static): {fmt(stats_hedge_static)}")

    # --- Plots ---
    plt.style.use("seaborn-v0_8")

    # Rolling beta
    fig, ax = plt.subplots(figsize=(10, 4))
    beta_roll.plot(ax=ax, lw=1.2)
    ax.axhline(beta_static, color="black", ls="--", lw=1.0, label=f"static beta={beta_static:.2f}")
    ax.set_title(f"Rolling Hedge Ratio: beta_t in SPY_t = a + beta_t * TLT_t (window={window})")
    ax.set_ylabel("beta")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "rolling_beta.png", dpi=160)
    plt.close(fig)

    # Equity curves (simple returns)
    eq = pd.DataFrame(
        {
            "SPY": equity_curve(np.expm1(sample["spy"])),
            "Hedged (rolling beta)": equity_curve(np.expm1(sample["hedged_roll"])),
            "Hedged (static beta)": equity_curve(np.expm1(sample["hedged_static"])),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    eq.plot(ax=ax, lw=1.3)
    ax.set_title("Equity Curves (Normalized to 1.0)")
    ax.set_ylabel("growth of $1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "equity_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
