"""Day 028 — PCA sector risk model + walk-forward min-var.

Self-contained script:
- Pulls ETF prices (sector SPDRs)
- Computes daily returns
- Walk-forward monthly rebalancing
- Compares min-var using sample covariance vs PCA-reconstructed covariance

Run:
  python day-028-pca-sector-risk-model/pca_minvar_backtest.py --start 2006-01-01 --lookback 756 --k 3 5 8
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


SECTOR_ETFS = [
    "XLB",  # Materials
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Staples
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Discretionary
]


@dataclass
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame


def annualized_return(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    return (1.0 + r).prod() ** (252.0 / len(r)) - 1.0


def annualized_vol(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    return r.std(ddof=1) * np.sqrt(252.0)


def sharpe(daily_returns: pd.Series) -> float:
    vol = annualized_vol(daily_returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualized_return(daily_returns) / vol


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) == 0:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()


def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    # Keep only tickers that exist (some older start dates can cause missing columns)
    df = df.dropna(axis=1, how="all")
    return df


def pca_covariance(sample_cov: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct covariance using top-k eigencomponents + diagonal residual.

    Let S = Q Λ Q'. Keep top-k eigenpairs. Residual (discarded) variance is
    put back on the diagonal so total variances match approximately.
    """
    # Eigen-decomposition for symmetric PSD
    evals, evecs = np.linalg.eigh(sample_cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    k = int(k)
    k = max(0, min(k, len(evals)))

    if k == 0:
        # Pure diagonal model
        return np.diag(np.diag(sample_cov))

    Qk = evecs[:, :k]
    Lk = np.diag(evals[:k])
    cov_k = Qk @ Lk @ Qk.T

    # Add residual variances to diagonal so diag matches sample_cov diag
    diag_resid = np.clip(np.diag(sample_cov) - np.diag(cov_k), 0.0, None)
    cov = cov_k + np.diag(diag_resid)

    # Symmetrize (numerical)
    cov = 0.5 * (cov + cov.T)
    return cov


def min_var_weights(cov: np.ndarray) -> np.ndarray:
    """Unconstrained minimum-variance weights with sum(w)=1.

    w* = (Σ^{-1} 1) / (1' Σ^{-1} 1)
    """
    n = cov.shape[0]
    ones = np.ones(n)

    # Small ridge for numerical stability
    ridge = 1e-8 * np.trace(cov) / n
    cov_reg = cov + ridge * np.eye(n)

    inv = np.linalg.inv(cov_reg)
    w = inv @ ones
    w = w / (ones @ inv @ ones)
    return w


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day in each month (based on the data index)."""
    df = pd.DataFrame(index=index)
    last = df.groupby([df.index.year, df.index.month]).tail(1).index
    return pd.DatetimeIndex(last)


def walk_forward_minvar(
    rets: pd.DataFrame,
    lookback: int,
    cov_builder,
) -> BacktestResult:
    dates = rets.index
    rebals = month_ends(dates)

    w_records = []
    port_rets = pd.Series(index=dates, dtype=float)

    for d in rebals:
        loc = dates.get_loc(d)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < lookback:
            continue

        train = rets.iloc[loc - lookback : loc]
        test_start = loc

        # Next rebalance date (exclusive end)
        next_i = np.searchsorted(rebals.values, d.to_datetime64(), side="right")
        if next_i >= len(rebals):
            test_end = len(rets)
        else:
            test_end = dates.get_loc(rebals[next_i]) + 1

        sample_cov = train.cov().values
        cov = cov_builder(sample_cov)
        w = min_var_weights(cov)

        w_records.append(pd.Series(w, index=rets.columns, name=d))

        # Apply weights over the holding period
        hold = rets.iloc[test_start:test_end]
        port_rets.iloc[test_start:test_end] = hold.values @ w

    weights = pd.DataFrame(w_records)
    equity = (1.0 + port_rets.fillna(0.0)).cumprod()
    return BacktestResult(equity=equity, weights=weights)


def summarize(label: str, equity: pd.Series) -> Dict[str, float]:
    daily = equity.pct_change().dropna()
    return {
        "label": label,
        "ann_return": annualized_return(daily),
        "ann_vol": annualized_vol(daily),
        "sharpe": sharpe(daily),
        "max_dd": max_drawdown(equity),
    }


def plot_equity(curves: Dict[str, pd.Series], outpath: str) -> None:
    plt.figure(figsize=(10, 5))
    for name, eq in curves.items():
        plt.plot(eq.index, eq.values, label=name)
    plt.title("Walk-forward min-var: sample covariance vs PCA covariance")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_weights(weights: pd.DataFrame, outpath: str) -> None:
    if weights.empty:
        return
    w = weights.copy()
    # Stackplot needs numpy arrays
    plt.figure(figsize=(10, 5))
    plt.stackplot(w.index, w.T.values, labels=w.columns, alpha=0.9)
    plt.title("Min-var weights over time (unconstrained)")
    plt.ylabel("Weight")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", ncols=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2006-01-01")
    ap.add_argument("--lookback", type=int, default=756, help="Trading days (~3y)")
    ap.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=[3, 5, 8],
        help="PCA component counts to test",
    )
    ap.add_argument("--tickers", type=str, nargs="*", default=SECTOR_ETFS)
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)

    prices = fetch_prices(args.tickers, start=args.start)
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    curves: Dict[str, pd.Series] = {}
    stats: List[Dict[str, float]] = []

    # Baseline: sample covariance
    res_sample = walk_forward_minvar(rets, args.lookback, cov_builder=lambda S: S)
    curves["SampleCov"] = res_sample.equity
    stats.append(summarize("SampleCov", res_sample.equity))

    for k in args.k:
        res = walk_forward_minvar(rets, args.lookback, cov_builder=lambda S, kk=k: pca_covariance(S, kk))
        name = f"PCA(k={k})"
        curves[name] = res.equity
        stats.append(summarize(name, res.equity))

        plot_weights(
            res.weights,
            os.path.join(outdir, f"weights_pca_k{k}.png"),
        )

    plot_equity(curves, os.path.join(outdir, "equity_curves.png"))

    st = pd.DataFrame(stats).set_index("label")
    st = st[["ann_return", "ann_vol", "sharpe", "max_dd"]].sort_index()

    print("Tickers:", list(prices.columns))
    print(f"Data range: {rets.index.min().date()} → {rets.index.max().date()}  (n={len(rets)})")
    print("\nPerformance summary (walk-forward):")
    print((st * pd.Series({"ann_return": 100, "ann_vol": 100, "sharpe": 1, "max_dd": 100})).round(2))
    print("\nSaved plots to:", outdir)


if __name__ == "__main__":
    main()
