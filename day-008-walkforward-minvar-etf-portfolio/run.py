"""Day 008 — Walk-forward minimum-variance ETF portfolio.

Monthly rebalance into a trailing-covariance minimum-variance portfolio.
- Closed-form unconstrained min-var weights
- Long-only projection via clip-to-zero + renormalize

Compares against equal-weight.

Dependencies: numpy, pandas, yfinance, matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class BacktestConfig:
    tickers: tuple[str, ...] = ("SPY", "QQQ", "IEF", "GLD")
    start: str = "2015-01-01"
    end: str | None = None
    lookback_days: int = 252
    rebalance: str = "M"  # month-end
    initial_value: float = 1.0


def download_prices(tickers: tuple[str, ...], start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns either a 1-level column index (single ticker) or multi-index.
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].copy()
        px.columns = [tickers[0]]

    px = px.dropna(how="all")
    return px


def ensure_prices_cache(cfg: BacktestConfig, folder: Path) -> pd.DataFrame:
    cache = folder / "prices.csv"
    if cache.exists():
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        # Ensure expected tickers exist; otherwise re-download.
        if set(cfg.tickers).issubset(set(px.columns)):
            return px[list(cfg.tickers)].dropna(how="all")

    px = download_prices(cfg.tickers, cfg.start, cfg.end)
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(cache)
    return px


def min_var_weights(cov: np.ndarray) -> np.ndarray:
    """Unconstrained minimum-variance weights: w ∝ inv(C) 1."""
    n = cov.shape[0]
    ones = np.ones(n)

    # Numerical safety: use pseudo-inverse if covariance is near-singular.
    inv = np.linalg.pinv(cov)
    w = inv @ ones
    denom = float(ones.T @ inv @ ones)
    if denom <= 0 or not np.isfinite(denom):
        return np.ones(n) / n
    w = w / denom
    return w


def long_only_project(w: np.ndarray, floor: float = 0.0) -> np.ndarray:
    w2 = np.maximum(w, floor)
    s = float(w2.sum())
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(w2) / len(w2)
    return w2 / s


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(daily_rets: pd.Series, equity: pd.Series) -> dict[str, float]:
    ann = 252
    mu = float(daily_rets.mean()) * ann
    vol = float(daily_rets.std(ddof=1)) * math.sqrt(ann)
    sharpe = mu / vol if vol > 0 else float("nan")

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")

    mdd = max_drawdown(equity)
    return {"CAGR": cagr, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": mdd}


def backtest(cfg: BacktestConfig, prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prices = prices.dropna().copy()
    rets = prices.pct_change().dropna()

    # Month-end rebalance dates (using returns index).
    rebal_dates = rets.resample(cfg.rebalance).last().index

    w_minvar = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    w_equal = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)

    n = rets.shape[1]
    equal = np.ones(n) / n

    for dt in rebal_dates:
        loc = rets.index.get_indexer([dt], method="ffill")[0]
        if loc < cfg.lookback_days:
            continue

        window = rets.iloc[loc - cfg.lookback_days : loc]
        cov = window.cov().values
        w = long_only_project(min_var_weights(cov))

        w_minvar.iloc[loc] = w
        w_equal.iloc[loc] = equal

    # Forward-fill weights between rebalance points; start when first weights appear.
    w_minvar = w_minvar.ffill().dropna()
    w_equal = w_equal.ffill().dropna()

    # Align returns to weights.
    rets2 = rets.loc[w_minvar.index]

    port_minvar = (rets2 * w_minvar).sum(axis=1)
    port_equal = (rets2 * w_equal).sum(axis=1)

    eq_minvar = (1.0 + port_minvar).cumprod() * cfg.initial_value
    eq_equal = (1.0 + port_equal).cumprod() * cfg.initial_value

    return (port_minvar, eq_minvar), (port_equal, eq_equal)


def main() -> None:
    folder = Path(__file__).resolve().parent
    cfg = BacktestConfig()

    prices = ensure_prices_cache(cfg, folder / "data")
    (r_mv, eq_mv), (r_eq, eq_eq) = backtest(cfg, prices)

    stats_mv = perf_stats(r_mv, eq_mv)
    stats_eq = perf_stats(r_eq, eq_eq)

    summary = pd.DataFrame([stats_mv, stats_eq], index=["MinVar (long-only)", "EqualWeight"])
    print("\nPerformance (approx):")
    print(summary.applymap(lambda x: f"{x: .2%}" if np.isfinite(x) else "nan"))

    # Plot
    import matplotlib.pyplot as plt

    out = folder / "equity_curves.png"
    plt.figure(figsize=(10, 5))
    plt.plot(eq_mv.index, eq_mv.values, label="MinVar (long-only)")
    plt.plot(eq_eq.index, eq_eq.values, label="EqualWeight", alpha=0.85)
    plt.title(f"Walk-forward portfolio equity (rebalance: {cfg.rebalance}, lookback: {cfg.lookback_days}d)\n{', '.join(cfg.tickers)}")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)

    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
