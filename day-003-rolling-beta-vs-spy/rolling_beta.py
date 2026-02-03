"""Day 003 — Rolling Beta vs SPY (CAPM-style)

Compute a rolling beta of an asset vs a benchmark using daily returns.

Beta_t = Cov(r_asset, r_mkt) / Var(r_mkt) over a rolling window.

Example:
  python day-003-rolling-beta-vs-spy/rolling_beta.py --ticker QQQ --benchmark SPY --start 2015-01-01 --window 60
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


@dataclass
class Config:
    ticker: str
    benchmark: str
    start: str
    window: int
    use_log_returns: bool


def download_adj_close(tickers: list[str], start: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns different shapes depending on single vs multi ticker.
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker: columns like ('Close','SPY') etc.
        if ("Close" in df.columns.get_level_values(0)):
            close = df["Close"].copy()
        else:
            # Fallback: try Adj Close
            close = df["Adj Close"].copy()
        close.columns = [c.upper() for c in close.columns]
        return close

    # Single-ticker DataFrame
    col = "Close" if "Close" in df.columns else "Adj Close"
    out = df[[col]].copy()
    out.columns = [tickers[0].upper()]
    return out


def returns_from_prices(prices: pd.DataFrame, use_log_returns: bool) -> pd.DataFrame:
    prices = prices.sort_index()
    if use_log_returns:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="any")


def rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, window: int) -> pd.Series:
    # Align
    df = pd.concat([asset_ret.rename("asset"), mkt_ret.rename("mkt")], axis=1).dropna()

    # Rolling covariance / variance
    cov = df["asset"].rolling(window).cov(df["mkt"])
    var = df["mkt"].rolling(window).var()

    beta = cov / var
    beta.name = f"beta_{window}d"
    return beta.dropna()


def plot_results(beta: pd.Series, asset_ret: pd.Series, mkt_ret: pd.Series, cfg: Config) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Rolling beta
    axes[0].plot(beta.index, beta.values, lw=1.6)
    axes[0].axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    axes[0].set_title(
        f"Rolling Beta: {cfg.ticker.upper()} vs {cfg.benchmark.upper()}  (window={cfg.window}d)"
    )
    axes[0].set_ylabel("Beta")
    axes[0].grid(True, alpha=0.25)

    # Scatter of returns (same sample window as beta for fairness)
    df = pd.concat([asset_ret.rename("asset"), mkt_ret.rename("mkt")], axis=1).dropna()
    df = df.loc[beta.index.min() :]

    x = df["mkt"].values
    y = df["asset"].values
    axes[1].scatter(x, y, s=10, alpha=0.35)

    if len(df) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        axes[1].plot(xs, intercept + slope * xs, color="black", lw=1.5, label=f"OLS slope≈{slope:.2f}")
        axes[1].legend(frameon=False)

    axes[1].set_xlabel(f"{cfg.benchmark.upper()} daily return")
    axes[1].set_ylabel(f"{cfg.ticker.upper()} daily return")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Rolling beta vs benchmark (CAPM-style).")
    p.add_argument("--ticker", default="QQQ", help="Asset ticker (default: QQQ)")
    p.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    p.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--window", type=int, default=60, help="Rolling window in trading days")
    p.add_argument("--simple", action="store_true", help="Use simple returns instead of log returns")
    a = p.parse_args()

    return Config(
        ticker=a.ticker,
        benchmark=a.benchmark,
        start=a.start,
        window=int(a.window),
        use_log_returns=not a.simple,
    )


def main() -> None:
    cfg = parse_args()

    tickers = [cfg.ticker.upper(), cfg.benchmark.upper()]
    prices = download_adj_close(tickers, cfg.start)

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise SystemExit(f"Missing price series for: {missing}. Got columns={list(prices.columns)}")

    rets = returns_from_prices(prices[tickers], cfg.use_log_returns)
    asset_ret = rets[cfg.ticker.upper()]
    mkt_ret = rets[cfg.benchmark.upper()]

    beta = rolling_beta(asset_ret, mkt_ret, cfg.window)

    # Quick summary
    print(f"Sample: {beta.index.min().date()} → {beta.index.max().date()}  (n={len(beta)})")
    print(f"Beta mean: {beta.mean():.3f} | median: {beta.median():.3f} | last: {beta.iloc[-1]:.3f}")

    plot_results(beta, asset_ret, mkt_ret, cfg)


if __name__ == "__main__":
    main()
