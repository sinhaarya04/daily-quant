"""Day 033 — Shrinkage Covariance for a Minimum-Variance Portfolio (Sample vs Ledoit–Wolf)

Self-contained script:
- pulls adjusted close data from Yahoo via yfinance
- builds rolling monthly-rebalanced min-var portfolios using:
  (a) sample covariance
  (b) Ledoit–Wolf shrinkage covariance
- compares out-of-sample performance

Outputs plots + CSVs under ./outputs/

Run:
  pip install -r requirements.txt
  python day-033-shrinkage-minvar-portfolio/main.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

import matplotlib.pyplot as plt


@dataclass
class Config:
    tickers: Tuple[str, ...] = ("SPY", "TLT", "GLD")
    start: str = "2006-01-01"
    end: str | None = None
    lookback_days: int = 252  # ~1y of trading days
    rebalance_freq: str = "M"  # month-end
    ann_factor: int = 252
    output_dir: str = os.path.join(os.path.dirname(__file__), "outputs")
    max_weight: float = 0.90  # basic concentration guardrail


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_prices(tickers: Tuple[str, ...], start: str, end: str | None) -> pd.DataFrame:
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance returns multiindex columns: (field, ticker)
        prices = data["Close"].copy()
    else:
        # single ticker case
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()
    prices = prices.loc[:, list(tickers)]
    return prices


def min_var_weights_from_cov(cov: np.ndarray) -> np.ndarray:
    """Unconstrained min-variance weights (sum to 1).

    w = inv(C) 1 / (1' inv(C) 1)
    """
    n = cov.shape[0]
    ones = np.ones(n)
    # add tiny jitter for numerical stability
    cov_j = cov + 1e-10 * np.eye(n)
    inv = np.linalg.pinv(cov_j)
    w = inv @ ones
    denom = float(ones.T @ inv @ ones)
    w = w / denom
    return w


def long_only_clamp(w: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """Simple long-only approximation: clamp negatives to 0, cap large weights, renormalize.

    Not a QP solver; just a pragmatic, dependency-light constraint proxy.
    """
    w = np.asarray(w).copy()
    w[w < 0] = 0.0
    w = np.minimum(w, max_weight)
    s = w.sum()
    if s <= 0:
        # fallback to equal weight
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w


def compute_turnover(prev_w: np.ndarray, new_w: np.ndarray) -> float:
    # 0.5 * sum |Δw| is common; we keep it as sum |Δw| for a simple proxy.
    return float(np.sum(np.abs(new_w - prev_w)))


def backtest(prices: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    rets = prices.pct_change().dropna()

    # rebalance dates = month end dates present in our index
    rebal_dates = rets.resample(cfg.rebalance_freq).last().index
    rebal_dates = rebal_dates.intersection(rets.index)

    tickers = list(prices.columns)
    n = len(tickers)

    w_sample = pd.DataFrame(index=rebal_dates, columns=tickers, dtype=float)
    w_lw = pd.DataFrame(index=rebal_dates, columns=tickers, dtype=float)

    # to store daily portfolio returns for each strategy
    port_ret = pd.DataFrame(index=rets.index, columns=["sample", "ledoit_wolf"], dtype=float)

    prev_ws = np.ones(n) / n
    prev_wl = np.ones(n) / n
    turnover = []

    for i, d in enumerate(rebal_dates):
        # window ends at d (inclusive)
        window = rets.loc[:d].tail(cfg.lookback_days)
        if len(window) < max(60, n + 5):
            # not enough history: keep equal weight
            ws = prev_ws
            wl = prev_wl
        else:
            X = window.values

            # sample covariance
            cov_s = np.cov(X, rowvar=False, ddof=1)
            ws = min_var_weights_from_cov(cov_s)
            ws = long_only_clamp(ws, max_weight=cfg.max_weight)

            # Ledoit-Wolf shrinkage covariance
            lw = LedoitWolf().fit(X)
            cov_lw = lw.covariance_
            wl = min_var_weights_from_cov(cov_lw)
            wl = long_only_clamp(wl, max_weight=cfg.max_weight)

        w_sample.loc[d] = ws
        w_lw.loc[d] = wl

        turnover.append(
            {
                "date": d,
                "turnover_sample": compute_turnover(prev_ws, ws),
                "turnover_ledoit_wolf": compute_turnover(prev_wl, wl),
            }
        )
        prev_ws, prev_wl = ws, wl

        # apply weights from next trading day until next rebalance date
        start_idx = rets.index.get_loc(d)
        if start_idx + 1 >= len(rets.index):
            continue
        start = rets.index[start_idx + 1]
        end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rets.index[-1]
        period = rets.loc[start:end]

        port_ret.loc[period.index, "sample"] = period.values @ ws
        port_ret.loc[period.index, "ledoit_wolf"] = period.values @ wl

    turnover_df = pd.DataFrame(turnover).set_index("date")

    # performance stats
    perf = {}
    for col in port_ret.columns:
        s = port_ret[col].dropna()
        ann_ret = (1 + s).prod() ** (cfg.ann_factor / len(s)) - 1
        ann_vol = s.std() * np.sqrt(cfg.ann_factor)
        sharpe = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
        dd = (1 + s).cumprod() / (1 + s).cumprod().cummax() - 1
        max_dd = dd.min()
        perf[col] = {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
    perf_df = pd.DataFrame(perf).T

    return {
        "returns": port_ret,
        "weights_sample": w_sample,
        "weights_ledoit_wolf": w_lw,
        "turnover": turnover_df,
        "performance": perf_df,
    }


def plot_equity_curves(port_ret: pd.DataFrame, outpath: str) -> None:
    eq = (1 + port_ret.fillna(0)).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["sample"], label="MinVar (Sample Cov)")
    plt.plot(eq.index, eq["ledoit_wolf"], label="MinVar (Ledoit-Wolf)")
    plt.title("Equity Curves (Monthly Rebalance, OOS)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_turnover(turnover: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(turnover.index, turnover["turnover_sample"], label="Sample")
    plt.plot(turnover.index, turnover["turnover_ledoit_wolf"], label="Ledoit-Wolf")
    plt.title("Turnover Proxy (sum |Δw| per rebalance)")
    plt.ylabel("Turnover")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    cfg = Config()
    ensure_dir(cfg.output_dir)

    prices = download_prices(cfg.tickers, cfg.start, cfg.end)
    results = backtest(prices, cfg)

    # save outputs
    results["returns"].to_csv(os.path.join(cfg.output_dir, "daily_portfolio_returns.csv"))
    results["weights_sample"].to_csv(os.path.join(cfg.output_dir, "weights_sample.csv"))
    results["weights_ledoit_wolf"].to_csv(os.path.join(cfg.output_dir, "weights_ledoit_wolf.csv"))
    results["turnover"].to_csv(os.path.join(cfg.output_dir, "turnover.csv"))
    results["performance"].to_csv(os.path.join(cfg.output_dir, "performance_summary.csv"))

    plot_equity_curves(results["returns"], os.path.join(cfg.output_dir, "equity_curves.png"))
    plot_turnover(results["turnover"], os.path.join(cfg.output_dir, "turnover.png"))

    print("Saved outputs to:", cfg.output_dir)
    print("\nPerformance summary:\n")
    print(results["performance"].to_string(float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
