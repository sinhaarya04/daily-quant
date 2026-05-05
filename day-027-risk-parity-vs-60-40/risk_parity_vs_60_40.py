"""Day 027 — Risk Parity (Inverse-Vol) vs 60/40 (SPY/TLT)

Self-contained mini backtest:
- Pulls daily adjusted close from yfinance
- Monthly rebalancing
- Compares fixed 60/40 vs inverse-vol (60d) risk-parity proxy

Run:
  python day-027-risk-parity-vs-60-40/risk_parity_vs_60_40.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    tickers: tuple[str, ...] = ("SPY", "TLT")
    start: str = "2003-01-01"  # TLT inception ~2002
    end: str | None = None
    rebalance: str = "M"  # month-end
    vol_lookback: int = 60
    initial_value: float = 1.0


def download_prices(cfg: Config) -> pd.DataFrame:
    px = yf.download(
        list(cfg.tickers),
        start=cfg.start,
        end=cfg.end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"].copy()
    else:
        # single ticker edge-case
        px = px.rename("Close").to_frame()

    px = px.dropna(how="any")
    if px.shape[1] != len(cfg.tickers):
        raise RuntimeError(f"Expected {len(cfg.tickers)} tickers, got columns={list(px.columns)}")
    return px


def month_end_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Last trading day of each month in the price index.
    g = pd.Series(index=index, data=1).groupby([index.year, index.month]).tail(1)
    return g.index


def inverse_vol_weights(rets: pd.DataFrame, lookback: int) -> pd.DataFrame:
    # Daily rolling vol; at each date t, use vol computed from t-lookback..t-1
    vol = rets.rolling(lookback).std().shift(1)
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    return w


def run_backtest(cfg: Config, prices: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    rets = prices.pct_change().dropna()

    # Rebalance dates: month-end trading days aligned to returns index
    rb_dates = month_end_rebalance_dates(rets.index)

    # Strategy 1: fixed 60/40
    w_6040 = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    w_6040.loc[:, :] = np.nan
    w_6040.loc[rb_dates, cfg.tickers[0]] = 0.60
    w_6040.loc[rb_dates, cfg.tickers[1]] = 0.40
    w_6040 = w_6040.ffill().dropna()

    # Strategy 2: inverse-vol "risk parity" proxy
    w_inv = inverse_vol_weights(rets, cfg.vol_lookback)
    w_inv = w_inv.loc[w_6040.index]  # align
    w_inv = w_inv.loc[rb_dates].reindex(w_6040.index).ffill().dropna()

    # Portfolio returns
    port_6040 = (w_6040 * rets.loc[w_6040.index]).sum(axis=1)
    port_inv = (w_inv * rets.loc[w_inv.index]).sum(axis=1)

    eq_6040 = (1.0 + port_6040).cumprod() * cfg.initial_value
    eq_inv = (1.0 + port_inv).cumprod() * cfg.initial_value

    return {
        "rets": rets,
        "weights_6040": w_6040,
        "weights_inv": w_inv,
        "equity_6040": eq_6040,
        "equity_inv": eq_inv,
        "portret_6040": port_6040,
        "portret_inv": port_inv,
    }


def ann_stats(port_rets: pd.Series) -> dict[str, float]:
    # assumes daily returns
    ann = 252
    mu = port_rets.mean() * ann
    vol = port_rets.std() * math.sqrt(ann)
    sharpe = (mu / vol) if vol > 0 else np.nan

    eq = (1.0 + port_rets).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = dd.min()
    return {
        "CAGR": float((eq.iloc[-1] ** (ann / len(eq))) - 1.0),
        "AnnVol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(mdd),
    }


def save_plots(outdir: str, results: dict[str, pd.Series | pd.DataFrame]) -> None:
    eq_6040 = results["equity_6040"]
    eq_inv = results["equity_inv"]
    w_6040 = results["weights_6040"]
    w_inv = results["weights_inv"]

    plt.figure(figsize=(10, 5))
    plt.plot(eq_6040.index, eq_6040.values, label="60/40 (monthly rebalance)")
    plt.plot(eq_inv.index, eq_inv.values, label="Inverse-vol (60d) (monthly rebalance)")
    plt.title("Equity Curves (normalized)")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/equity_curve.png", dpi=160)
    plt.close()

    # weights: show only month-end (rebalance) points to reduce noise
    rb = pd.Series(index=w_6040.index, data=1).groupby([w_6040.index.year, w_6040.index.month]).tail(1).index

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    w_6040.loc[rb].plot(kind="area", stacked=True, ax=ax[0], alpha=0.9)
    ax[0].set_title("60/40 Weights")
    ax[0].set_ylabel("Weight")
    ax[0].set_ylim(0, 1)
    ax[0].grid(True, alpha=0.3)

    w_inv.loc[rb].plot(kind="area", stacked=True, ax=ax[1], alpha=0.9)
    ax[1].set_title("Inverse-Vol Weights (60d)")
    ax[1].set_ylabel("Weight")
    ax[1].set_ylim(0, 1)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/weights.png", dpi=160)
    plt.close()


def main() -> None:
    cfg = Config()
    outdir = "day-027-risk-parity-vs-60-40"

    prices = download_prices(cfg)
    results = run_backtest(cfg, prices)

    s6040 = ann_stats(results["portret_6040"])
    sinv = ann_stats(results["portret_inv"])

    print("=== Annualized stats (approx) ===")
    print("60/40:", {k: round(v, 4) for k, v in s6040.items()})
    print("InvVol:", {k: round(v, 4) for k, v in sinv.items()})

    save_plots(outdir, results)
    print(f"Saved plots to: {outdir}/equity_curve.png and {outdir}/weights.png")


if __name__ == "__main__":
    main()
