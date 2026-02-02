"""Day 002 — GARCH(1,1) Volatility

Example:
  python day-002-garch-volatility/garch_vol.py --ticker SPY --start 2015-01-01

Requires:
  numpy, pandas, yfinance, matplotlib, arch
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def get_prices(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker={ticker} start={start}")
    if "Close" not in df.columns:
        raise RuntimeError("Expected 'Close' column in yfinance output")
    px = df["Close"].dropna()
    px.name = "close"
    return px


def log_returns(px: pd.Series) -> pd.Series:
    r = np.log(px).diff().dropna()
    r.name = "log_ret"
    return r


def fit_garch(returns: pd.Series):
    # Lazy import so the error message is clear if arch isn't installed.
    try:
        from arch import arch_model
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'arch'. Install it via: pip install arch\n"
            "(or add it to requirements.txt)"
        ) from e

    # arch expects returns in percent units for numerical stability.
    r_pct = 100.0 * returns

    am = arch_model(
        r_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )
    res = am.fit(disp="off")
    return res


def annualize_vol_from_pct_sigma(sigma_pct: pd.Series, periods_per_year: int = 252) -> pd.Series:
    # sigma_pct is conditional std dev of returns in percent.
    # Convert percent -> decimal and annualize.
    sigma = (sigma_pct / 100.0) * np.sqrt(periods_per_year)
    sigma.name = "ann_vol"
    return sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--save", default="", help="Optional path to save the plot PNG")
    args = ap.parse_args()

    px = get_prices(args.ticker, args.start)
    r = log_returns(px)

    res = fit_garch(r)
    cond_vol_pct = res.conditional_volatility
    cond_vol_ann = annualize_vol_from_pct_sigma(cond_vol_pct)

    print("\n=== GARCH(1,1) fit (Student-t) ===")
    print(res.summary())

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    px.plot(ax=axes[0], color="black", linewidth=1)
    axes[0].set_title(f"{args.ticker} (adjusted close)")
    axes[0].set_ylabel("Price")

    r.plot(ax=axes[1], color="#1f77b4", linewidth=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_title("Log returns")
    axes[1].set_ylabel("Return")

    cond_vol_ann.plot(ax=axes[2], color="#d62728", linewidth=1)
    axes[2].set_title("Conditional volatility (annualized)")
    axes[2].set_ylabel("Ann. vol")

    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"\nSaved plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
