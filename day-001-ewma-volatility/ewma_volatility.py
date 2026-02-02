import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass
class Config:
    ticker: str
    lam: float
    years: int
    trading_days: int = 252


def ewma_volatility(returns: pd.Series, lam: float) -> pd.Series:
    """RiskMetrics-style EWMA volatility (annualized).

    sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2
    """
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("No returns to compute EWMA volatility.")

    # initialize with sample variance
    var0 = float(r.var(ddof=1))
    vars_ = np.empty(len(r), dtype=float)
    vars_[0] = var0

    for i in range(1, len(r)):
        vars_[i] = lam * vars_[i - 1] + (1.0 - lam) * (r.iloc[i - 1] ** 2)

    vol = np.sqrt(vars_) * math.sqrt(252.0)  # annualize
    return pd.Series(vol, index=r.index, name="ewma_vol")


def main(cfg: Config) -> None:
    df = yf.download(cfg.ticker, period=f"{cfg.years}y", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker={cfg.ticker}")

    px = df["Close"].astype(float)
    logret = np.log(px).diff().rename("logret")

    vol = ewma_volatility(logret, cfg.lam)

    # quick & dirty 1-day 95% VaR estimate assuming normality
    # VaR ≈ z * sigma_daily
    sigma_daily = vol / math.sqrt(252.0)
    var95 = (1.645 * sigma_daily).rename("VaR_95")

    out = pd.concat([px, logret, vol, var95], axis=1).dropna()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(out.index, out["Close"], linewidth=1.2)
    axes[0].set_title(f"{cfg.ticker} (Adj Close)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(out.index, out["ewma_vol"], linewidth=1.2)
    axes[1].set_title(f"EWMA Vol (annualized), lambda={cfg.lam}")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(out.index, out["VaR_95"], linewidth=1.2)
    axes[2].set_title("1-day 95% VaR proxy (normal approx, on log returns)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--lambda", dest="lam", type=float, default=0.94)
    p.add_argument("--years", type=int, default=5)
    args = p.parse_args()

    if not (0.0 < args.lam < 1.0):
        raise SystemExit("--lambda must be between 0 and 1")

    main(Config(ticker=args.ticker, lam=args.lam, years=args.years))
