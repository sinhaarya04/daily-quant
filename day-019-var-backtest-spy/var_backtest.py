"""Day 019 — Value-at-Risk (VaR) + Simple Backtest (SPY)

Rolling 1-day VaR computed two ways:
- Historical VaR: empirical quantile over a trailing window
- Parametric (Gaussian) VaR: mean/std over a trailing window with fixed z-scores

Backtest: count exceptions (realized return < -VaR).

Dependencies: numpy, pandas, yfinance, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# Fixed z-scores for standard normal left-tail quantiles
# alpha=0.05 => ppf(0.05) ≈ -1.64485
# alpha=0.01 => ppf(0.01) ≈ -2.32635
Z_LEFT_TAIL = {
    0.10: -1.28155,
    0.05: -1.64485,
    0.025: -1.95996,
    0.01: -2.32635,
}


@dataclass
class VaRResult:
    df: pd.DataFrame
    summary: pd.DataFrame


def download_prices(ticker: str, start: str) -> pd.Series:
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError(f"No data downloaded for {ticker}.")

    # With auto_adjust=True, 'Close' is adjusted close
    if "Close" not in data.columns:
        raise RuntimeError("Expected 'Close' column from yfinance.")

    px = data["Close"].dropna()
    px.name = "price"
    return px


def compute_log_returns(price: pd.Series) -> pd.Series:
    r = np.log(price).diff().dropna()
    r.name = "ret"
    return r


def rolling_var(returns: pd.Series, window: int, alpha: float) -> VaRResult:
    if alpha not in Z_LEFT_TAIL:
        raise ValueError(
            f"alpha={alpha} not supported for parametric VaR in this mini-project. "
            f"Choose one of: {sorted(Z_LEFT_TAIL.keys())}"
        )

    r = returns.copy().dropna()
    df = pd.DataFrame({"ret": r})

    # Rolling stats
    roll = df["ret"].rolling(window=window)
    mu = roll.mean()
    sigma = roll.std(ddof=1)

    # Historical VaR: left-tail quantile (negative number), convert to positive loss
    hist_q = roll.quantile(alpha)
    var_hist = -hist_q

    # Parametric VaR: mu + sigma * z_alpha (z_alpha is negative); convert to positive loss
    z = Z_LEFT_TAIL[alpha]
    var_param = -(mu + sigma * z)

    out = df.assign(
        mu=mu,
        sigma=sigma,
        var_hist=var_hist,
        var_param=var_param,
    ).dropna()

    # Exceptions: realized return < -VaR (worse than predicted)
    out["exc_hist"] = out["ret"] < -out["var_hist"]
    out["exc_param"] = out["ret"] < -out["var_param"]

    n = len(out)
    summary = pd.DataFrame(
        {
            "n_obs": [n],
            "window": [window],
            "alpha": [alpha],
            "expected_exception_rate": [alpha],
            "hist_exception_rate": [out["exc_hist"].mean() if n else np.nan],
            "param_exception_rate": [out["exc_param"].mean() if n else np.nan],
            "hist_exceptions": [int(out["exc_hist"].sum()) if n else 0],
            "param_exceptions": [int(out["exc_param"].sum()) if n else 0],
        }
    )

    return VaRResult(df=out, summary=summary)


def plot_var(df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))

    # Plot returns
    ax.plot(df.index, df["ret"], color="black", lw=0.7, alpha=0.7, label="log return")

    # Plot VaR bands (as negative thresholds)
    ax.plot(df.index, -df["var_hist"], color="#1f77b4", lw=1.1, label="-VaR (historical)")
    ax.plot(df.index, -df["var_param"], color="#ff7f0e", lw=1.1, label="-VaR (parametric)")

    # Mark exceptions
    exc_h = df[df["exc_hist"]]
    exc_p = df[df["exc_param"]]
    ax.scatter(exc_h.index, exc_h["ret"], s=10, color="#1f77b4", alpha=0.8, label="exceptions (hist)")
    ax.scatter(exc_p.index, exc_p["ret"], s=10, color="#ff7f0e", alpha=0.8, label="exceptions (param)")

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("log return")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--alpha", type=float, default=0.05, help="Left-tail probability (e.g. 0.05 for 95% VaR)")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    args = p.parse_args()

    outdir = Path(args.outdir)

    px = download_prices(args.ticker, args.start)
    rets = compute_log_returns(px)

    res = rolling_var(rets, window=args.window, alpha=args.alpha)

    summary_path = outdir / "var_backtest_summary.csv"
    res.summary.to_csv(summary_path, index=False)

    plot_path = outdir / "var_backtest_plot.png"
    plot_var(
        res.df,
        title=f"{args.ticker} 1-day VaR backtest | window={args.window} | alpha={args.alpha}",
        out_path=plot_path,
    )

    # Print a small console summary
    s = res.summary.iloc[0].to_dict()
    print("--- VaR backtest summary ---")
    for k in [
        "n_obs",
        "window",
        "alpha",
        "expected_exception_rate",
        "hist_exception_rate",
        "param_exception_rate",
        "hist_exceptions",
        "param_exceptions",
    ]:
        print(f"{k}: {s[k]}")
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
