"""Day 034 — Time-Varying Beta via Recursive Least Squares (RLS)

We model asset returns r_a(t) as:
  r_a(t) = alpha(t) + beta(t) * r_b(t) + eps(t)

RLS update (forgetting factor lambda):
  K_t = P_{t-1} x_t / (lambda + x_t^T P_{t-1} x_t)
  theta_t = theta_{t-1} + K_t (y_t - x_t^T theta_{t-1})
  P_t = (P_{t-1} - K_t x_t^T P_{t-1}) / lambda

Where x_t = [1, r_b(t)] and theta = [alpha, beta].

Outputs:
- time-varying beta series
- rolling OLS beta for comparison
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass
class RLSResult:
    alpha: pd.Series
    beta: pd.Series


def fetch_adj_close(tickers: list[str], start: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all")
    return px


def rls_time_varying_beta(asset_ret: pd.Series, bench_ret: pd.Series, lambda_: float = 0.99) -> RLSResult:
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    y = df.iloc[:, 0].to_numpy(dtype=float)
    x1 = df.iloc[:, 1].to_numpy(dtype=float)

    # theta = [alpha, beta]
    theta = np.zeros(2, dtype=float)

    # Large initial covariance => allow fast initial learning
    P = np.eye(2, dtype=float) * 1e3

    alphas = np.empty(len(df), dtype=float)
    betas = np.empty(len(df), dtype=float)

    for t in range(len(df)):
        x = np.array([1.0, x1[t]], dtype=float)  # shape (2,)

        denom = lambda_ + x @ P @ x
        K = (P @ x) / denom  # shape (2,)

        err = y[t] - (x @ theta)
        theta = theta + K * err

        # Joseph form not necessary here; keep it simple.
        P = (P - np.outer(K, x) @ P) / lambda_

        alphas[t] = theta[0]
        betas[t] = theta[1]

    alpha_s = pd.Series(alphas, index=df.index, name="alpha_rls")
    beta_s = pd.Series(betas, index=df.index, name="beta_rls")
    return RLSResult(alpha=alpha_s, beta=beta_s)


def rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series, window: int = 126) -> pd.Series:
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    ra = df.iloc[:, 0]
    rb = df.iloc[:, 1]

    cov = ra.rolling(window).cov(rb)
    var = rb.rolling(window).var()
    beta = cov / var
    beta.name = f"beta_roll_{window}d"
    return beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", type=str, default="QQQ")
    ap.add_argument("--benchmark", type=str, default="SPY")
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--lambda_", type=float, default=0.99, help="RLS forgetting factor (near 1.0 = smoother)")
    ap.add_argument("--roll_window", type=int, default=126, help="rolling window in trading days")
    ap.add_argument("--outdir", type=str, default="day-034-rls-time-varying-beta/out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    px = fetch_adj_close([args.asset, args.benchmark], start=args.start)
    rets = px.pct_change().dropna()

    asset_ret = rets[args.asset].rename("asset")
    bench_ret = rets[args.benchmark].rename("bench")

    rls = rls_time_varying_beta(asset_ret, bench_ret, lambda_=args.lambda_)
    beta_roll = rolling_beta(asset_ret, bench_ret, window=args.roll_window)

    beta_df = pd.concat([rls.beta, beta_roll], axis=1)

    # Plot beta series
    plt.figure(figsize=(11, 5))
    plt.plot(beta_df.index, beta_df["beta_rls"], label=f"RLS beta (lambda={args.lambda_})", linewidth=1.2)
    plt.plot(beta_df.index, beta_df[beta_roll.name], label=f"Rolling beta ({args.roll_window}d)", linewidth=1.2, alpha=0.9)
    plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.4)
    plt.title(f"Time-varying beta: {args.asset} vs {args.benchmark}")
    plt.ylabel("beta")
    plt.legend()
    plt.tight_layout()
    beta_path = outdir / "beta_time_series.png"
    plt.savefig(beta_path, dpi=150)
    plt.close()

    # Quick sanity: implied hedged (asset - beta*bench) using RLS beta
    aligned = pd.concat([asset_ret, bench_ret, rls.beta], axis=1).dropna()
    hedged = aligned["asset"] - aligned["beta_rls"] * aligned["bench"]

    plt.figure(figsize=(11, 5))
    (1 + hedged).cumprod().plot(color="tab:purple", linewidth=1.2)
    plt.title(f"RLS-beta hedged return index: {args.asset} - beta(t)*{args.benchmark}")
    plt.ylabel("growth of $1")
    plt.tight_layout()
    hedge_path = outdir / "hedged_growth.png"
    plt.savefig(hedge_path, dpi=150)
    plt.close()

    # Save beta data
    beta_df.to_csv(outdir / "beta_series.csv", index=True)

    print("Wrote:")
    print(f"- {beta_path}")
    print(f"- {hedge_path}")
    print(f"- {outdir / 'beta_series.csv'}")


if __name__ == "__main__":
    main()
