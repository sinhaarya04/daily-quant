# Day 028 — PCA Sector Risk Model (Eigen-Covariance) + Walk-forward Min-Var

Build a simple **PCA (eigen) risk model** for US sector ETFs, then use it inside a **walk-forward minimum-variance** portfolio.

Why this is interesting:
- Sample covariance matrices are noisy.
- PCA keeps the **top risk factors** (principal components) and discards smaller, noisier components.
- You can compare a min-var portfolio built with:
  - raw sample covariance
  - PCA “factor” covariance with `k` components

## What this project does
- Downloads daily adjusted closes with `yfinance`
- Computes daily returns
- Runs a **monthly rebalance** backtest:
  - estimate covariance on a trailing window
  - compute min-var weights
  - hold for ~1 month
- Compares performance for different PCA component counts `k`

## How to run
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python day-028-pca-sector-risk-model/pca_minvar_backtest.py --start 2006-01-01 --lookback 756 --k 3 5 8
```

Outputs:
- prints summary stats to the terminal
- saves plots into `day-028-pca-sector-risk-model/output/`

## Notes
- This is educational code (not financial advice).
- Min-var here is **unconstrained** (weights can be negative). Add constraints if you want a tradable long-only portfolio.
