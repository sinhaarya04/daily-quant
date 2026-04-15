# Day 021 — Variance Ratio Test (Random Walk?) on SPY

This mini-project implements the **Lo–MacKinlay variance ratio (VR) test** to check whether SPY daily returns behave like a random walk (i.e., no serial correlation in increments).

## What it does
- Downloads **SPY** adjusted-close prices via `yfinance`
- Computes daily log returns
- Computes **VR(k)** for multiple horizons (k = 2, 5, 10, 20)
- Reports both:
  - **Homoskedastic** test statistic (classic VR)
  - **Heteroskedastic-robust** test statistic (recommended for financial returns)
- Plots a **rolling 5-year VR(5)** to visualize regime changes

## How to run
From the repo root (recommended: use a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 day-021-variance-ratio-random-walk/main.py
```

Outputs:
- a printed summary table in the terminal
- `output_vr_table.csv`
- `rolling_vr5.png`

## Notes
- VR(k) > 1 suggests **positive** serial correlation at horizon k (trend/continuation)
- VR(k) < 1 suggests **negative** serial correlation (mean reversion)
- Statistical significance depends heavily on sample period and market regime.

Educational only — not financial advice.
