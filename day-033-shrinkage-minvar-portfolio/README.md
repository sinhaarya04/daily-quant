# Day 033 — Shrinkage Covariance for a Minimum-Variance Portfolio (Sample vs Ledoit–Wolf)

This mini-project compares **minimum-variance portfolio** weights built from:
1) the plain **sample covariance**, and
2) the **Ledoit–Wolf shrinkage** covariance estimator,
using a simple rolling, monthly rebalanced backtest.

Assets (default): **SPY / TLT / GLD** (all free via `yfinance`).

## What you’ll see
- How shrinkage stabilizes covariance estimates (especially with limited history)
- A rolling **out-of-sample** comparison of:
  - realized volatility
  - cumulative returns
  - turnover (rough proxy for trading intensity)

## How to run
From the repo root:

```bash
pip install -r requirements.txt
python day-033-shrinkage-minvar-portfolio/main.py
```

Outputs (plots + CSV) are written to:
- `day-033-shrinkage-minvar-portfolio/outputs/`

## Notes / caveats
- This is **educational**: ignores transaction costs, slippage, borrow, taxes.
- Long-only enforcement is a simple non-negativity clamp + renormalization (not a full QP).
