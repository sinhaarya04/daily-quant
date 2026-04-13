# Day 019 — Value-at-Risk (VaR) + Simple Backtest (SPY)

This mini-project computes **rolling 1-day VaR** for SPY using two common methods:
- **Historical VaR** (empirical quantile of past returns)
- **Parametric (Gaussian) VaR** (mean + std with fixed z-scores)

It then performs a lightweight backtest by counting **exceptions** (days when the realized return is worse than the predicted VaR).

## How to run

From the repo root:

```bash
pip install -r requirements.txt
python day-019-var-backtest-spy/var_backtest.py
```

Optional arguments:

```bash
python day-019-var-backtest-spy/var_backtest.py \
  --ticker SPY \
  --window 252 \
  --start 2005-01-01 \
  --alpha 0.05
```

Outputs (written to the day folder):
- `var_backtest_summary.csv`
- `var_backtest_plot.png`

## Notes
- VaR is a **risk estimate**, not a guarantee.
- The Gaussian VaR assumes returns are (approximately) normal; historical VaR avoids that assumption but depends on the chosen window.
