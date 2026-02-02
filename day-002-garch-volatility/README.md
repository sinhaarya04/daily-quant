# Day 002 — GARCH(1,1) Volatility

Fit a **GARCH(1,1)** model to daily returns and visualize the **conditional volatility** time series.

## Why this
EWMA is a great baseline. GARCH is the next step: it estimates volatility dynamics from data (mean reversion + volatility clustering) and is widely used in quant risk/derivatives.

## What this project does
- Downloads daily close prices (default: SPY)
- Computes log returns
- Fits a **GARCH(1,1)** model with Student-t errors
- Plots:
  - price
  - returns
  - conditional volatility (annualized)
- Prints key fitted parameters

## Run
From repo root:

```bash
pip install -r requirements.txt
python day-002-garch-volatility/garch_vol.py --ticker SPY --start 2015-01-01
```

## Notes
- This is educational (not financial advice).
- GARCH can be sensitive to data window and distributional assumptions.
