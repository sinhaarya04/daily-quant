# Day 014 — Rolling Hedge Ratio (SPY ↔ TLT)

Goal: estimate a **rolling hedge ratio** (beta) between equity and bonds, then build a simple **hedged return series**:

\[ r^{hedged}_t = r^{SPY}_t - \beta_t \cdot r^{TLT}_t \]

This is a tiny, reproducible example of rolling OLS (no `statsmodels`).

## What it does
- Downloads daily prices for **SPY** and **TLT** via `yfinance`
- Computes daily log returns
- Fits a rolling OLS regression over a window (default **60 trading days**) to estimate \(\beta_t\)
- Compares:
  - unhedged SPY returns
  - hedged returns using rolling \(\beta_t\)
  - (optional) hedged returns using a single full-sample \(\beta\)
- Saves plots to `outputs/`

## How to run
From repo root (recommended: use a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 day-014-rolling-hedge-ratio-spy-tlt/rolling_hedge_ratio.py \
  --start 2006-01-01 --window 60
```

Arguments:
- `--start` start date (YYYY-MM-DD)
- `--end` end date (default: today)
- `--window` rolling window length in trading days

Outputs:
- `outputs/rolling_beta.png`
- `outputs/equity_curve.png`

## Notes
- This is an educational mini-project, not investment advice.
- Rolling betas can be noisy; try different windows and assets.
