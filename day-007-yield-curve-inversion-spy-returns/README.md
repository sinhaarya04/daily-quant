# Day 007 — Yield Curve Inversion vs Forward SPY Returns

**Goal:** pull the 10Y–2Y Treasury term spread from FRED and test whether yield curve inversions (spread < 0) have historically coincided with different *forward 12‑month* SPY returns.

This is intentionally simple:
- monthly data alignment
- forward 12m total return (using SPY adjusted close)
- conditional summary stats + quick plots

## How to run
From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python day-007-yield-curve-inversion-spy-returns/analyze.py
```

Outputs are written to:
- `day-007-yield-curve-inversion-spy-returns/outputs/`

## Data sources
- FRED (free CSV endpoint): `T10Y2Y` (10-Year Treasury Constant Maturity Minus 2-Year)
- Yahoo Finance via `yfinance`: SPY adjusted close

## Notes
- This is educational and not investment advice.
- The analysis uses end-of-month values. More careful studies might use daily data, different horizons, and regime controls.
