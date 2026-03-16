# Day 012 — Sahm Rule (Unemployment) Tactical Allocation

Goal: build a tiny, reproducible backtest that uses a **Sahm-rule-style recession trigger** (based on US unemployment) to switch between **SPY** and a simple **cash proxy** (3M T-bill rate from FRED).

Data (free):
- SPY prices: `yfinance`
- Unemployment rate (`UNRATE`) and 3M T-bill rate (`TB3MS`): FRED CSV endpoints (no API key)

## How to run

From the repo root (recommended: use a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r day-012-sahm-rule-tactical-allocation/requirements.txt
python day-012-sahm-rule-tactical-allocation/run.py
```

This will:
- download data
- compute the Sahm-style trigger
- run a monthly backtest (signal is **shifted by 1 month** to reduce lookahead)
- print summary stats
- save a plot to `day-012-sahm-rule-tactical-allocation/outputs/equity_curves.png`

## Notes / caveats
- This is educational, not investment advice.
- “Cash” is approximated from `TB3MS` as `rate/12` per month (simple approximation).
- FRED series are monthly; SPY is daily and is resampled to month-end.
