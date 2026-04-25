# Day 023 — Financial Stress Regime Filter (FRED STLFSI4 → SPY/Cash)

A tiny backtest that uses the **St. Louis Fed Financial Stress Index (STLFSI4)** from FRED as a simple risk regime filter:

- If stress is **above a threshold** → go to cash (0% daily return)
- Otherwise → hold **SPY**

This is deliberately simple and focuses on **clean data alignment + no lookahead**.

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python day-023-financial-stress-regime-filter/stress_filter.py --start 2000-01-01 --threshold 0.0
```

Outputs:
- prints performance stats for buy & hold vs the filtered strategy
- saves a plot to `day-023-financial-stress-regime-filter/outputs/equity.png`

## Notes
- FRED series is fetched from: https://fred.stlouisfed.org/series/STLFSI4
- Signal is **lagged by 1 business day** (use yesterday’s stress value for today’s position).
- Cash is modeled as 0% daily return (no interest). You can swap in a T-bill series if you want.
