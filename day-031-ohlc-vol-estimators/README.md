# Day 031 — OHLC Volatility Estimators (Close-to-Close vs Parkinson vs Garman–Klass vs Rogers–Satchell)

Goal: compare common **daily volatility estimators** that use different subsets of OHLC data.

We:
- download daily OHLC for **SPY** (default: last 10y)
- compute several volatility estimators
- compare their level, correlation, and how they relate to next-day absolute returns

## How to run

```bash
# from repo root
pip install -r requirements.txt

python day-031-ohlc-vol-estimators/main.py --ticker SPY --years 10
```

Outputs:
- summary printed to stdout
- plots saved to `day-031-ohlc-vol-estimators/out/`

## Notes
- These are **daily** estimators. They are not a replacement for true realized volatility from intraday data.
- We annualize using \(\sqrt{252}\).
