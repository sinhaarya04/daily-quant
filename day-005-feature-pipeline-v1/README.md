# Day 005 — Feature pipeline v1 (tabular) + data validation

Goal: build one `make_features()` that returns a clean (X, y) table for ML, with
basic schema + leakage checks.

## What this includes
- Pull daily OHLCV data (default: SPY) via `yfinance`
- Create simple, *lagged* features
- Create a next-day target (classification or regression)
- Validate:
  - no NaNs in final dataset
  - no future-looking columns (feature timestamps <= label timestamp)
  - alignment of X and y indices

## Quickstart
```bash
pip install -r ../requirements.txt
python features.py --ticker SPY --start 2010-01-01 --horizon 1
```

## Output
- Prints dataset shape + a small schema summary
- Writes `data/features_SPY.csv` (optional)

## Next steps (Day 006)
- Add naive/linear baselines and evaluate with walk-forward splits.
