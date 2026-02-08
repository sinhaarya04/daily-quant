# Day 006 — Baseline ML classifier (time-series safe)

Goal: take the feature pipeline from Day 005 and train a simple, **time-series-safe** baseline model.

We:
- download SPY data via `yfinance`
- build features (using `day-005-feature-pipeline-v1/features.py`)
- create a binary label: whether **next-day return is positive**
- split chronologically (no shuffle)
- train a couple baselines (LogReg + RandomForest)
- report accuracy + ROC-AUC

## Run

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python day-006-baseline-ml-classifier/train.py --ticker SPY --start 2010-01-01
```

## Notes
- This is intentionally simple. The point is wiring + correctness (no leakage).
- Next steps: add walk-forward validation, probability calibration, and a trading/backtest layer.
