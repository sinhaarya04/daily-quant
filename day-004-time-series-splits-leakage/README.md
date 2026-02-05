# Day 004 — Time-series splits + leakage checklist

Goal: build **time-series-safe train/validation splits** and a **leakage checklist** you can reuse in every project.

This mini-project includes:
- A simple **walk-forward / anchored split** generator
- A quick **leakage audit** (common failure modes)
- A tiny demo showing how leakage can sneak in (and how to detect it)

## What “leakage” looks like in practice
Common sources:
1. **Random train/test split** on time-series
2. **Using today’s close to predict today’s close** (missing feature lag)
3. **Fitting scalers/encoders on the full dataset** (fit only on train)
4. **Using future-derived labels** (e.g., forward returns) without aligning timestamps properly
5. **Feature engineering with rolling windows that peek ahead**
6. **Data vendor fields that are revised later** (survivorship / restatements)

## Run
```bash
python day-004-time-series-splits-leakage/leakage_demo.py
```

## Output
You’ll see:
- the generated split ranges
- a check that split boundaries don’t overlap
- a synthetic example where a leaky feature produces unrealistically high scores

## Next
You can lift the functions from `leakage_demo.py` into a shared `utils/` folder later.
