# Day 024 — Real Rate Regimes (DGS10 − T10YIE) vs SPY Returns

Idea: build a simple **proxy for the 10Y real rate** as:

> **Real10Y(t) = DGS10(t) − T10YIE(t)**

where:
- **DGS10** = 10-Year Treasury Constant Maturity Rate (FRED)
- **T10YIE** = 10-Year Breakeven Inflation Rate (FRED)

Then check how **SPY forward returns** and **realized volatility** differ across real-rate regimes.

## What this does
- Downloads DGS10 and T10YIE from FRED (no API key; via `fredgraph.csv`).
- Downloads SPY prices from Yahoo Finance.
- Aligns dates, builds the real-rate proxy.
- Compares **forward 21-trading-day returns** (≈1 month) across regimes:
  - real rate above/below median
  - real rate quartiles
- Saves a few plots to `./output/`.

## How to run
From the repo root:

```bash
pip install -r requirements.txt
python3 day-024-real-rate-regimes-spy/real_rate_regimes_spy.py
```

Outputs (PNGs) are written to:

- `day-024-real-rate-regimes-spy/output/`

## Notes
- This is exploratory/educational.
- The real rate here is a **proxy** (nominal − breakeven inflation), not a traded real yield series.
