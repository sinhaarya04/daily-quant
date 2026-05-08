# Day 029 — Inflation expectations & real-rate regimes vs SPY

A tiny macro/markets study:

- Pull **5Y breakeven inflation** (`T5YIE`) and **10Y real rate** (`DFII10`) from **FRED** (via the free `fredgraph.csv` endpoint; no API key).
- Pull **SPY** from `yfinance`.
- Define a simple **4-state regime** based on:
  - real rate < 0 or >= 0
  - breakeven inflation rising or falling over a lookback window
- Compare **next 1-month (21 trading days) forward SPY returns** by regime.
- Build a **toy timing strategy**: hold SPY only in the best historical regime (for illustration; no costs).

## How to run

From the repo root:

```bash
pip install -r requirements.txt
python day-029-inflation-breakeven-realrate-regimes/main.py --start 2003-01-01 --out day-029-inflation-breakeven-realrate-regimes/outputs
```

Outputs (CSV + PNGs) are written to the `--out` folder.

## Notes

- This is an educational mini-project, not investment advice.
- The toy strategy is deliberately naive (no transaction costs/slippage, in-sample regime selection).
