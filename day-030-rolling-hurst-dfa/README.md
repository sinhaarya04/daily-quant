# Day 030 — Rolling Hurst Exponent (DFA) on SPY

Estimate the **Hurst exponent** of SPY using **Detrended Fluctuation Analysis (DFA)** and look at how it relates to simple trend/mean-reversion behavior.

- H \> 0.5 often indicates persistence (trend-like)
- H \< 0.5 often indicates anti-persistence (mean-reversion-like)

## What this does
1. Downloads SPY daily prices with `yfinance`
2. Runs **rolling DFA** (Hurst) on log-prices
3. Plots the rolling H series
4. Buckets days by H quantiles and compares next-day returns

## How to run
From the repo root:

```bash
pip install -r requirements.txt
python day-030-rolling-hurst-dfa/main.py
```

This will save plots into `day-030-rolling-hurst-dfa/out/` and print a small summary table.

## Notes
- DFA is implemented from scratch (no `statsmodels`).
- This is a small exploratory study, not financial advice.
