# Day 026 — Overnight vs Intraday Returns (SPY)

Decompose SPY’s daily return into:

- **Overnight**: previous close → today open
- **Intraday**: today open → today close

Then compare their distributions and long-run contribution to equity returns.

## What this does
- Downloads **SPY** OHLCV from Yahoo Finance (via `yfinance`).
- Computes:
  - overnight return: `Open_t / Close_{t-1} - 1`
  - intraday return: `Close_t / Open_t - 1`
  - total return: `Close_t / Close_{t-1} - 1`
- Reports summary stats (mean, vol, Sharpe-like, hit rate, skew/kurtosis).
- Writes a CSV of daily returns and an SVG plot:
  - `output/spy_overnight_intraday.csv`
  - `output/cum_returns.svg`

## How to run
From the repo root:

```bash
pip install -r requirements.txt
python3 day-026-overnight-vs-intraday-spy/overnight_vs_intraday_spy.py
```

## Notes
- This is exploratory/educational.
- Yahoo data can occasionally have missing opens/closes around corporate actions/holidays; the script drops invalid rows.
