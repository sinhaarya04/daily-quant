# Day 25 — Credit spread regimes vs. SPY forward returns

Mini-project: use a simple credit-stress proxy from FRED (Moody's Baa yield minus 10Y Treasury) to define regimes and examine SPY forward returns by regime.

## Data
- FRED (no API key):
  - `BAA` — Moody’s Seasoned Baa Corporate Bond Yield
  - `DGS10` — 10-Year Treasury Constant Maturity Rate
- `SPY` prices via `yfinance`

## How to run
From the repo root:

```bash
pip install -r requirements.txt
python day-025-credit-spread-regimes-spy/credit_spread_regimes_spy.py
```

Outputs:
- Prints a summary table of forward returns (1M/3M/6M) by regime
- Writes `credit_spread_regimes_spy.png` in the current working directory

## Notes
This is an exploratory, in-sample analysis (no transaction costs, no formal strategy backtest).
