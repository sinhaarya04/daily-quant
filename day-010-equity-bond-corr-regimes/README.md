# Day 010 — Equity/Bond Correlation Regimes (SPY vs TLT) + 10Y Yield

A tiny exploratory study of **equity/bond correlation regimes**:

- Pull **SPY** and **TLT** daily prices from `yfinance`.
- Compute daily returns and a **rolling correlation** (default 63 trading days).
- Pull **10Y Treasury yield (DGS10)** from FRED via `pandas_datareader`.
- Plot: equity curve (SPY/TLT), rolling corr, and the 10Y yield level.

This is educational (not financial advice).

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python equity_bond_corr_regimes.py --start 2005-01-01 --window 63
```

## Output

- Prints a couple quick summary stats (share of time corr < 0, etc.)
- Saves a plot to `out_corr_regimes.png`

## Notes

- Data source:
  - Prices: `yfinance` (adjusted prices)
  - Rates: FRED `DGS10` via `pandas_datareader`
