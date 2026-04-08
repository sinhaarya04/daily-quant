# Day 017 — Sector Rotation via Relative Momentum (Monthly)

A tiny sector-rotation backtest using **relative momentum** across US sector ETFs.

## Idea
Each month:
1. Compute each sector ETF’s trailing **lookback** return (default: 6 months)
2. Pick the **top K** sectors (default: 3)
3. Hold them **equal-weight** for the next month

Benchmark: buy-and-hold **SPY**.

## Data
- Prices: `yfinance` (adjusted close)
- Universe (default): `XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE`

## How to run
From repo root:

```bash
pip install -r requirements.txt
python day-017-sector-rotation-relative-momentum/main.py --start 2005-01-01 --lookback 6 --topk 3
```

Outputs (created under `output/`):
- `equity_curves.csv`
- `weights.csv`
- `equity_curves.png`

## Notes
- This is an educational mini-project, not investment advice.
- `yfinance` data quality can vary; results may differ slightly across runs.
