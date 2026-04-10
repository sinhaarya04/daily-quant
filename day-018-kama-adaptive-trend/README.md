# Day 018 — KAMA (Kaufman Adaptive Moving Average) Trend Filter (SPY)

This mini-project implements **Kaufman’s Adaptive Moving Average (KAMA)** and uses it as a simple trend filter:
- **Long SPY** when `Close > KAMA`
- **Cash** otherwise

It compares:
- Buy & Hold (SPY)
- 200-day SMA filter
- KAMA filter

## How to run

From the repo root:

```bash
pip install -r requirements.txt
python day-018-kama-adaptive-trend/main.py --ticker SPY --start 2005-01-01
```

Outputs:
- Prints performance stats table
- Saves a plot to `day-018-kama-adaptive-trend/output.png`

## Notes
- Uses daily adjusted close from `yfinance`.
- Includes a simple proportional transaction cost in bps (default: 1 bp per position change).
- Educational only (not financial advice).
