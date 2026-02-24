# Day 009 — Trend + Vol Targeting on SPY

A tiny backtest of a simple **trend-following + volatility targeting** strategy on SPY:

- **Trend filter:** only hold SPY when price is above a long moving average (default: 200 trading days).
- **Vol targeting:** scale exposure to target a fixed annualized volatility (default: 10%/yr) using rolling realized vol.
- Optional **leverage cap** and a simple **transaction cost** model.

This is educational (not financial advice).

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python trend_vol_target_spy.py --start 2005-01-01
```

## What it outputs

- Summary stats for **Buy & Hold SPY** vs **Trend+VolTarget**
- A plot saved to `out_equity.png`

## Notes

Data source: `yfinance` (SPY adjusted close).
