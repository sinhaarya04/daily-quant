# Day 032 — Industrial Production Regime Filter (FRED INDPRO → SPY/Cash)

Idea: use **US Industrial Production** (FRED: `INDPRO`) as a slow-moving macro signal to toggle a simple risk-on/risk-off allocation.

We:
- download **SPY** prices (yfinance)
- download **INDPRO** from FRED (free CSV endpoint)
- compute **YoY growth** and a **rolling z-score**
- define a monthly regime:
  - **Risk-on** if YoY growth is positive *and* z-score is above 0
  - otherwise **Cash** (0% return)
- backtest vs buy-and-hold SPY

## How to run

```bash
# from repo root
pip install -r requirements.txt

python day-032-industrial-production-regime/main.py --ticker SPY --start 2000-01-01 --zwin 60
```

Outputs:
- console summary stats
- plots saved to `day-032-industrial-production-regime/out/`

## Notes
- This is intentionally simple: no transaction costs, slippage, or dividends on cash.
- INDPRO is monthly and revised; treat results as educational, not tradable.
