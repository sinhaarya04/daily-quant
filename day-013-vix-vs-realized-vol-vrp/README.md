# Day 13 — VIX vs Realized Vol (Volatility Risk Premium)

Mini-project: pull **VIX** from FRED and **SPY** from Yahoo Finance, compute a simple 21-trading-day realized volatility proxy, and compare it to VIX.

We’ll look at:
- Realized vol (21d rolling, annualized)
- VIX (as an implied-vol proxy)
- VRP proxy: `VIX - RealizedVol` (both in vol points)

## How to run

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 day-013-vix-vs-realized-vol-vrp/analyze_vrp.py --start 2004-01-01 --window 21
```

Outputs:
- Console summary stats
- `day-013-vix-vs-realized-vol-vrp/vrp_timeseries.png`

## Notes
- VIX is an option-implied **30-day** variance estimate; realized vol here is a **21 trading-day** close-to-close proxy. So this is not apples-to-apples, but it’s a useful quick diagnostic.
- All data sources are free (FRED + yfinance).
