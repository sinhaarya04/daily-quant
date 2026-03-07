# Day 011 — Turn-of-the-Month Effect (SPY)

Mini-project: test the classic **turn-of-the-month (TOM)** calendar anomaly on SPY.

We compare:
- **Buy & Hold SPY** (always invested)
- **TOM strategy**: invested only on the **last trading day of each month** and the **first N trading days of each month** (default N=3)

Outputs include summary stats (mean, vol, Sharpe) and a cumulative equity curve plot.

## How to run

From the repo root:

```bash
pip install -r requirements.txt
python day-011-turn-of-month-effect/tom_effect.py
```

Optional arguments:

```bash
python day-011-turn-of-month-effect/tom_effect.py --ticker SPY --start 1993-01-01 --first-days 3 --plot
```

## Notes
- Uses **yfinance** daily data.
- Uses adjusted close returns for simplicity (dividends/splits-adjusted).
- This is educational; not financial advice.
