# Day 008 — Walk-forward Minimum-Variance ETF Portfolio

Build a tiny, self-contained backtest that rebalances **monthly** into a **long-only minimum-variance** portfolio using a trailing covariance estimate, then compares it to an equal-weight baseline.

- Data: free daily prices via `yfinance`
- Assets (default): `SPY, QQQ, IEF, GLD`
- Rebalance: month-end
- Lookback window: 252 trading days (≈ 1 year)

## How to run

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 day-008-walkforward-minvar-etf-portfolio/run.py
```

Outputs:
- `day-008-walkforward-minvar-etf-portfolio/equity_curves.png`
- Printed performance summary in the terminal

## Notes
This is intentionally lightweight and educational (not investment advice). The "min-var" solution is computed in closed form and then projected to **long-only** by clipping negative weights to 0 and re-normalizing.
