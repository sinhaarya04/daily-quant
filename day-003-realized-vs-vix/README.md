# Day 003 — Realized Volatility vs VIX

Compare **realized volatility** (from SPY returns) to **VIX** (implied volatility) using free Yahoo Finance data.

## What this project does
- Downloads daily close prices for:
  - `SPY` (to compute realized volatility)
  - `^VIX` (to proxy implied volatility)
- Computes rolling realized volatility (default: 20 trading days), annualized
- Aligns and plots:
  - realized vol vs VIX through time
  - a scatter plot + best-fit line
- Prints summary stats: correlation and a simple linear fit

## Run
From repo root:

```bash
pip install -r requirements.txt
python day-003-realized-vs-vix/rv_vs_vix.py --start 2015-01-01 --window 20
```

Optional args:
- `--ticker` (default `SPY`)
- `--vix_ticker` (default `^VIX`)
- `--window` rolling window in trading days (default `20`)

## Notes
- VIX is quoted in **percent** (e.g., 18.5), while realized vol here is in **decimal** (e.g., 0.185). The script converts them to the same scale before comparing.
- Educational only; not financial advice.
