# Day 003 — Rolling Beta vs SPY (CAPM-style)

Estimate a **rolling beta** of an asset vs **SPY** using daily returns.

## What this project does
- Downloads adjusted close prices (Yahoo via `yfinance`)
- Computes daily returns (log returns by default)
- Computes **rolling beta** over a window (default: 60 trading days)
  - 
  \( \beta_t = \frac{\mathrm{Cov}(r^{asset}, r^{mkt})}{\mathrm{Var}(r^{mkt})} \)
- Plots:
  - rolling beta through time
  - asset vs SPY returns (scatter)

## Run
From repo root:

```bash
pip install -r requirements.txt
python day-003-rolling-beta-vs-spy/rolling_beta.py --ticker QQQ --benchmark SPY --start 2015-01-01 --window 60
```

Optional args:
- `--ticker` asset ticker (default `QQQ`)
- `--benchmark` benchmark ticker (default `SPY`)
- `--start` start date (YYYY-MM-DD)
- `--window` rolling window in trading days (default `60`)
- `--simple` use simple returns instead of log returns

## Notes
- Rolling beta is **window-dependent**; shorter windows are noisier.
- This is educational only; not financial advice.
