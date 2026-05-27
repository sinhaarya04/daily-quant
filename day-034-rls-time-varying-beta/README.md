# Day 034 — Time-Varying Beta via Recursive Least Squares (RLS)

Goal: estimate a **dynamic beta** of QQQ relative to SPY using a simple **Recursive Least Squares** filter (a lightweight cousin of a Kalman filter), and compare it to a standard rolling OLS beta.

Data: daily adjusted closes from **yfinance**.

## How to run

From repo root:

```bash
pip install -r requirements.txt
python3 day-034-rls-time-varying-beta/rls_beta.py --asset QQQ --benchmark SPY --start 2010-01-01
```

This will write plots to `day-034-rls-time-varying-beta/out/`.

## What to look for

- RLS beta should react faster than a long rolling window, but smoother than a very short window.
- Try different `--lambda_` (forgetting factor) values:
  - closer to **1.0** → smoother/slower
  - smaller (e.g. **0.97**) → faster/more jittery
