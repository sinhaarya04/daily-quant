# Day 001 — EWMA Volatility (and a quick VaR sketch)

Compute an **EWMA (RiskMetrics-style) volatility** series on daily returns and plot it.

## What’s inside
- `ewma_volatility.py` — downloads daily prices (Yahoo via `yfinance`), computes log returns, EWMA vol, and plots.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)

pip install -r requirements.txt
python day-001-ewma-volatility/ewma_volatility.py --ticker SPY --lambda 0.94 --years 5
```

## Notes
- EWMA variance recursion:  
  \( \sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2 \)
- Default \(\lambda = 0.94\) is the classic RiskMetrics daily value.

