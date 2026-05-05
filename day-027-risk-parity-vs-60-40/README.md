# Day 027 — Risk Parity (Vol-Inverse) vs 60/40 (SPY/TLT)

Goal: build a tiny, self-contained backtest comparing:
- **60/40**: fixed weights (SPY 60%, TLT 40%), monthly rebalance
- **Risk parity (simple)**: weights proportional to **1 / trailing vol** (60 trading days), monthly rebalance

Data: daily adjusted close from **yfinance**.

## How to run
From the repo root:

```bash
pip install -r requirements.txt
python day-027-risk-parity-vs-60-40/risk_parity_vs_60_40.py
```

Outputs (saved in the day folder):
- `equity_curve.png`
- `weights.png`

## Notes
- This is a deliberately simple “risk parity” proxy (inverse-vol weighting), not full covariance-based risk parity.
- Results depend on the chosen lookback (60d) and rebalance frequency (monthly).
