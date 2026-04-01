# Day 015 — Volume Shock vs Next-Day Returns (SPY)

Hypothesis: unusually **high** (or **low**) trading volume may be associated with different **next-day** return behavior.

This mini-project:
- downloads daily OHLCV via **yfinance**
- computes a rolling **volume z-score**
- buckets days by z-score (tails vs normal)
- compares the distribution of **next-day returns** across buckets

## How to run

From repo root:

```bash
pip install -r requirements.txt
python3 day-015-volume-shock-next-day-returns/main.py --ticker SPY --start 2000-01-01
```

Outputs (tables + plots) are written to:

- `day-015-volume-shock-next-day-returns/outputs/`

## Notes
- Uses **adjusted** prices for returns (when available) to reduce corporate-action artifacts.
- This is an educational statistical check, not a trading recommendation.
