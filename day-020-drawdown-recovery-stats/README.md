# Day 020 — Drawdowns & Recovery Time Stats (SPY)

A tiny study of **drawdown episodes** (peak → trough → recovery) and how long markets spend “underwater”.

We:
- download daily prices with `yfinance`
- compute the drawdown (price / running max − 1)
- segment drawdown episodes
- report the biggest drawdowns and recovery durations
- save a few plots (equity curve + underwater drawdown chart + recovery histogram)

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python day-020-drawdown-recovery-stats/analyze.py --ticker SPY --start 1993-01-01
```

Outputs (plots + CSV) are written to:

```
 day-020-drawdown-recovery-stats/outputs/
```

## Notes
- Uses **Adj Close** when available.
- The most recent drawdown episode may be **unrecovered** (no new high yet); it’s still reported.
