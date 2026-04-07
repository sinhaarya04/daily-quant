# Day 016 — Rolling Pairs Trading (Z-score of Spread)

A tiny, reproducible **pairs trading** backtest:
- download two assets with `yfinance`
- estimate a **rolling hedge ratio** via rolling OLS (`sklearn`)
- form the **spread** and its **rolling z-score**
- trade mean reversion with simple entry/exit rules + transaction costs

> Educational only. This is a toy backtest (no borrow constraints, slippage model is simplistic, no parameter search discipline, etc.).

## How to run

From repo root:

```bash
pip3 install -r requirements.txt
python3 day-016-rolling-pairs-trading-zscore/pairs_zscore_backtest.py --y KO --x PEP --start 2010-01-01
```

You can change tickers/parameters:

```bash
python3 day-016-rolling-pairs-trading-zscore/pairs_zscore_backtest.py \
  --y SPY --x QQQ \
  --reg-window 90 \
  --z-window 20 \
  --entry 2.0 \
  --exit 0.5 \
  --tc-bps 2
```

## Outputs
- prints summary stats (CAGR/vol/Sharpe/max drawdown, trade count)
- saves plots to `day-016-rolling-pairs-trading-zscore/out/`

