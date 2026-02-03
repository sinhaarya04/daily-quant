# 30 Days of Projects — daily-quant (ML-heavy)

Goal: **~30-minute quant mini-projects** with small, reproducible scripts/notebooks.
Each day should ideally include: `README.md` + `*.py` (or notebook) + minimal deps.

## Days 001–003 (your 30-day sequence)
1. **Day 001 — EWMA Volatility (VaR sketch)**
   - Deliverable: compute EWMA vol on log returns, plot; quick VaR estimate.
2. **Day 002 — GARCH(1,1) Volatility**
   - Deliverable: fit GARCH(1,1) (t-errors), plot conditional vol, print params.
3. **Day 003 — Rolling beta vs SPY (CAPM regression)**
   - Deliverable: rolling OLS beta; visualize regime shifts.

> Note: your repo currently also has a folder `day-003-realized-vs-vix/` comparing realized vol to VIX. If you want your numbering to match the 30-day challenge exactly, we can rename/re-number folders.

## Days 004–030 (proposed)
4. **Day 004 — Realized Volatility vs VIX**
   - Deliverable: rolling realized vol vs VIX plot + scatter + correlation/fit.
5. **Day 005 — Data sanity + stylized facts**
   - Returns distribution, autocorr, volatility clustering, QQ-plot.
6. **Day 006 — PCA on sector returns (risk factors)**
   - PCA on XLF/XLK/…; explain variance; first PC as “market” proxy.
7. **Day 007 — Simple factor model (value/momentum proxies)**
   - Construct naive factors from ETF baskets; regress returns on factors.
8. **Day 008 — Feature engineering: lag/rolling features**
   - Build a feature table: lagged returns, rolling mean/vol, RSI, MACD.
9. **Day 009 — Train/test split for time series + baseline predictor**
   - Walk-forward split; baseline: predict sign with last return / rolling mean.
10. **Day 010 — Logistic regression for direction (classification)**
   - Predict next-day up/down using engineered features; report AUC/accuracy.
11. **Day 011 — Regularization (L1/L2) + stability**
   - Compare ridge vs lasso; inspect coefficient stability across windows.
12. **Day 012 — Tree models (RandomForest / XGBoost-light)**
   - Feature importance + calibration; avoid lookahead.
13. **Day 013 — Purged time-series CV (leakage control)**
   - Implement purged/embargoed CV for financial labels.
14. **Day 014 — Labeling: triple-barrier (Lopez de Prado)**
   - Create event-based labels; compare to fixed-horizon labels.
15. **Day 015 — Probabilities to positions (sizing)**
   - Map model probs → position size; compare threshold vs Kelly-fraction-lite.
16. **Day 016 — Backtesting scaffold (costs + slippage)**
   - Minimal vectorized backtest; include transaction costs; equity curve.
17. **Day 017 — Risk metrics dashboard**
   - Sharpe, Sortino, max drawdown, turnover, hit-rate; rolling metrics.
18. **Day 018 — Vol targeting**
   - Use realized vol to scale positions to a target vol; compare before/after.
19. **Day 019 — Regime detection with HMM (or k-means on vol/returns)**
   - Identify regimes; conditional performance by regime.
20. **Day 020 — Clustering stocks by correlation**
   - Hierarchical clustering; plot dendrogram/cluster heatmap.
21. **Day 021 — Mean reversion: pairs trading (distance / cointegration)**
   - Find candidate pair; z-score spread; simple entry/exit; backtest.
22. **Day 022 — Momentum: cross-sectional top-k**
   - Universe of ETFs/stocks; rank by lookback return; rebalance monthly.
23. **Day 023 — Portfolio optimization (mean-variance with constraints)**
   - Minimum variance / risk parity baseline; compare allocations.
24. **Day 024 — Shrinkage covariance (Ledoit–Wolf)**
   - Show effect on optimized weights and realized risk.
25. **Day 025 — Forecast volatility with ML (regression)**
   - Predict next 5–20d realized vol with ridge/GBM; evaluate RMSE.
26. **Day 026 — LSTM/GRU toy model on returns/vol (careful eval)**
   - Tiny network; walk-forward; compare to linear baseline.
27. **Day 027 — Attention/Transformer-lite for time series (toy)**
   - Small transformer encoder on features; emphasize leakage-safe evaluation.
28. **Day 028 — Options-ish: implied vs realized spread signal**
   - Use VIX–RV spread as feature; test if spread predicts future RV/returns.
29. **Day 029 — Interpretability: SHAP (or permutation importance)**
   - Explain a model’s drivers; stability across time.
30. **Day 030 — End-to-end report + lessons learned**
   - One markdown report: what worked, what didn’t, next iteration plan.

## Suggested folder naming
Use: `day-XXX-topic/` e.g. `day-010-logreg-direction/`.

## Suggested minimal deps
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `yfinance`
- Optional later: `statsmodels`, `arch`, `hmmlearn`, `cvxpy`, `shap`, `torch`
