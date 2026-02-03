# 30 Days of Projects — daily-quant (ML-first)

Goal: **~30-minute ML-oriented quant mini-projects** — small, focused, reproducible.
Each day: `day-XXX-topic/` with `README.md` + script/notebook + minimal deps.

## Anchors (your challenge)
1. **Day 001 — EWMA Volatility (VaR sketch)**
2. **Day 002 — GARCH(1,1) Volatility**
3. **Day 003 — Rolling beta vs SPY (CAPM regression)**

> Repo note: your GitHub repo currently contains `day-003-realized-vs-vix/` as well. If you want strict numbering alignment with the 30-day challenge, we can rename/re-number later.

## ML-first 30-day sequence (not pairs-trading; minimize overlap w/ your other repos)
4. **Time-series splits + leakage checklist**
   - Walk-forward, anchored splits, feature lagging rules; unit test for lookahead.
5. **Feature pipeline v1 (tabular) + data validation**
   - Build a single `make_features()` that outputs a clean (X, y) table + schema checks.
6. **Baselines that don’t embarrass you**
   - Naive (random/last sign), linear regression, ridge; compare with proper backtest metric.
7. **Target engineering: returns vs vol-adjusted returns vs quantiles**
   - Define multiple targets; compare stability and noise.
8. **Probabilistic classification (direction) + calibration**
   - Logistic vs calibrated models (Platt/isotonic); reliability curve.
9. **Gradient boosting for tabular signals**
   - LightGBM/XGBoost or sklearn HistGB; feature importance + stability.
10. **Hyperparameter search (time-series safe)**
   - Random search with walk-forward CV; log results; avoid leakage.
11. **Conformal prediction for trading signals**
   - Predict intervals / set-valued predictions; trade only when confident.
12. **Meta-labeling (Lopez de Prado style)**
   - Primary signal + meta-model that predicts when to trust it.
13. **Cost-aware objective / turnover regularization**
   - Add transaction-cost penalty; evaluate turnover vs performance.
14. **Position sizing from probabilities**
   - Map p→position (linear, Kelly-lite, volatility scaling); compare drawdowns.
15. **Backtest scaffold v2 (event-based) + slippage model**
   - Simple, explicit fills + costs; keep it reproducible.
16. **Model monitoring: drift + performance decay**
   - PSI/KS drift checks; rolling AUC; alert when distribution shifts.
17. **Online learning (streaming updates)**
   - SGDClassifier/PassiveAggressive; compare online vs batch.
18. **Ensembling + stacking (time-series safe)**
   - Blend linear + trees; stacking with out-of-fold predictions.
19. **Unsupervised feature learning on returns (autoencoder / PCA++)**
   - Latent factors as features; compare vs raw engineered features.
20. **Clustering regimes from learned embeddings (not classic HMM)**
   - Cluster latent states; conditional model performance by cluster.
21. **Graph features from correlation networks**
   - Build a correlation graph; node centrality/community as features.
22. **Cross-sectional modeling (many assets) + ranking loss**
   - Learn to rank (pairwise/logit) for top-k selection across tickers.
23. **Labeling with triple-barrier (event labels)**
   - Implement event labels; compare to fixed-horizon.
24. **Imbalanced learning + focal loss / class weights**
   - When labels are sparse; measure precision/recall tradeoffs.
25. **Interpretability: permutation + SHAP stability over time**
   - Does “important” stay important? Visualize drift in explanations.
26. **NLP add-on: FinBERT embeddings for news (upgrade your sentiment project)**
   - Pull headlines (or use a sample set) → embeddings → predictive features.
27. **Multimodal: combine price features + NLP features**
   - Late fusion model; see incremental lift.
28. **Vol surface / options features (but ML-focused)**
   - Build a small options-feature set (IV level, skew proxy) and test marginal value.
29. **Risk model integration: predict returns with uncertainty → optimize**
   - Use predicted mean + variance to size positions; compare to naive sizing.
30. **Final day: reproducible report + lessons learned**
   - One markdown writeup + plots + “what I’d do next”.

## Suggested deps (keep it lean)
- Core: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `yfinance`
- Often useful: `scipy`, `statsmodels`
- Optional ML: `lightgbm`/`xgboost`, `torch`, `shap`, `networkx`, `transformers`
