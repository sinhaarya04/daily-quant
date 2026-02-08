"""Day 006: Baseline ML classifier (time-series safe)

Trains quick baselines to predict whether next-day return is positive.

Usage:
  python day-006-baseline-ml-classifier/train.py --ticker SPY --start 2010-01-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Import Day 005 feature builder
# Repo structure: day-005-feature-pipeline-v1/features.py
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY5_DIR = REPO_ROOT / "day-005-feature-pipeline-v1"
sys.path.insert(0, str(DAY5_DIR))

try:
    from features import build_features  # type: ignore
except Exception as e:
    raise SystemExit(
        "Failed to import build_features from Day 005. "
        "Make sure day-005-feature-pipeline-v1/features.py exists.\n"
        f"Error: {e}"
    )


@dataclass
class Result:
    name: str
    accuracy: float
    roc_auc: float | None


def download_ohlcv(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data downloaded for {ticker}")
    # Standardize columns
    df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
    # yfinance sometimes returns multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]
    return df


def make_label(df: pd.DataFrame) -> pd.Series:
    """Binary label: 1 if next-day close-to-close return > 0."""
    ret1 = df["close"].pct_change().shift(-1)
    y = (ret1 > 0).astype(int)
    return y


def chrono_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8):
    n = len(X)
    cut = int(n * train_frac)
    X_train, X_test = X.iloc[:cut].copy(), X.iloc[cut:].copy()
    y_train, y_test = y.iloc[:cut].copy(), y.iloc[cut:].copy()
    return X_train, X_test, y_train, y_test


def evaluate_clf(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> Result:
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    roc = None
    # Try ROC-AUC if we have predict_proba
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X_test)[:, 1]
            roc = float(roc_auc_score(y_test, p))
        except Exception:
            roc = None

    return Result(name=name, accuracy=acc, roc_auc=roc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()

    raw = download_ohlcv(args.ticker, start=args.start, end=args.end)

    # Day 005 features
    feat = build_features(raw)

    # Label aligned to the same index
    y = make_label(raw).reindex(feat.index)

    # Drop NaNs from features/label (label NaN at end due to shift)
    data = feat.copy()
    data["y"] = y
    data = data.dropna()

    X = data.drop(columns=["y"])
    y = data["y"].astype(int)

    X_train, X_test, y_train, y_test = chrono_split(X, y, train_frac=args.train_frac)

    models = []

    # Baseline 1: Logistic Regression (scaled)
    logreg = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    models.append(("LogReg", logreg))

    # Baseline 2: RandomForest (no scaling needed)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1,
    )
    models.append(("RandomForest", rf))

    results: list[Result] = []
    for name, m in models:
        m.fit(X_train, y_train)
        results.append(evaluate_clf(name, m, X_test, y_test))

    # Print
    print(f"Ticker: {args.ticker}")
    print(f"Rows: total={len(X):,} train={len(X_train):,} test={len(X_test):,}")
    print("\nResults:")
    for r in results:
        if r.roc_auc is None:
            print(f"- {r.name:12s}  accuracy={r.accuracy:.3f}")
        else:
            print(f"- {r.name:12s}  accuracy={r.accuracy:.3f}  roc_auc={r.roc_auc:.3f}")


if __name__ == "__main__":
    main()
