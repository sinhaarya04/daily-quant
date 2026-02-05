"""Day 004 — Time-series splits + leakage checklist

This script is intentionally small and readable.

It demonstrates:
- walk-forward (anchored) splitting
- why random splits leak on time-series
- how a single leaky feature can inflate performance

Deps: numpy, pandas, scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    """A single time-series split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    anchored: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Split]]:
    """Generate walk-forward splits.

    Parameters
    ----------
    index: DatetimeIndex
        Time index (must be sorted ascending).
    train_size: int
        Number of observations in train window.
    test_size: int
        Number of observations in test window.
    step_size: int | None
        How far to advance each split. Defaults to test_size.
    anchored: bool
        If True, training window expands from the start (anchored walk-forward).
        If False, training window rolls forward with fixed length.

    Yields
    ------
    (train_idx, test_idx, Split)
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas.DatetimeIndex")

    if len(index) < train_size + test_size:
        raise ValueError("Not enough data for one split")

    if step_size is None:
        step_size = test_size

    # Ensure sorted
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")

    start = 0
    # First test window starts after initial train window
    test_start = train_size

    while test_start + test_size <= len(index):
        if anchored:
            train_start = 0
        else:
            train_start = test_start - train_size

        train_end = test_start  # exclusive
        test_end = test_start + test_size  # exclusive

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        meta = Split(
            train_start=index[train_idx[0]],
            train_end=index[train_idx[-1]],
            test_start=index[test_idx[0]],
            test_end=index[test_idx[-1]],
        )

        yield train_idx, test_idx, meta

        test_start += step_size


def assert_no_overlap(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    inter = np.intersect1d(train_idx, test_idx)
    if inter.size:
        raise AssertionError(f"Train/Test overlap at positions: {inter[:10]}")


def make_synthetic_prices(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    # Random-walk-ish returns
    rets = rng.normal(0.0, 0.01, size=n)
    px = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": px}, index=dates)
    df["ret_1d"] = df["close"].pct_change()
    # Forward return (label): tomorrow's return sign
    df["fwd_ret_1d"] = df["ret_1d"].shift(-1)
    df["y"] = (df["fwd_ret_1d"] > 0).astype(int)
    return df.dropna()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Safe features: all available at time t for predicting t+1
    out["ret_1d_lag1"] = out["ret_1d"].shift(1)
    out["vol_20_lag1"] = out["ret_1d"].rolling(20).std().shift(1)

    # Intentionally leaky feature: uses *future* info (forward return)
    out["LEAK_fwd_ret"] = out["fwd_ret_1d"]  # <--- leakage

    return out.dropna()


def eval_auc(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

    model = LogisticRegression(max_iter=2000)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


def main() -> None:
    df = make_synthetic_prices()
    df = build_features(df)

    safe_cols = ["ret_1d_lag1", "vol_20_lag1"]
    leaky_cols = safe_cols + ["LEAK_fwd_ret"]

    X_safe = df[safe_cols]
    X_leaky = df[leaky_cols]
    y = df["y"]

    print("\n=== Walk-forward split demo (anchored) ===")
    splits = list(
        walk_forward_splits(
            df.index,
            train_size=500,
            test_size=100,
            step_size=100,
            anchored=True,
        )
    )
    for i, (tr, te, meta) in enumerate(splits[:3], start=1):
        assert_no_overlap(tr, te)
        print(
            f"Split {i}: train {meta.train_start.date()} → {meta.train_end.date()} | "
            f"test {meta.test_start.date()} → {meta.test_end.date()}"
        )

    auc_safe = []
    auc_leaky = []
    for tr, te, _meta in splits:
        auc_safe.append(eval_auc(X_safe, y, tr, te))
        auc_leaky.append(eval_auc(X_leaky, y, tr, te))

    print("\nAUC (safe features)  :", round(float(np.mean(auc_safe)), 4), "+/-", round(float(np.std(auc_safe)), 4))
    print("AUC (LEAKY features) :", round(float(np.mean(auc_leaky)), 4), "+/-", round(float(np.std(auc_leaky)), 4))

    print("\n=== Why random split is dangerous ===")
    Xtr, Xte, ytr, yte = train_test_split(X_safe, y, test_size=0.25, random_state=42, shuffle=True)
    m = LogisticRegression(max_iter=2000)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    print("Random-split AUC (safe features):", round(roc_auc_score(yte, p), 4))

    print(
        "\nLeakage checklist:\n"
        "- [ ] All features are lagged so they only use info available at prediction time\n"
        "- [ ] Any scaler/encoder is fit on TRAIN only (use Pipeline)\n"
        "- [ ] Splits are walk-forward (no shuffle), with no overlap\n"
        "- [ ] Labels are aligned correctly (forward returns shifted, rows dropped consistently)\n"
        "- [ ] Rolling features use past-only windows (shift after rolling)\n"
        "- [ ] Corporate actions / survivorship bias considered (if using equities universe)\n"
    )


if __name__ == "__main__":
    main()
