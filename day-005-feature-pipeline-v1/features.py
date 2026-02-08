"""Day 005: Feature pipeline v1 (tabular) + data validation.

Design principles:
- Every feature is computed using information available at or before time t.
- The label is computed for time t based on t+H (future), but stored aligned to t.
- Return a clean (X, y) with matching index, no NaNs, and documented columns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit(
        "Missing dependency yfinance. Run: pip install yfinance"
    ) from e


@dataclass(frozen=True)
class FeatureSpec:
    horizon: int = 1
    ret_lags: tuple[int, ...] = (1, 2, 5, 10)
    vol_windows: tuple[int, ...] = (5, 10, 20)
    ma_windows: tuple[int, ...] = (5, 10, 20)


def download_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker=}, {start=}")

    # Standardize columns
    df = df.rename(columns=str.lower)
    # yfinance returns: open high low close adj close volume
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})

    return df


def make_features(prices: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned to the same datetime index.

    X columns are *features at time t*.
    y is label for time t defined using future return over horizon H.
    """

    close = prices["close"].copy()

    # --- Target (future return) ---
    fwd_ret = close.pct_change(spec.horizon).shift(-spec.horizon)
    # binary direction label (next-horizon up/down)
    y = (fwd_ret > 0).astype(int).rename("y_up")

    X = pd.DataFrame(index=prices.index)

    # --- Lagged returns ---
    for lag in spec.ret_lags:
        X[f"ret_{lag}d"] = close.pct_change(lag)

    # --- Rolling volatility of 1d returns ---
    r1 = close.pct_change(1)
    for w in spec.vol_windows:
        X[f"vol_{w}d"] = r1.rolling(w).std()

    # --- Moving-average ratios (close / MA - 1) ---
    for w in spec.ma_windows:
        ma = close.rolling(w).mean()
        X[f"ma_ratio_{w}d"] = (close / ma) - 1.0

    # IMPORTANT: ensure all features only use data up to t.
    # (All above are rolling/pct_change backward-looking, so OK.)

    # --- Final clean dataset ---
    df = X.join(y, how="inner")

    # Drop rows with missing values (from rolling windows + last horizon rows)
    df = df.dropna()

    y_clean = df.pop("y_up")
    X_clean = df

    validate_xy(X_clean, y_clean)
    return X_clean, y_clean


def validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    # index alignment
    if not X.index.equals(y.index):
        raise AssertionError("X and y indices are not identical")

    # no NaNs
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise AssertionError(f"NaNs present in X columns: {bad}")
    if y.isna().any():
        raise AssertionError("NaNs present in y")

    # types
    if not np.issubdtype(y.dtype, np.integer):
        raise AssertionError(f"Expected integer classification label, got {y.dtype}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--write", action="store_true", help="Write features CSV to data/")
    args = ap.parse_args()

    spec = FeatureSpec(horizon=args.horizon)
    prices = download_ohlcv(args.ticker, args.start)
    X, y = make_features(prices, spec)

    print(f"{args.ticker}: X shape={X.shape}, y shape={y.shape}")
    print("Features:", list(X.columns))
    print("Date range:", X.index.min().date(), "→", X.index.max().date())
    print("y mean (up rate):", float(y.mean()))

    if args.write:
        out = pd.concat([X, y], axis=1)
        out_dir = "data"
        import os

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"features_{args.ticker}.csv")
        out.to_csv(path)
        print("Wrote:", path)


if __name__ == "__main__":
    main()
