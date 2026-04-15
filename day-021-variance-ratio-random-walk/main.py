import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass
class VRResult:
    k: int
    n: int
    vr: float
    z_homosked: float
    z_heterosked: float


def variance_ratio_test(returns: pd.Series, k: int) -> VRResult:
    """Lo–MacKinlay variance ratio test.

    Parameters
    ----------
    returns : pd.Series
        1-period returns (typically daily), indexed by date.
    k : int
        Aggregation horizon.

    Returns
    -------
    VRResult
        VR(k) plus homoskedastic and heteroskedastic-robust z statistics.

    Notes
    -----
    This follows common implementations of Lo & MacKinlay (1988).
    The heteroskedastic-robust statistic is generally preferred for finance.
    """
    r = returns.dropna().astype(float).values
    n = len(r)
    if n <= k + 2:
        raise ValueError(f"Not enough data for k={k}: n={n}")

    mu = r.mean()
    x = r - mu

    # 1-step variance
    sigma2_1 = (x @ x) / n

    # k-step variance from overlapping k-period sums
    # y_t = sum_{j=0}^{k-1} x_{t-j}, for t=k-1..n-1
    y = np.convolve(x, np.ones(k), mode="valid")
    sigma2_k = (y @ y) / (len(y) * k)

    vr = sigma2_k / sigma2_1

    # Homoskedastic variance of VR(k)
    phi = (2 * (2 * k - 1) * (k - 1)) / (3 * k * n)
    z_h = (vr - 1.0) / math.sqrt(phi) if phi > 0 else float("nan")

    # Heteroskedastic-robust variance (Lo–MacKinlay)
    denom = (x @ x) ** 2
    theta = 0.0
    for j in range(1, k):
        # delta_j = sum x_t^2 x_{t-j}^2 / (sum x_t^2)^2
        prod = (x[j:] ** 2) * (x[:-j] ** 2)
        delta_j = prod.sum() / denom
        weight = (2.0 * (k - j) / k) ** 2
        theta += weight * delta_j

    z_het = (vr - 1.0) / math.sqrt(theta) if theta > 0 else float("nan")

    return VRResult(k=k, n=n, vr=float(vr), z_homosked=float(z_h), z_heterosked=float(z_het))


def rolling_vr(returns: pd.Series, k: int, window: int) -> pd.Series:
    """Rolling VR(k) using the same VR definition as in variance_ratio_test."""

    def _vr_on_window(x: np.ndarray) -> float:
        r = x.astype(float)
        n = len(r)
        mu = r.mean()
        z = r - mu
        sigma2_1 = (z @ z) / n
        y = np.convolve(z, np.ones(k), mode="valid")
        sigma2_k = (y @ y) / (len(y) * k)
        return float(sigma2_k / sigma2_1)

    return returns.dropna().rolling(window).apply(lambda s: _vr_on_window(s.values), raw=False)


def main():
    ticker = "SPY"
    start = "1993-01-01"

    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if px.empty:
        raise RuntimeError("No data returned from yfinance.")

    adj = px["Close"].rename("price")
    rets = np.log(adj).diff().rename("log_ret")

    ks = [2, 5, 10, 20]
    results: List[VRResult] = [variance_ratio_test(rets, k) for k in ks]

    out = pd.DataFrame(
        {
            "k": [r.k for r in results],
            "n": [r.n for r in results],
            "VR(k)": [r.vr for r in results],
            "z_homosked": [r.z_homosked for r in results],
            "z_heterosked": [r.z_heterosked for r in results],
        }
    )

    # Simple 2-sided p-values using normal approximation
    # (kept dependency-free; scipy would be nicer but is intentionally avoided)
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    out["p_homosked"] = out["z_homosked"].map(lambda z: 2.0 * (1.0 - norm_cdf(abs(z))))
    out["p_heterosked"] = out["z_heterosked"].map(lambda z: 2.0 * (1.0 - norm_cdf(abs(z))))

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    print("\nLo–MacKinlay Variance Ratio Test (SPY daily log returns)\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    out.to_csv("output_vr_table.csv", index=False)

    # Rolling VR(5) on a 5-year window
    k_roll = 5
    window = 252 * 5
    rv = rolling_vr(rets, k=k_roll, window=window)

    fig, ax = plt.subplots(figsize=(10, 4))
    rv.plot(ax=ax, lw=1.2, color="black")
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_title(f"Rolling Variance Ratio VR({k_roll}) — {ticker} (window={window} trading days)")
    ax.set_ylabel("VR")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("rolling_vr5.png", dpi=160)


if __name__ == "__main__":
    main()
