import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def fred_series(series_id: str) -> pd.Series:
    """Fetch a FRED series as a pandas Series using the fredgraph CSV endpoint.

    No API key required.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    # columns: DATE, <series_id>
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = series_id
    return s


def main():
    # --- Data ---
    start = "1990-01-01"

    # Corporate credit spread proxy: Baa - 10y Treasury (both in %)
    baa = fred_series("BAA")  # Moody's Seasoned Baa Corporate Bond Yield
    dgs10 = fred_series("DGS10")  # 10-Year Treasury Constant Maturity Rate

    spread = (baa - dgs10).rename("BAA_minus_DGS10")

    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)["Close"].rename("SPY")

    df = pd.concat([spread, spy], axis=1).dropna()

    # --- Features / regimes ---
    # Smooth spread a bit (monthly-ish) while keeping daily alignment.
    df["spread_sma_20"] = df["BAA_minus_DGS10"].rolling(20).mean()
    df = df.dropna()

    # Define regimes by spread percentiles (low/medium/high credit stress)
    q1, q2 = df["spread_sma_20"].quantile([0.33, 0.66])

    def regime(x):
        if x <= q1:
            return "Low spread (risk-on)"
        if x <= q2:
            return "Mid spread"
        return "High spread (risk-off)"

    df["regime"] = df["spread_sma_20"].apply(regime)

    # --- Forward returns ---
    # Use trading-day approximations
    horizons = {"1M": 21, "3M": 63, "6M": 126}
    for label, h in horizons.items():
        df[f"fwd_ret_{label}"] = df["SPY"].shift(-h) / df["SPY"] - 1.0

    df = df.dropna()

    # --- Summary table ---
    summary = (
        df.groupby("regime")[[f"fwd_ret_{k}" for k in horizons.keys()]]
        .agg(["mean", "median", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
    )
    summary.columns = [
        f"{col[0]}_{col[1] if col[1] not in ('<lambda_0>', '<lambda_1>') else ('p25' if col[1]=='<lambda_0>' else 'p75')}"
        for col in summary.columns
    ]

    print("=== Credit spread regime forward return summary (SPY) ===")
    print(summary.round(4))
    print("\nRegime counts:")
    print(df["regime"].value_counts())

    # --- Plots ---
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df.index, df["SPY"], color="black", lw=1)
    axes[0].set_title("SPY (adjusted close) with credit spread regimes")
    axes[0].set_ylabel("Price")

    colors = {
        "Low spread (risk-on)": "#2ca02c",
        "Mid spread": "#ff7f0e",
        "High spread (risk-off)": "#d62728",
    }

    # Shade regimes
    for reg, g in df.groupby((df["regime"] != df["regime"].shift()).cumsum()):
        rname = g["regime"].iloc[0]
        axes[0].axvspan(g.index[0], g.index[-1], alpha=0.10, color=colors[rname])

    axes[1].plot(df.index, df["BAA_minus_DGS10"], lw=0.8, label="BAA - DGS10")
    axes[1].plot(df.index, df["spread_sma_20"], lw=1.2, label="20d SMA")
    axes[1].axhline(q1, ls="--", lw=1, color="gray")
    axes[1].axhline(q2, ls="--", lw=1, color="gray")
    axes[1].set_ylabel("Spread (%)")
    axes[1].legend(loc="upper right")

    # Boxplots of forward returns
    order = ["Low spread (risk-on)", "Mid spread", "High spread (risk-off)"]
    data = [df.loc[df["regime"] == r, "fwd_ret_3M"].values for r in order]
    axes[2].boxplot(data, labels=order, showfliers=False)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_title("SPY 3M forward returns by credit spread regime")
    axes[2].set_ylabel("Forward return")

    fig.tight_layout()
    out = "credit_spread_regimes_spy.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
