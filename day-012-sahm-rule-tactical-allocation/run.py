"""Day 012 — Sahm Rule Tactical Allocation

A minimal monthly backtest:
- Risk-on: SPY
- Risk-off: cash proxy from 3M T-bill rate (FRED: TB3MS)

Signal:
- Sahm-style trigger using unemployment rate (FRED: UNRATE)
  trigger_t = (3mo avg UNRATE)_t - min_{last 12 mo}(3mo avg UNRATE)
  risk_off when trigger >= 0.50 percentage points

To reduce lookahead, the strategy uses signal shifted by 1 month.

Outputs:
- prints summary stats
- saves equity curve plot to outputs/equity_curves.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_fred_series(series: str) -> pd.Series:
    """Fetch a FRED series via the public CSV endpoint (no API key)."""
    url = FRED_CSV.format(series=series)
    df = pd.read_csv(url)
    # FRED uses 'observation_date' (older examples sometimes show 'DATE')
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = df[date_col]
    s.name = series
    return s.dropna()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


@dataclass
class Stats:
    cagr: float
    ann_vol: float
    sharpe: float
    max_dd: float


def perf_stats(returns: pd.Series, periods_per_year: int = 12) -> Stats:
    returns = returns.dropna()
    if len(returns) == 0:
        return Stats(np.nan, np.nan, np.nan, np.nan)

    equity = (1 + returns).cumprod()
    years = len(returns) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    ann_ret = returns.mean() * periods_per_year
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

    mdd = max_drawdown(equity)
    return Stats(float(cagr), float(ann_vol), float(sharpe), float(mdd))


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- Prices (daily) -> monthly returns
    px = yf.download("SPY", start="1993-01-01", auto_adjust=True, progress=False)
    # yfinance sometimes returns MultiIndex columns even for a single ticker
    if isinstance(px.columns, pd.MultiIndex):
        spy = px["Close"]["SPY"].rename("SPY")
    else:
        spy = px["Close"].rename("SPY")

    spy_m = spy.resample("ME").last()
    r_spy = spy_m.pct_change().rename("SPY")

    # --- Macro series (monthly)
    unrate = fetch_fred_series("UNRATE")  # percent
    tb3ms = fetch_fred_series("TB3MS")    # percent annualized

    # Align to month-end index
    unrate = unrate.resample("ME").last()
    tb3ms = tb3ms.resample("ME").last()

    # Sahm-style trigger
    unrate_3ma = unrate.rolling(3).mean()
    unrate_3ma_min12 = unrate_3ma.rolling(12).min()
    sahm_gap = (unrate_3ma - unrate_3ma_min12).rename("sahm_gap")
    risk_off_raw = (sahm_gap >= 0.50).rename("risk_off")

    # Cash monthly return approximation from annualized % rate
    r_cash = (tb3ms / 100.0 / 12.0).rename("CASH")

    # Combine & align
    df = pd.concat([r_spy, r_cash, risk_off_raw], axis=1, sort=False).dropna()

    # Use signal with 1-month delay to reduce lookahead
    df["risk_off_lag"] = df["risk_off"].shift(1).fillna(False)

    # Strategy: if risk-off -> cash else SPY
    df["STRAT"] = np.where(df["risk_off_lag"], df["CASH"], df["SPY"]).astype(float)

    # Equity curves
    equity = (1 + df[["SPY", "STRAT"]]).cumprod()

    # Stats
    s_bh = perf_stats(df["SPY"])
    s_st = perf_stats(df["STRAT"])

    print("=== Day 012: Sahm Rule Tactical Allocation (Monthly) ===")
    print(f"Sample: {df.index.min().date()} → {df.index.max().date()}  (n={len(df)})")
    print()
    print("Buy & Hold SPY")
    print(f"  CAGR     : {s_bh.cagr:6.2%}")
    print(f"  Ann Vol  : {s_bh.ann_vol:6.2%}")
    print(f"  Sharpe   : {s_bh.sharpe:6.2f}")
    print(f"  Max DD   : {s_bh.max_dd:6.2%}")
    print()
    print("Strategy (Risk-off => Cash)")
    print(f"  CAGR     : {s_st.cagr:6.2%}")
    print(f"  Ann Vol  : {s_st.ann_vol:6.2%}")
    print(f"  Sharpe   : {s_st.sharpe:6.2f}")
    print(f"  Max DD   : {s_st.max_dd:6.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    equity.plot(ax=ax)
    ax.set_title("Equity Curves — SPY vs Sahm-style Risk-off Timing")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)

    # Shade risk-off periods (lagged, i.e., when strategy actually in cash)
    risk_off = df["risk_off_lag"].astype(bool)
    if risk_off.any():
        # contiguous spans
        in_span = False
        start = None
        for t, flag in risk_off.items():
            if flag and not in_span:
                start = t
                in_span = True
            if in_span and (not flag):
                ax.axvspan(start, t, color="gray", alpha=0.15, lw=0)
                in_span = False
        if in_span:
            ax.axvspan(start, risk_off.index[-1], color="gray", alpha=0.15, lw=0)

    out_path = os.path.join(out_dir, "equity_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
