"""Day 023 — Financial Stress Regime Filter (FRED STLFSI4 → SPY/Cash)

Idea:
  - Use the St. Louis Fed Financial Stress Index (STLFSI4) as a regime indicator.
  - Hold SPY when stress is below a threshold; otherwise sit in cash.

Design goals:
  - Self-contained, minimal deps (numpy/pandas/matplotlib/yfinance).
  - Correct time alignment and no lookahead (signal is lagged by 1 day).

Not investment advice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4"


def fetch_fred_series(csv_url: str = FRED_CSV_URL) -> pd.Series:
    df = pd.read_csv(csv_url)

    # FRED exports sometimes use DATE, sometimes observation_date.
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    value_col = "STLFSI4" if "STLFSI4" in df.columns else df.columns.difference([date_col])[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # FRED sometimes uses '.' for missing.
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.name = "STLFSI4"
    return s.dropna()


def fetch_spy(start: str) -> pd.Series:
    df = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No SPY data returned from yfinance")

    close = df["Close"]
    # Newer yfinance returns MultiIndex columns: (Price, Ticker).
    if isinstance(close, pd.DataFrame):
        px = close["SPY"].copy()
    else:
        px = close.copy()

    px.name = "SPY"
    return px


@dataclass
class Perf:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float


def perf_stats(daily_rets: pd.Series, periods_per_year: int = 252) -> Perf:
    r = daily_rets.dropna()
    if len(r) < 10:
        return Perf(np.nan, np.nan, np.nan, np.nan)

    eq = (1.0 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / vol if vol and vol > 0 else np.nan

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = dd.min()

    return Perf(cagr=cagr, vol=vol, sharpe=sharpe, max_dd=max_dd)


def make_strategy(spy_px: pd.Series, stress: pd.Series, threshold: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Align to business days and forward-fill weekly stress values.
    idx = pd.date_range(spy_px.index.min(), spy_px.index.max(), freq="B")
    spy_px = spy_px.reindex(idx).ffill()
    stress = stress.reindex(idx).ffill()

    spy_ret = spy_px.pct_change()
    spy_ret.name = "spy_ret"

    # Signal uses yesterday's stress reading.
    invested = (stress.shift(1) < threshold).astype(int)
    invested.name = "invested"

    strat_ret = invested * spy_ret
    strat_ret.name = "strat_ret"

    return spy_ret, strat_ret, invested


def plot_equity(spy_ret: pd.Series, strat_ret: pd.Series, out_path: str, title: str) -> None:
    eq_bh = (1 + spy_ret.fillna(0)).cumprod()
    eq_st = (1 + strat_ret.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq_bh.index, eq_bh.values, label="Buy & Hold (SPY)", linewidth=1.6)
    ax.plot(eq_st.index, eq_st.values, label="Stress filter", linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2000-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--threshold", type=float, default=0.0, help="Stress threshold; above => cash")
    p.add_argument(
        "--out",
        type=str,
        default="day-023-financial-stress-regime-filter/outputs/equity.png",
        help="Output plot path",
    )
    args = p.parse_args()

    stress = fetch_fred_series()
    spy_px = fetch_spy(args.start)

    spy_ret, strat_ret, _invested = make_strategy(spy_px, stress, threshold=args.threshold)

    stats = pd.DataFrame(
        {
            "Buy&Hold": perf_stats(spy_ret).__dict__,
            f"Stress<th{args.threshold:g}": perf_stats(strat_ret).__dict__,
        }
    )

    # Nicer formatting for printing (keep as strings to avoid dtype issues).
    fmt = {
        "cagr": lambda x: f"{x:.2%}",
        "vol": lambda x: f"{x:.2%}",
        "sharpe": lambda x: f"{x:.2f}",
        "max_dd": lambda x: f"{x:.2%}",
    }

    printable = pd.DataFrame(index=stats.index, columns=stats.columns, dtype=object)
    for row in stats.index:
        f = fmt.get(row, lambda x: str(x))
        printable.loc[row] = [f(v) if pd.notna(v) else "nan" for v in stats.loc[row].values]

    print("\nPerformance (annualized where applicable)")
    print(printable)

    plot_equity(
        spy_ret,
        strat_ret,
        out_path=args.out,
        title=f"Day 023 — FRED STLFSI4 stress filter (threshold={args.threshold:g})",
    )


if __name__ == "__main__":
    main()
