"""Day 016 — Rolling Pairs Trading (Z-score of Spread)

Toy, educational pairs-trading backtest:
- Pulls two adjusted close series via yfinance
- Estimates a rolling hedge ratio (y ~ beta * x) with rolling OLS
- Forms spread = y - beta*x, then rolling z-score
- Trades mean reversion with simple entry/exit rules

Notes:
- This is intentionally minimal and not production-grade.
- Ignores borrow constraints, funding, intraday effects, corporate actions edge cases, etc.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass
class BacktestResult:
    df: pd.DataFrame
    stats: dict
    out_dir: str


def _max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def _annualized_stats(daily_ret: pd.Series, periods_per_year: int = 252) -> dict:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return {
            "n_days": int(len(daily_ret)),
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
        }

    cum = (1.0 + daily_ret).cumprod()
    years = len(daily_ret) / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float((daily_ret.mean() / daily_ret.std(ddof=0)) * np.sqrt(periods_per_year)) if daily_ret.std(ddof=0) > 0 else np.nan

    return {
        "n_days": int(len(daily_ret)),
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": float(cum.iloc[-1] - 1.0),
        "max_drawdown": _max_drawdown(cum),
    }


def download_prices(y: str, x: str, start: str) -> pd.DataFrame:
    # Use auto_adjust to avoid needing separate Adj Close handling.
    data = yf.download([y, x], start=start, auto_adjust=True, progress=False)

    # yfinance returns different column formats depending on tickers.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        # Single ticker: promote to frame
        close = data[["Close"]].rename(columns={"Close": y})

    close = close.rename(columns={y: "y", x: "x"})
    close = close[["y", "x"]].dropna()
    if close.empty:
        raise ValueError("No price data returned. Check tickers/start date.")
    return close


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta for y ~ beta*x with intercept forced to 0.

    For a pairs spread, a no-intercept beta is a reasonable minimal choice:
        beta = Cov(y, x) / Var(x)
    computed on a rolling window.
    """

    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    return beta


def backtest_pairs_zscore(
    prices: pd.DataFrame,
    reg_window: int,
    z_window: int,
    entry: float,
    exit: float,
    tc_bps: float,
) -> BacktestResult:
    df = prices.copy()

    df["beta"] = rolling_beta(df["y"], df["x"], window=reg_window)
    df["spread"] = df["y"] - df["beta"] * df["x"]

    spread_ma = df["spread"].rolling(z_window).mean()
    spread_sd = df["spread"].rolling(z_window).std(ddof=0)
    df["z"] = (df["spread"] - spread_ma) / spread_sd

    # Signals
    # +1 = long spread (long y, short beta*x)
    # -1 = short spread (short y, long beta*x)
    sig = pd.Series(0.0, index=df.index)

    long_entry = df["z"] <= -entry
    short_entry = df["z"] >= entry
    flat_exit = df["z"].abs() <= exit

    position = 0.0
    for t in df.index:
        if np.isnan(df.at[t, "z"]) or np.isnan(df.at[t, "beta"]):
            sig.at[t] = 0.0
            position = 0.0
            continue

        if position == 0.0:
            if long_entry.at[t]:
                position = 1.0
            elif short_entry.at[t]:
                position = -1.0
        else:
            if flat_exit.at[t]:
                position = 0.0

        sig.at[t] = position

    df["pos"] = sig

    # Convert position to weights (gross leverage normalized to 1)
    # long spread:  w_y = +1, w_x = -beta
    # short spread: w_y = -1, w_x = +beta
    w_y = df["pos"]
    w_x = -df["pos"] * df["beta"]

    gross = w_y.abs() + w_x.abs()
    w_y = w_y / gross.replace(0, np.nan)
    w_x = w_x / gross.replace(0, np.nan)
    w_y = w_y.fillna(0.0)
    w_x = w_x.fillna(0.0)

    df["w_y"] = w_y
    df["w_x"] = w_x

    # Returns
    df["ret_y"] = df["y"].pct_change()
    df["ret_x"] = df["x"].pct_change()

    # Use previous day's weights (trade at close, hold next day) to avoid lookahead.
    df["port_ret_gross"] = df["w_y"].shift(1) * df["ret_y"] + df["w_x"].shift(1) * df["ret_x"]

    # Transaction costs: proportional to turnover (sum abs(delta weights)).
    turnover = (df[["w_y", "w_x"]].diff().abs().sum(axis=1)).fillna(0.0)
    df["turnover"] = turnover
    df["tc"] = (tc_bps / 1e4) * df["turnover"]

    df["port_ret"] = df["port_ret_gross"] - df["tc"]
    df["cum"] = (1.0 + df["port_ret"].fillna(0.0)).cumprod()

    # Trade count (approx): count position changes
    trades = int((df["pos"].diff().abs() > 0).sum())

    stats = _annualized_stats(df["port_ret"])
    stats.update(
        {
            "entry": float(entry),
            "exit": float(exit),
            "reg_window": int(reg_window),
            "z_window": int(z_window),
            "tc_bps": float(tc_bps),
            "trades": trades,
        }
    )

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    return BacktestResult(df=df, stats=stats, out_dir=out_dir)


def plot_result(res: BacktestResult, y: str, x: str) -> None:
    df = res.df

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(df.index, df["cum"], label="Equity (cum)")
    axes[0].set_title(f"Day 016 — Rolling Pairs Trading: {y} vs {x}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(df.index, df["z"], label="Spread z-score", color="tab:purple")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(res.stats["entry"], color="red", ls="--", lw=1, label="entry")
    axes[1].axhline(-res.stats["entry"], color="red", ls="--", lw=1)
    axes[1].axhline(res.stats["exit"], color="green", ls=":", lw=1, label="exit")
    axes[1].axhline(-res.stats["exit"], color="green", ls=":", lw=1)
    axes[1].fill_between(df.index, -0.1, 0.1, where=(df["pos"] == 0), color="gray", alpha=0.05, label="flat")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    axes[2].plot(df.index, df["y"], label=y)
    axes[2].plot(df.index, df["x"], label=x)
    axes[2].set_title("Prices (auto-adjusted)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.tight_layout()

    out_path = os.path.join(res.out_dir, f"day016_pairs_{y}_{x}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--y", type=str, default="KO", help="Dependent/target asset (y)")
    p.add_argument("--x", type=str, default="PEP", help="Hedge asset (x)")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--reg-window", type=int, default=90)
    p.add_argument("--z-window", type=int, default=20)
    p.add_argument("--entry", type=float, default=2.0)
    p.add_argument("--exit", type=float, default=0.5)
    p.add_argument("--tc-bps", type=float, default=2.0)
    args = p.parse_args()

    prices = download_prices(args.y, args.x, args.start)
    res = backtest_pairs_zscore(
        prices,
        reg_window=args.reg_window,
        z_window=args.z_window,
        entry=args.entry,
        exit=args.exit,
        tc_bps=args.tc_bps,
    )

    print("\n=== Params ===")
    print(
        f"y={args.y} x={args.x} start={args.start} reg_window={args.reg_window} z_window={args.z_window} "
        f"entry={args.entry} exit={args.exit} tc_bps={args.tc_bps}"
    )

    print("\n=== Results ===")
    for k in ["n_days", "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "trades"]:
        v = res.stats.get(k)
        if isinstance(v, float):
            print(f"{k:>14}: {v: .4f}")
        else:
            print(f"{k:>14}: {v}")

    plot_result(res, args.y, args.x)
    out_csv = os.path.join(res.out_dir, f"day016_pairs_{args.y}_{args.x}.csv")
    res.df.to_csv(out_csv)

    print(f"\nSaved: {res.out_dir}/")


if __name__ == "__main__":
    main()
