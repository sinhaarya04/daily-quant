"""Day 011 — Turn-of-the-Month Effect (SPY)

A small, runnable test of the classic turn-of-the-month (TOM) calendar anomaly.

Strategy:
- Hold the asset on the *last* trading day of each month
- Hold the asset on the first N trading days of each month (default: N=3)

Compare to buy-and-hold using simple daily returns from adjusted close.

Usage:
  python day-011-turn-of-month-effect/tom_effect.py --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252


@dataclass
class Stats:
    ann_return: float
    ann_vol: float
    sharpe: float
    win_rate_invested: float


def annualized_stats(daily_returns: pd.Series) -> Stats:
    r = daily_returns.dropna()
    if len(r) == 0:
        return Stats(np.nan, np.nan, np.nan, np.nan)

    mu = r.mean() * TRADING_DAYS_PER_YEAR
    vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = np.nan if vol == 0 else mu / vol

    invested = r != 0
    win_rate_invested = (r[invested] > 0).mean() if invested.any() else np.nan

    return Stats(mu, vol, sharpe, win_rate_invested)


def build_tom_signal(index: pd.DatetimeIndex, first_days: int) -> pd.Series:
    """Boolean Series indexed by dates: True if in TOM window."""
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")

    hold = pd.Series(False, index=index)

    # For each month: mark first N trading days of that month.
    for _, g in df.groupby("ym"):
        dates = g.index
        if len(dates) == 0:
            continue
        hold.loc[dates[:first_days]] = True

    # Mark the last trading day of each month.
    last_days = df.groupby("ym").apply(lambda g: g.index.max())
    hold.loc[pd.DatetimeIndex(last_days.values)] = True

    return hold


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns.fillna(0.0)).cumprod()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker (default: SPY)")
    p.add_argument("--start", default="1993-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--first-days", type=int, default=3, help="# first trading days of month to hold (default: 3)")
    p.add_argument("--plot", action="store_true", help="Save an equity curve plot to PNG")
    args = p.parse_args()

    if args.first_days < 0 or args.first_days > 10:
        raise SystemExit("--first-days should be between 0 and 10")

    px = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
    if px.empty:
        raise SystemExit("No data returned. Check ticker/date range.")

    close = px["Close"].copy()
    # yfinance returns a Series for single tickers in some versions, but a 1-col DataFrame in others.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("close")
    rets = close.pct_change().rename("ret")

    signal = build_tom_signal(rets.index, first_days=args.first_days)
    strat_rets = (rets * signal.astype(float)).rename("tom_ret")

    bh_stats = annualized_stats(rets)
    tom_stats = annualized_stats(strat_rets)

    print(f"Data: {args.ticker} | {rets.index.min().date()} → {rets.index.max().date()} | N={len(rets.dropna())} days")
    print()
    print("Annualized stats (simple, from daily returns):")
    print("- Buy & Hold")
    print(f"  return: {bh_stats.ann_return: .2%} | vol: {bh_stats.ann_vol: .2%} | Sharpe: {bh_stats.sharpe: .2f} | win-rate: {bh_stats.win_rate_invested: .2%}")
    print("- Turn-of-Month")
    print(f"  return: {tom_stats.ann_return: .2%} | vol: {tom_stats.ann_vol: .2%} | Sharpe: {tom_stats.sharpe: .2f} | win-rate (invested days): {tom_stats.win_rate_invested: .2%}")
    print()

    exposure = signal.mean()
    print(f"Average exposure (fraction of days invested): {exposure: .2%}")

    eq_bh = equity_curve(rets).rename("buy_hold")
    eq_tom = equity_curve(strat_rets).rename("tom")

    if args.plot:
        import matplotlib.pyplot as plt

        out_path = f"day-011-turn-of-month-effect/equity_{args.ticker}_first{args.first_days}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(eq_bh.index, eq_bh.values, label="Buy & Hold")
        plt.plot(eq_tom.index, eq_tom.values, label=f"TOM (last + first {args.first_days})")
        plt.title(f"Turn-of-the-Month effect — {args.ticker}")
        plt.ylabel("Equity (start = 1.0)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
