import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO"


@dataclass
class Summary:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float


def monthly_returns_from_adjclose(adj: pd.Series) -> pd.Series:
    px_m = adj.resample("M").last()
    rets = px_m.pct_change().dropna()
    rets.name = "ret"
    return rets


def perf_stats(monthly_rets: pd.Series) -> Summary:
    # monthly → annualized
    mu = monthly_rets.mean() * 12.0
    vol = monthly_rets.std(ddof=0) * np.sqrt(12.0)
    sharpe = np.nan if vol == 0 else mu / vol

    eq = (1.0 + monthly_rets).cumprod()
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = dd.min()

    years = (monthly_rets.index[-1] - monthly_rets.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan

    return Summary(cagr=cagr, vol=vol, sharpe=sharpe, max_dd=max_dd)


def fetch_indpro(start: str) -> pd.Series:
    df = pd.read_csv(FRED_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    s = df["INDPRO"].replace(".", np.nan).astype(float).dropna()
    s = s.loc[pd.to_datetime(start) :]
    s.name = "INDPRO"
    return s


def zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win).mean()
    s = x.rolling(win).std(ddof=0)
    return (x - m) / s


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--zwin", type=int, default=60, help="z-score window in months")
    ap.add_argument("--outdir", default="day-032-industrial-production-regime/out")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)

    # Prices
    px = yf.download(args.ticker, start=args.start, auto_adjust=True, progress=False)
    if px.empty:
        raise SystemExit(f"No price data returned for {args.ticker}")
    adj = px["Close"].rename("adj_close")
    rets_spy = monthly_returns_from_adjclose(adj)

    # Macro (monthly)
    indpro = fetch_indpro(args.start)
    indpro_yoy = indpro.pct_change(12).rename("indpro_yoy").dropna()
    indpro_z = zscore(indpro_yoy, win=args.zwin).rename("indpro_yoy_z")

    macro = pd.concat([indpro_yoy, indpro_z], axis=1).dropna()

    # Align to month-end returns (avoid lookahead: use macro value from same month to decide next month exposure)
    macro_me = macro.resample("M").last().reindex(rets_spy.index).dropna()

    signal = ((macro_me["indpro_yoy"] > 0) & (macro_me["indpro_yoy_z"] > 0)).astype(int)
    signal.name = "risk_on"

    # Hold SPY next month if signal at t is 1
    strat_rets = signal.shift(1).fillna(0) * rets_spy
    strat_rets.name = "strategy"

    bh_rets = rets_spy.rename("buy_hold")

    df = pd.concat([bh_rets, strat_rets, signal], axis=1).dropna()

    s_bh = perf_stats(df["buy_hold"])
    s_st = perf_stats(df["strategy"])

    print("=== Inputs ===")
    print(f"ticker: {args.ticker}")
    print(f"start : {args.start}")
    print(f"zwin  : {args.zwin} months")
    print()

    def fmt(s: Summary) -> str:
        return (
            f"CAGR={s.cagr:6.2%}  vol={s.vol:6.2%}  sharpe={s.sharpe:5.2f}  maxDD={s.max_dd:6.2%}"
        )

    print("=== Performance (monthly backtest) ===")
    print(f"Buy&Hold: {fmt(s_bh)}")
    print(f"Strategy: {fmt(s_st)}")
    print()
    print("% months risk-on:", f"{df['risk_on'].mean():.1%}")

    # Plot equity curves
    eq = (1 + df[["buy_hold", "strategy"]]).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["buy_hold"], label="Buy & Hold")
    plt.plot(eq.index, eq["strategy"], label="INDPRO regime")
    plt.title(f"Day 032 — INDPRO Regime Filter ({args.ticker})")
    plt.ylabel("Equity (growth of $1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "equity_curves.png"), dpi=150)
    plt.close()

    # Plot macro signal
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(macro_me.index, macro_me["indpro_yoy"], label="INDPRO YoY", color="tab:blue")
    ax1.axhline(0, color="k", lw=1, alpha=0.5)
    ax1.set_ylabel("YoY")

    ax2 = ax1.twinx()
    ax2.plot(macro_me.index, macro_me["indpro_yoy_z"], label="YoY z-score", color="tab:orange", alpha=0.8)
    ax2.set_ylabel("z-score")
    ax2.axhline(0, color="tab:orange", lw=1, alpha=0.3)

    plt.title("Industrial Production signal")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "macro_signal.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
