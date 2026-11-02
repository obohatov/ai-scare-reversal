"""Simulate strategies built on sharp falls against holding the S&P 500.

Run from the repository root:

    python -m scripts.simulate

The default run uses the validation period. Strategies must be fixed from that
run before the test period is opened once with:

    python -m scripts.simulate --period test --confirm-frozen

Between trades capital stays in an S&P 500 ETF, so a strategy only has to add
value on top of the index. Stock positions enter at the next session's open.
The ETF sleeve moves close to close, which is a small approximation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.data_repo import DATA_DIR, PANEL_PATH
from scripts.train import PREDICTIONS_PATH
from scripts.transform import EVENTS_PATH, HORIZON

BENCHMARK = "SPY"
START_CAPITAL = 100_000.0
STOCK_FEE = 0.001  # per side, as in the course
ETF_FEE = 0.0002  # per side
TRADING_DAYS = 252
TOP_SCORE_QUANTILE = 0.90
TEST_END = "2026-10-30"  # later rows may come from an unfinished session
SHORT_BORROW_RATE = 0.01  # a year, on the value of open shorts
SPREAD_SIDE_POSITIONS = 10


@dataclass(frozen=True)
class Strategy:
    name: str
    rule: str  # all_drops, no_news or top_score
    max_positions: int = 10
    stop_loss: float | None = None
    position_share: float | None = None  # of equity; default is 1 / max_positions


STRATEGIES = (
    Strategy("all_drops", "all_drops"),
    Strategy("no_news", "no_news"),
    Strategy("no_news_5_positions", "no_news", max_positions=5),
    Strategy("no_news_stop_15", "no_news", stop_loss=0.15),
    Strategy("no_news_core_20", "no_news", position_share=0.02),
    Strategy("top_score", "top_score"),
)


def select_signals(
    events: pd.DataFrame, strategy: Strategy, threshold: float
) -> pd.DataFrame:
    """Apply a rule and rank each day's signals, best first."""
    if strategy.rule == "all_drops":
        chosen = events
    elif strategy.rule == "no_news":
        chosen = events[(events["has_8k_2d"] == 0) & (events["has_earnings_2d"] == 0)]
    elif strategy.rule == "top_score":
        chosen = events[events["score"] >= threshold]
    else:
        raise ValueError(f"Unknown rule: {strategy.rule}")
    if strategy.rule == "top_score":
        return chosen.sort_values(["date", "score"], ascending=[True, False])
    return chosen.sort_values(["date", "rel_z", "ticker"])


def vector_results(signals: pd.DataFrame, benchmark: pd.Series) -> dict[str, Any]:
    """Equal-sized trades with no capital limit: quick, but overstates capacity."""
    rows = signals[signals["growth_30d"].notna()]
    if rows.empty:
        return {"trades": 0}
    start = benchmark.reindex(rows["date"]).to_numpy()
    end = benchmark.reindex(rows["exit_date"]).to_numpy()
    net = rows["growth_30d"].to_numpy() - 1 - 2 * STOCK_FEE
    excess = net - (end / start - 1)
    return {
        "trades": len(rows),
        "mean_net_return": round(float(np.nanmean(net)), 4),
        "median_net_return": round(float(np.nanmedian(net)), 4),
        "win_rate": round(float(np.nanmean(net > 0)), 4),
        "mean_excess_vs_index": round(float(np.nanmean(excess)), 4),
        "share_beating_index": round(float(np.nanmean(excess > 0)), 4),
    }


def exact_equity(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    strategy: Strategy,
    start_capital: float = START_CAPITAL,
    benchmark: str = BENCHMARK,
    horizon: int = HORIZON,
) -> tuple[pd.Series, dict[str, Any]]:
    """Walk day by day with a position limit and reinvested gains."""
    calendar = prices.index
    position_of = {day: number for number, day in enumerate(calendar)}
    index_prices = prices[benchmark].ffill()
    by_entry: dict[int, list] = {}
    for row in signals.itertuples():
        number = position_of.get(row.date)
        if number is None or number + 1 >= len(calendar) or pd.isna(row.entry_open):
            continue
        by_entry.setdefault(number + 1, []).append(row)

    index_value = start_capital * (1 - ETF_FEE)
    open_positions: list[dict[str, Any]] = []
    equity = []
    trades = invested_days = stopped = 0
    previous_equity = start_capital
    for number, day in enumerate(calendar):
        if number > 0:
            index_value *= index_prices.iloc[number] / index_prices.iloc[number - 1]
        for row in by_entry.get(number, []):
            held = {item["ticker"] for item in open_positions}
            if len(open_positions) >= strategy.max_positions or row.ticker in held:
                continue
            share = strategy.position_share or 1 / strategy.max_positions
            size = min(previous_equity * share, index_value)
            if size <= 0:
                continue
            index_value -= size
            open_positions.append(
                {
                    "ticker": row.ticker,
                    "units": size * (1 - STOCK_FEE - ETF_FEE) / row.entry_open,
                    "entry": row.entry_open,
                    "last": row.entry_open,
                    "exit_number": number + horizon,
                }
            )
            trades += 1

        remaining = []
        for item in open_positions:
            price = prices.at[day, item["ticker"]]
            if not pd.isna(price):
                item["last"] = float(price)
            hit_stop = (
                strategy.stop_loss is not None
                and item["last"] <= item["entry"] * (1 - strategy.stop_loss)
            )
            if hit_stop or number >= item["exit_number"]:
                index_value += item["units"] * item["last"] * (1 - STOCK_FEE - ETF_FEE)
                stopped += int(hit_stop)
            else:
                remaining.append(item)
        open_positions = remaining
        invested_days += len(open_positions)
        previous_equity = index_value + sum(
            item["units"] * item["last"] for item in open_positions
        )
        equity.append(previous_equity)

    days = max(len(calendar), 1)
    stats = {
        "trades": trades,
        "stopped_out": stopped,
        "average_positions": round(invested_days / days, 2),
    }
    return pd.Series(equity, index=calendar, name=strategy.name), stats


def spread_signals(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long falls without company news; short falls after earnings."""
    ordered = events.sort_values(["date", "rel_z", "ticker"])
    longs = ordered[(ordered["has_8k_2d"] == 0) & (ordered["has_earnings_2d"] == 0)]
    shorts = ordered[ordered["has_earnings_2d"] == 1]
    return longs, shorts


def spread_gap(longs: pd.DataFrame, shorts: pd.DataFrame) -> dict[str, Any]:
    """Compare sector-relative outcomes of the two groups, trade by trade."""
    long_rel = longs["rel_growth_30d"].dropna()
    short_rel = shorts["rel_growth_30d"].dropna()
    if long_rel.empty or short_rel.empty:
        return {"long_events": len(long_rel), "short_events": len(short_rel)}
    gap = float(long_rel.mean() - short_rel.mean())
    error = float(
        np.sqrt(long_rel.var() / len(long_rel) + short_rel.var() / len(short_rel))
    )
    return {
        "long_events": len(long_rel),
        "short_events": len(short_rel),
        "long_mean_rel_growth": round(float(long_rel.mean()), 4),
        "short_mean_rel_growth": round(float(short_rel.mean()), 4),
        "mean_gap": round(gap, 4),
        "std_error": round(error, 4),
    }


def spread_equity(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    prices: pd.DataFrame,
    rates: pd.Series,
    side_positions: int = SPREAD_SIDE_POSITIONS,
    start_capital: float = START_CAPITAL,
    horizon: int = HORIZON,
) -> tuple[pd.Series, dict[str, Any]]:
    """Run a market-neutral book; idle collateral earns the overnight rate.

    Each position takes half of 1 / side_positions of equity, so full books hold
    100% long and 100% short. `rates` is in percent a year.
    """
    calendar = prices.index
    position_of = {day: number for number, day in enumerate(calendar)}
    entries: dict[int, list[tuple[int, Any]]] = {}
    for sign, frame in ((1, longs), (-1, shorts)):
        for row in frame.itertuples():
            number = position_of.get(row.date)
            if number is None or number + 1 >= len(calendar) or pd.isna(row.entry_open):
                continue
            entries.setdefault(number + 1, []).append((sign, row))

    cash = start_capital
    book: list[dict[str, Any]] = []
    equity_curve = []
    trades = {1: 0, -1: 0}
    previous_equity = start_capital
    for number, day in enumerate(calendar):
        short_value = sum(item["units"] * item["last"] for item in book if item["sign"] < 0)
        rate = rates.get(day, 0.0)
        rate = 0.0 if pd.isna(rate) else float(rate)
        cash += max(cash - short_value, 0.0) * rate / 100 / TRADING_DAYS
        cash -= short_value * SHORT_BORROW_RATE / TRADING_DAYS

        for sign, row in entries.get(number, []):
            same_side = [item for item in book if item["sign"] == sign]
            held = {item["ticker"] for item in book}
            if len(same_side) >= side_positions or row.ticker in held:
                continue
            size = previous_equity / (2 * side_positions)
            units = size / row.entry_open
            cash -= sign * size + size * STOCK_FEE
            book.append(
                {
                    "ticker": row.ticker,
                    "sign": sign,
                    "units": units,
                    "last": row.entry_open,
                    "exit_number": number + horizon,
                }
            )
            trades[sign] += 1

        remaining = []
        for item in book:
            price = prices.at[day, item["ticker"]]
            if not pd.isna(price):
                item["last"] = float(price)
            if number >= item["exit_number"]:
                value = item["units"] * item["last"]
                cash += item["sign"] * value - value * STOCK_FEE
            else:
                remaining.append(item)
        book = remaining
        previous_equity = cash + sum(
            item["sign"] * item["units"] * item["last"] for item in book
        )
        equity_curve.append(previous_equity)
    series = pd.Series(equity_curve, index=calendar, name="news_spread")
    return series, {"long_trades": trades[1], "short_trades": trades[-1]}


def equal_weight_curve(
    panel: pd.DataFrame, calendar: pd.DatetimeIndex, start_capital: float = START_CAPITAL
) -> pd.Series:
    """Average daily return of index members on each day, compounded."""
    rows = panel.sort_values(["ticker", "date"]).copy()
    rows["ret"] = rows.groupby("ticker", sort=False)["adj_close"].pct_change(fill_method=None)
    daily = rows[rows["in_sp500"]].groupby("date")["ret"].mean()
    daily = daily.reindex(calendar).fillna(0.0)
    daily.iloc[0] = 0.0
    return start_capital * (1 + daily).cumprod()


def cash_curve(
    rates: pd.Series, calendar: pd.DatetimeIndex, start_capital: float = START_CAPITAL
) -> pd.Series:
    """Grow capital at the overnight rate, in percent a year."""
    daily = rates.reindex(calendar).ffill().fillna(0.0) / 100 / TRADING_DAYS
    daily.iloc[0] = 0.0
    return start_capital * (1 + daily).cumprod()


def performance(equity: pd.Series) -> dict[str, float]:
    """Annualised return and risk from a daily equity curve."""
    returns = equity.pct_change().dropna()
    years = len(returns) / TRADING_DAYS
    total = equity.iloc[-1] / equity.iloc[0]
    volatility = float(returns.std() * np.sqrt(TRADING_DAYS))
    mean = float(returns.mean() * TRADING_DAYS)
    return {
        "cagr": round(float(total ** (1 / years) - 1), 4),
        "volatility": round(volatility, 4),
        "sharpe": round(mean / volatility, 3) if volatility else 0.0,
        "max_drawdown": round(float((equity / equity.cummax() - 1).min()), 4),
        "final_equity": round(float(equity.iloc[-1]), 0),
    }


def load_events(period: str) -> tuple[pd.DataFrame, float]:
    """Load the period's universe and the score cut-off fixed on validation."""
    events = pd.read_parquet(EVENTS_PATH)
    scores = pd.read_parquet(PREDICTIONS_PATH, columns=["date", "ticker", "score"])
    events = events.merge(scores, on=["date", "ticker"], how="inner")
    validation = events[(events["split"] == "validation") & events["in_sp500"]]
    threshold = float(validation["score"].quantile(TOP_SCORE_QUANTILE))
    if period == "validation":
        return validation.reset_index(drop=True), threshold
    test = events[(events["split"] == "test") & events["in_frozen_universe"]]
    return test.reset_index(drop=True), threshold


def load_market(events: pd.DataFrame) -> pd.DataFrame:
    """Read the panel columns the simulation needs, from the first signal on."""
    columns = ["date", "ticker", "adj_close", "in_sp500", "dff"]
    panel = pd.read_parquet(PANEL_PATH, columns=columns)
    return panel[panel["date"] >= events["date"].min()]


def load_prices(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted closes up to the last possible exit."""
    tickers = set(events["ticker"]) | {BENCHMARK}
    rows = panel[panel["ticker"].isin(tickers)]
    wide = rows.pivot(index="date", columns="ticker", values="adj_close")
    wide = wide[wide[BENCHMARK].notna() & (wide.index <= TEST_END)].astype(float)
    last_exit = events["exit_date"].max()
    if pd.notna(last_exit) and events["exit_date"].notna().all():
        wide = wide[wide.index <= last_exit]
    return wide


def run(period: str) -> dict[str, Any]:
    """Compare every strategy with the index over one period."""
    events, threshold = load_events(period)
    panel = load_market(events)
    prices = load_prices(events, panel)
    rates = panel.groupby("date")["dff"].first().astype(float)
    index_equity = START_CAPITAL * prices[BENCHMARK] / prices[BENCHMARK].iloc[0]
    report: dict[str, Any] = {
        "period": period,
        "first": str(prices.index[0].date()),
        "last": str(prices.index[-1].date()),
        "events": len(events),
        "score_threshold": round(threshold, 4),
        "index": performance(index_equity),
        "equal_weight_index": performance(equal_weight_curve(panel, prices.index)),
        "cash": performance(cash_curve(rates, prices.index)),
        "strategies": {},
    }
    longs, shorts = spread_signals(events)
    spread, spread_stats = spread_equity(longs, shorts, prices, rates)
    report["news_spread"] = {
        "gap": spread_gap(longs, shorts),
        "exact": {**spread_stats, **performance(spread)},
    }
    print("news_spread", report["news_spread"], flush=True)
    for strategy in STRATEGIES:
        signals = select_signals(events, strategy, threshold)
        equity, stats = exact_equity(signals, prices, strategy)
        report["strategies"][strategy.name] = {
            "config": asdict(strategy),
            "signals": len(signals),
            "vector": vector_results(signals, prices[BENCHMARK]),
            "exact": {**stats, **performance(equity)},
        }
        print(strategy.name, report["strategies"][strategy.name]["exact"], flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-frozen", action="store_true")
    args = parser.parse_args(argv)
    if args.period == "test" and not args.confirm_frozen:
        parser.error("The test period needs --confirm-frozen: fix the strategies first.")
    report = run(args.period)
    path = DATA_DIR.parent / "reports" / f"simulation_{args.period}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("first", "last", "events", "index", "equal_weight_index", "cash")}, indent=2))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
