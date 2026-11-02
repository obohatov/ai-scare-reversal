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


@dataclass(frozen=True)
class Strategy:
    name: str
    rule: str  # all_drops, no_news or top_score
    max_positions: int = 10
    stop_loss: float | None = None


STRATEGIES = (
    Strategy("all_drops", "all_drops"),
    Strategy("no_news", "no_news"),
    Strategy("no_news_5_positions", "no_news", max_positions=5),
    Strategy("no_news_stop_15", "no_news", stop_loss=0.15),
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
            size = min(previous_equity / strategy.max_positions, index_value)
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


def load_prices(events: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted closes from the first signal to the last possible exit."""
    panel = pd.read_parquet(PANEL_PATH, columns=["date", "ticker", "adj_close"])
    tickers = set(events["ticker"]) | {BENCHMARK}
    rows = panel[panel["ticker"].isin(tickers) & (panel["date"] >= events["date"].min())]
    wide = rows.pivot(index="date", columns="ticker", values="adj_close")
    wide = wide[wide[BENCHMARK].notna()].astype(float)
    last_exit = events["exit_date"].max()
    if pd.notna(last_exit) and events["exit_date"].notna().all():
        wide = wide[wide.index <= last_exit]
    return wide


def run(period: str) -> dict[str, Any]:
    """Compare every strategy with the index over one period."""
    events, threshold = load_events(period)
    prices = load_prices(events)
    index_equity = START_CAPITAL * prices[BENCHMARK] / prices[BENCHMARK].iloc[0]
    report: dict[str, Any] = {
        "period": period,
        "first": str(prices.index[0].date()),
        "last": str(prices.index[-1].date()),
        "events": len(events),
        "score_threshold": round(threshold, 4),
        "index": performance(index_equity),
        "strategies": {},
    }
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
    print(json.dumps({key: report[key] for key in ("first", "last", "events", "index")}, indent=2))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
