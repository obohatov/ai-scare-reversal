"""Turn the panel into a table of sharp falls with features and a target.

Run from the repository root:

    python -m scripts.transform

Every feature uses data available at the close of the event day. Trades enter
at the next session's open. Test-period outcomes stay out of the printed
summary, so they cannot influence the hand labels or model choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.data_repo import DATA_DIR, PANEL_PATH, RAW_DIR
from scripts.settings import REPORTS_DIR
from scripts.universe import AI_EVENT_EXTRAS, BENCHMARKS, EUROPEAN_PEERS

EVENTS_PATH = DATA_DIR / "processed" / "events.parquet"
FEATURE_SETS_PATH = REPORTS_DIR / "feature_sets.json"
SUMMARY_PATH = REPORTS_DIR / "features_summary.json"

HORIZON = 30
Z_THRESHOLD = -2.5
MIN_RELATIVE_FALL = -0.03
COOLDOWN_SESSIONS = 5
MIN_SECTOR_PEERS = 5
TRAIN_END = "2022-12-31"
VALIDATION_END = "2024-12-31"
EMBARGO_DAYS = 45  # longer than the 30-session label, in calendar days
FREEZE_DATE = "2024-12-31"

TO_PREDICT = ["is_positive_rel_30d", "rel_growth_30d", "growth_30d"]
DUMMIES = ["sector", "month", "weekday", "drop_size_bin", "vix_regime"]
NUMERIC = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d",
    "volatility_21d", "volatility_63d",
    "dist_sma_50", "dist_sma_200", "dist_52w_high", "dist_52w_low",
    "rsi_14", "atr_14", "gap_open", "close_location",
    "volume_ratio_20", "log_turnover_20",
    "rel_ret_1d", "rel_ret_5d_before", "rel_z", "rel_volatility_60",
    "sector_ret_1d", "sector_share_down", "sector_share_sharp",
    "market_ret_1d", "market_share_down", "market_volatility_21",
    "events_past_252", "sessions_since_event",
    "has_8k_2d", "has_earnings_2d",
    "pageviews_ratio_60",
    "dgs10", "dgs2", "vixcls", "dff",
    "yield_curve", "vix_change_5d",
]


def last_complete_session(dates: pd.Series, today: date) -> pd.Timestamp:
    """Drop today's row: its prices may come from an unfinished session."""
    latest = pd.Timestamp(dates.max())
    if latest.date() < today:
        return latest
    return pd.Timestamp(dates[dates < pd.Timestamp(today)].max())


def session_cutoff_date(now: datetime | None = None) -> date:
    """Return the first date treated as unfinished.

    US markets are closed by 22:00 UTC, so a later run keeps that day's rows.
    """
    moment = now or datetime.now(timezone.utc)
    return (moment + timedelta(hours=2)).date()


def select_rows(
    panel: pd.DataFrame, allowed: set[str], today: date
) -> pd.DataFrame:
    """Keep complete sessions and stocks; benchmarks are not tradable names."""
    cutoff = last_complete_session(panel["date"], today)
    rows = panel[
        (panel["date"] <= cutoff)
        & panel["ticker"].isin(allowed)
        & ~panel["ticker"].isin(BENCHMARKS)
        & (panel["adj_close"] > 0)
    ]
    rows = rows.sort_values(["ticker", "date"]).reset_index(drop=True)
    rows["sector"] = rows["sector"].astype(str)
    return rows


def rolling(frame: pd.DataFrame, column: str, window: int, how: str) -> pd.Series:
    """Per-ticker rolling statistic, with a minimum of two thirds of the window."""
    minimum = max(2, window * 2 // 3)
    return frame.groupby("ticker", sort=False)[column].transform(
        lambda series: getattr(series.rolling(window, min_periods=minimum), how)()
    )


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add features built from each stock's own prices and volume."""
    df = frame
    by_ticker = df.groupby("ticker", sort=False)
    df["ret_1d"] = by_ticker["adj_close"].pct_change(fill_method=None)
    for days in (5, 21, 63, 252):
        df[f"ret_{days}d"] = by_ticker["adj_close"].pct_change(days, fill_method=None)
    df["volatility_21d"] = rolling(df, "ret_1d", 21, "std")
    df["volatility_63d"] = rolling(df, "ret_1d", 63, "std")
    df["dist_sma_50"] = df["adj_close"] / rolling(df, "adj_close", 50, "mean") - 1
    df["dist_sma_200"] = df["adj_close"] / rolling(df, "adj_close", 200, "mean") - 1
    df["dist_52w_high"] = df["adj_close"] / rolling(df, "adj_close", 252, "max") - 1
    df["dist_52w_low"] = df["adj_close"] / rolling(df, "adj_close", 252, "min") - 1

    change = by_ticker["adj_close"].diff()
    df["_gain"] = change.clip(lower=0)
    df["_loss"] = -change.clip(upper=0)
    gain = rolling(df, "_gain", 14, "mean")
    loss = rolling(df, "_loss", 14, "mean")
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    previous_close = by_ticker["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["_range"] = true_range / df["close"]
    df["atr_14"] = rolling(df, "_range", 14, "mean")
    df["gap_open"] = df["open"] / previous_close - 1
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / spread

    average_volume = by_ticker["volume"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=10).mean()
    )
    df["volume_ratio_20"] = df["volume"] / average_volume.replace(0, np.nan)
    # London prices are in pence. Other currencies stay unconverted: rough size only.
    pence = df["ticker"].str.endswith(".L")
    df["_turnover"] = df["close"] * df["volume"] * np.where(pence, 0.01, 1.0)
    df["log_turnover_20"] = np.log1p(rolling(df, "_turnover", 20, "mean"))
    return df.drop(columns=["_gain", "_loss", "_range", "_turnover"])


def add_sector_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure each stock against its sector without counting itself."""
    df = frame
    known = df["sector"].ne("Unknown")
    groups = df[known].groupby(["date", "sector"])["ret_1d"]
    total = groups.transform("sum").reindex(df.index)
    count = groups.transform("count").reindex(df.index)
    peers = count - df["ret_1d"].notna().astype(int)
    df["sector_ret_1d"] = (total - df["ret_1d"].fillna(0)) / peers.where(
        peers >= MIN_SECTOR_PEERS
    )
    df["rel_ret_1d"] = df["ret_1d"] - df["sector_ret_1d"]

    by_ticker = df.groupby("ticker", sort=False)
    df["rel_volatility_60"] = by_ticker["rel_ret_1d"].transform(
        lambda series: series.shift(1).rolling(60, min_periods=40).std()
    )
    df["rel_z"] = df["rel_ret_1d"] / df["rel_volatility_60"]
    rel_5d = by_ticker["rel_ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    df["rel_ret_5d_before"] = rel_5d.groupby(df["ticker"], sort=False).shift(1)

    df["_down"] = (df["ret_1d"] < -0.02).astype(float).where(df["ret_1d"].notna())
    df["_sharp"] = (df["rel_z"] <= Z_THRESHOLD).astype(float).where(df["rel_z"].notna())
    df["sector_share_down"] = df.groupby(["date", "sector"])["_down"].transform("mean")
    df["sector_share_sharp"] = df.groupby(["date", "sector"])["_sharp"].transform("mean")
    df["market_ret_1d"] = df.groupby("date")["ret_1d"].transform("mean")
    df["market_share_down"] = df.groupby("date")["_down"].transform("mean")

    daily = df.groupby("date")["ret_1d"].mean().sort_index()
    volatility = daily.rolling(21, min_periods=14).std()
    df["market_volatility_21"] = df["date"].map(volatility)
    return df.drop(columns=["_down", "_sharp"])


def add_context_features(frame: pd.DataFrame, covered: set[str]) -> pd.DataFrame:
    """Add news, attention, macro and calendar features."""
    df = frame
    by_ticker = df.groupby("ticker", sort=False)
    for source, target in (("has_8k", "has_8k_2d"), ("has_earnings_8k", "has_earnings_2d")):
        recent = by_ticker[source].transform(
            lambda series: series.astype(float).rolling(2, min_periods=1).max()
        )
        # A missing EDGAR history means "unknown", not "no news".
        df[target] = recent.where(df["ticker"].isin(covered))

    # Page views can arrive a day late, so only the previous day is safe.
    views = by_ticker["pageviews"].shift(1)
    typical = views.groupby(df["ticker"], sort=False).transform(
        lambda series: series.shift(1).rolling(60, min_periods=30).median()
    )
    df["pageviews_ratio_60"] = views / typical.replace(0, np.nan)

    df["yield_curve"] = df["dgs10"] - df["dgs2"]
    df["vix_change_5d"] = by_ticker["vixcls"].diff(5)
    df["month"] = df["date"].dt.month.astype(str)
    df["weekday"] = df["date"].dt.day_of_week.astype(str)
    df["drop_size_bin"] = pd.cut(
        df["rel_ret_1d"],
        [-np.inf, -0.15, -0.10, -0.06, MIN_RELATIVE_FALL, np.inf],
        labels=["over_15", "10_to_15", "6_to_10", "3_to_6", "under_3"],
    ).astype(str)
    df["vix_regime"] = pd.cut(
        df["vixcls"], [-np.inf, 15, 20, 30, np.inf], labels=["calm", "normal", "high", "panic"]
    ).astype(str)
    return df


def add_target(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Compare next-open-to-close growth with close-to-close sector growth.

    The spans differ by the overnight move before entry, because sector returns
    exist only from close to close. `align_target` gives the matched version.
    """
    df = frame
    by_ticker = df.groupby("ticker", sort=False)
    adjusted_open = df["open"] * df["adj_close"] / df["close"].replace(0, np.nan)
    df["entry_open"] = adjusted_open.groupby(df["ticker"], sort=False).shift(-1)
    df["exit_close"] = by_ticker["adj_close"].shift(-(horizon + 1))
    df["exit_date"] = by_ticker["date"].shift(-(horizon + 1))
    df["growth_30d"] = df["exit_close"] / df["entry_open"]

    sector_log = np.log1p(df["sector_ret_1d"].fillna(0))
    cumulative = sector_log.groupby(df["ticker"], sort=False).cumsum()
    future = cumulative.groupby(df["ticker"], sort=False).shift(-(horizon + 1))
    df["sector_growth_30d"] = np.exp(future - cumulative)
    df["rel_growth_30d"] = df["growth_30d"] / df["sector_growth_30d"]
    positive = (df["rel_growth_30d"] > 1).astype(float)
    df["is_positive_rel_30d"] = positive.where(df["rel_growth_30d"].notna())
    return df


def align_target(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Measure stock and sector over one close-to-close span.

    This replaces next-open entry, so it describes the signal and not a tradable
    result. It runs after `add_target`, whose sector growth it reuses.
    """
    df = frame
    future = df.groupby("ticker", sort=False)["adj_close"].shift(-(horizon + 1))
    df["growth_30d"] = future / df["adj_close"]
    df["rel_growth_30d"] = df["growth_30d"] / df["sector_growth_30d"]
    positive = (df["rel_growth_30d"] > 1).astype(float)
    df["is_positive_rel_30d"] = positive.where(df["rel_growth_30d"].notna())
    return df


def find_events(frame: pd.DataFrame, cooldown: int = COOLDOWN_SESSIONS) -> pd.Series:
    """Flag sharp relative falls; repeats within the cooldown are not new events."""
    raw = (frame["rel_z"] <= Z_THRESHOLD) & (frame["rel_ret_1d"] <= MIN_RELATIVE_FALL)
    session = frame.groupby("ticker", sort=False).cumcount()
    kept = pd.Series(False, index=frame.index)
    last_seen: dict[str, int] = {}
    for index in frame.index[raw]:
        ticker, number = frame.at[index, "ticker"], session.at[index]
        if ticker not in last_seen or number - last_seen[ticker] >= cooldown:
            kept.at[index] = True
            last_seen[ticker] = number
    return kept


def add_event_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Count earlier events for each stock, excluding the current day."""
    df = frame
    df["is_event"] = find_events(df)
    event = df["is_event"].astype(float)
    earlier = event.groupby(df["ticker"], sort=False).shift(1)
    df["events_past_252"] = earlier.groupby(df["ticker"], sort=False).transform(
        lambda series: series.rolling(252, min_periods=1).sum()
    )
    session = df.groupby("ticker", sort=False).cumcount().astype(float)
    last_event = session.where(df["is_event"])
    last_event = last_event.groupby(df["ticker"], sort=False).shift(1)
    last_event = last_event.groupby(df["ticker"], sort=False).ffill()
    df["sessions_since_event"] = session - last_event
    return df


def assign_split(dates: pd.Series) -> pd.Series:
    """Date split with a gap, so training labels never reach a later period."""
    train_end = pd.Timestamp(TRAIN_END)
    validation_end = pd.Timestamp(VALIDATION_END)
    gap = pd.Timedelta(days=EMBARGO_DAYS)
    split = pd.Series("embargo", index=dates.index, dtype="object")
    split[dates <= train_end - gap] = "train"
    split[(dates > train_end) & (dates <= validation_end - gap)] = "validation"
    split[dates > validation_end] = "test"
    return split


def universe_groups(tickers: pd.Series, frozen: set[str]) -> pd.DataFrame:
    """Label why a stock is here. Event extras were chosen by outcome."""
    return pd.DataFrame(
        {
            "is_peer": tickers.isin(EUROPEAN_PEERS),
            "is_event_extra": tickers.isin(AI_EVENT_EXTRAS),
            "in_frozen_universe": tickers.isin(frozen),
        },
        index=tickers.index,
    )


def build_daily(
    panel: pd.DataFrame, allowed: set[str], covered: set[str], today: date
) -> pd.DataFrame:
    """Run every step and return one row per stock and session."""
    df = select_rows(panel, allowed, today)
    df = add_price_features(df)
    df = add_sector_features(df)
    df = add_context_features(df, covered)
    df = add_target(df)
    return add_event_history(df)


def daily_baseline(daily: pd.DataFrame) -> dict:
    """How often an index member beats its sector on an ordinary day.

    Results for the test period are left out on purpose.
    """
    rows = daily[daily["in_sp500"] & daily["is_positive_rel_30d"].notna()]
    splits = assign_split(rows["date"])
    baseline = {}
    for split in ("train", "validation"):
        ordinary = rows[(splits == split) & ~rows["is_event"]]
        baseline[split] = {
            "days": len(ordinary),
            "beat_sector": round(float(ordinary["is_positive_rel_30d"].mean()), 4),
            "median_rel_growth": round(float(ordinary["rel_growth_30d"].median()), 4),
        }
    return baseline


def events_from_daily(daily: pd.DataFrame, frozen: set[str]) -> pd.DataFrame:
    """Keep the event rows and the columns later stages need."""
    df = daily
    events = df[df["is_event"]].copy()
    events["split"] = assign_split(events["date"])
    events = events.join(universe_groups(events["ticker"], frozen))
    tradable = events["in_sp500"] | events["is_peer"] | events["is_event_extra"]
    columns = [
        "date", "ticker", "split", "exit_date", "entry_open", "exit_close",
        "sector_growth_30d", "in_sp500", "in_frozen_universe", "is_peer",
        "is_event_extra", *TO_PREDICT, *NUMERIC, *DUMMIES,
    ]
    return events.loc[tradable, columns].reset_index(drop=True)


def build_events(
    panel: pd.DataFrame,
    allowed: set[str],
    frozen: set[str],
    covered: set[str],
    today: date,
) -> pd.DataFrame:
    """Run every step and return one row per event."""
    return events_from_daily(build_daily(panel, allowed, covered, today), frozen)


def summarise(events: pd.DataFrame) -> dict:
    """Report counts everywhere, but outcomes only outside the test period."""
    summary: dict = {"events": len(events), "tickers": int(events["ticker"].nunique())}
    for split, rows in events.groupby("split"):
        labelled = rows["is_positive_rel_30d"].notna()
        item = {
            "events": len(rows),
            "first": str(rows["date"].min().date()),
            "last": str(rows["date"].max().date()),
            "labelled": int(labelled.sum()),
        }
        if split != "test":
            item["base_rate"] = round(float(rows.loc[labelled, "is_positive_rel_30d"].mean()), 4)
        summary[split] = item
    test = events[events["split"] == "test"]
    summary["test_frozen_universe_events"] = int(test["in_frozen_universe"].sum())
    summary["has_8k_known_share"] = round(float(events["has_8k_2d"].notna().mean()), 3)
    summary["pageviews_known_share"] = round(
        float(events["pageviews_ratio_60"].notna().mean()), 3
    )
    summary["numeric_features"] = len(NUMERIC)
    return summary


def frozen_members(membership: pd.DataFrame) -> set[str]:
    """Return the index members at the freeze date."""
    known = membership[membership["date"] <= FREEZE_DATE]
    return set(known.loc[known["date"] == known["date"].max(), "ticker"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    panel = pd.read_parquet(PANEL_PATH)
    membership = pd.read_parquet(RAW_DIR / "membership.parquet")
    allowed = set(membership["ticker"]) | set(EUROPEAN_PEERS) | set(AI_EVENT_EXTRAS)
    covered = {path.stem for path in (RAW_DIR / "edgar").glob("*.parquet")}
    daily = build_daily(panel, allowed, covered, session_cutoff_date())
    events = events_from_daily(daily, frozen_members(membership))

    events.to_parquet(EVENTS_PATH)
    feature_sets = {"TO_PREDICT": TO_PREDICT, "NUMERIC": NUMERIC, "DUMMIES": DUMMIES}
    FEATURE_SETS_PATH.write_text(json.dumps(feature_sets, indent=2) + "\n", encoding="utf-8")
    summary = summarise(events)
    summary["ordinary_days"] = daily_baseline(daily)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
