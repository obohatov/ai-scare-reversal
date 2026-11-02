"""Measure what happened after the hand-labelled AI events.

Run from the repository root, after the test protocol is committed:

    python -m scripts.ai_slice

Each event counts once, as the mean of its stocks. With few overlapping events
chosen because stocks fell, the output is descriptive rather than a proof.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.data_repo import PANEL_PATH, RAW_DIR
from scripts.settings import PROJECT_ROOT, REPORTS_DIR
from scripts.simulate import BENCHMARK, TEST_END
from scripts.transform import add_target, assign_split, build_daily
from scripts.universe import AI_EVENT_EXTRAS, EUROPEAN_PEERS

LABELS_PATH = PROJECT_ROOT / "events" / "ai_events.csv"
REPORT_PATH = REPORTS_DIR / "ai_slice.json"
HORIZONS = (30, 90)
LABEL_COLUMNS = ["substitutes_paid_workflow", "depends_on_proprietary_data", "product_status"]
TOLERANCE_DAYS = 5


def expand_events(labels: pd.DataFrame) -> pd.DataFrame:
    """Return one row per event and ticker."""
    rows = []
    for event in labels.to_dict("records"):
        for source in ("named_tickers", "proposed_tickers"):
            text = event.get(source)
            tickers = str(text).split() if pd.notna(text) else []
            rows += [{**event, "ticker": ticker, "ticker_source": source} for ticker in tickers]
    frame = pd.DataFrame(rows)
    frame["event_date"] = pd.to_datetime(frame["event_date"]).astype("datetime64[ns]")
    return frame


def locate(expanded: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach each stock's first session on or after the event date."""
    columns = [
        "date", "ticker", "rel_ret_1d", "is_event", "exit_date",
        "growth_30d", "rel_growth_30d",
    ]
    return pd.merge_asof(
        expanded.sort_values("event_date"),
        daily[columns].sort_values("date").assign(
            date=lambda frame: frame["date"].astype("datetime64[ns]")
        ),
        left_on="event_date",
        right_on="date",
        by="ticker",
        direction="forward",
        tolerance=pd.Timedelta(days=TOLERANCE_DAYS),
    )


def add_index_excess(rows: pd.DataFrame, index_prices: pd.Series) -> pd.DataFrame:
    """Compare each stock with the index over the same dates."""
    start = index_prices.reindex(rows["date"]).to_numpy()
    end = index_prices.reindex(rows["exit_date"]).to_numpy()
    return rows.assign(excess_vs_index=rows["growth_30d"].to_numpy() - end / start)


def event_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce every event to one row, so that large events do not dominate."""
    table = rows.groupby("event_id", sort=True).agg(
        event_date=("event_date", "first"),
        group=("affected_group", "first"),
        stocks=("ticker", "count"),
        measured=("rel_growth_30d", "count"),
        sharp_fall_stocks=("is_event", "sum"),
        mean_rel_growth=("rel_growth_30d", "mean"),
        share_beating_sector=("rel_growth_30d", lambda values: (values.dropna() > 1).mean()),
        mean_excess_vs_index=("excess_vs_index", "mean"),
    )
    labels = rows.groupby("event_id")[LABEL_COLUMNS].first()
    return table.join(labels)


def mean_with_error(values: pd.Series) -> dict[str, Any]:
    """Mean across events and its standard error."""
    clean = values.dropna()
    count = len(clean)
    error = float(clean.std() / np.sqrt(count)) if count > 1 else None
    return {
        "events": count,
        "mean": round(float(clean.mean()), 4) if count else None,
        "std_error": round(error, 4) if error is not None else None,
    }


def group_means(table: pd.DataFrame) -> dict[str, dict]:
    """Compare event means by each hand label."""
    return {
        column: {
            str(name): mean_with_error(part["mean_rel_growth"])
            for name, part in table.groupby(column)
        }
        for column in LABEL_COLUMNS
    }


def comparison_group(daily: pd.DataFrame, listed: set[str]) -> dict[str, Any]:
    """Sharp falls of index members in the test period, outside the AI lists."""
    rows = daily[daily["is_event"] & daily["in_sp500"] & ~daily["ticker"].isin(listed)]
    rows = rows[(assign_split(rows["date"]) == "test") & rows["rel_growth_30d"].notna()]
    values = rows["rel_growth_30d"]
    return {
        "events": len(values),
        "mean_rel_growth": round(float(values.mean()), 4) if len(values) else None,
        "share_beating_sector": round(float((values > 1).mean()), 4) if len(values) else None,
    }


def analyse(labels: pd.DataFrame, daily: pd.DataFrame, index_prices: pd.Series) -> dict:
    """Build the report for one horizon, as set by the target columns."""
    expanded = expand_events(labels)
    rows = add_index_excess(locate(expanded, daily), index_prices)
    table = event_table(rows)
    records = table.reset_index().assign(event_date=lambda t: t["event_date"].astype(str))
    return {
        "all_events": mean_with_error(table["mean_rel_growth"]),
        "by_label": group_means(table),
        "comparison_sharp_falls": comparison_group(daily, set(expanded["ticker"])),
        "events": json.loads(records.round(4).to_json(orient="records")),
    }


def load_daily() -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the daily table up to the protocol's last date."""
    panel = pd.read_parquet(PANEL_PATH)
    membership = pd.read_parquet(RAW_DIR / "membership.parquet")
    allowed = set(membership["ticker"]) | set(EUROPEAN_PEERS) | set(AI_EVENT_EXTRAS)
    covered = {path.stem for path in Path(RAW_DIR / "edgar").glob("*.parquet")}
    end = pd.Timestamp(TEST_END)
    today = (end + pd.Timedelta(days=1)).date()
    daily = build_daily(panel, allowed, covered, date(today.year, today.month, today.day))
    index_rows = panel[(panel["ticker"] == BENCHMARK) & (panel["date"] <= end)]
    index_prices = index_rows.set_index("date")["adj_close"].astype(float).sort_index()
    return daily, index_prices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    labels = pd.read_csv(LABELS_PATH, encoding="utf-8-sig")
    daily, index_prices = load_daily()
    report = {}
    for horizon in HORIZONS:
        daily = add_target(daily, horizon)
        report[str(horizon)] = analyse(labels, daily, index_prices)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["all_events"] for key, value in report.items()}, indent=2))
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
