"""Explore sharp falls on the training and validation periods only.

Run from the repository root:

    python -m scripts.eda

The test period is not read, so nothing here can shape later choices with
knowledge of its outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.data_repo import DATA_DIR
from scripts.transform import DUMMIES, EVENTS_PATH, NUMERIC

REPORT_PATH = DATA_DIR.parent / "reports" / "eda.json"
TARGET = "is_positive_rel_30d"
TOP_FEATURES = 15
MIN_GROUP_SIZE = 30


def news_group(events: pd.DataFrame) -> pd.Series:
    """Sort falls the way Savor (2012) does: with or without company news."""
    group = pd.Series("no_8k", index=events.index, dtype="object")
    group[events["has_8k_2d"] == 1] = "other_8k"
    group[events["has_earnings_2d"] == 1] = "earnings_8k"
    group[events["has_8k_2d"].isna()] = "unknown"
    return group


def group_table(events: pd.DataFrame, by: pd.Series) -> dict:
    """Return outcome statistics per group, with a standard error."""
    table = {}
    for name, rows in events.groupby(by, observed=True):
        count = len(rows)
        if count < MIN_GROUP_SIZE:
            continue
        rate = float(rows[TARGET].mean())
        table[str(name)] = {
            "events": count,
            "beat_sector": round(rate, 4),
            "std_error": round(float(np.sqrt(rate * (1 - rate) / count)), 4),
            "median_rel_growth": round(float(rows["rel_growth_30d"].median()), 4),
        }
    return table


def by_year(events: pd.DataFrame) -> dict:
    """Show whether a pattern holds across years rather than in one episode."""
    return group_table(events, events["date"].dt.year)


def top_correlations(events: pd.DataFrame, limit: int = TOP_FEATURES) -> dict:
    """Rank numeric features by absolute rank correlation with the target."""
    values = events[NUMERIC].corrwith(events[TARGET], method="spearman").dropna()
    order = values.abs().sort_values(ascending=False).index[:limit]
    return {name: round(float(values[name]), 4) for name in order}


def analyse(events: pd.DataFrame) -> dict:
    """Build the report from labelled rows outside the test period."""
    rows = events[events["split"].isin(["train", "validation"])]
    rows = rows[rows[TARGET].notna()]
    report: dict = {}
    for split, part in rows.groupby("split"):
        tables = {name: group_table(part, part[name]) for name in DUMMIES}
        report[str(split)] = {
            "events": len(part),
            "news": group_table(part, news_group(part)),
            "breakdowns": tables,
            "year": by_year(part),
            "top_correlations": top_correlations(part),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    report = analyse(pd.read_parquet(EVENTS_PATH))
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for split in ("train", "validation"):
        print(split, json.dumps(report[split]["news"]))
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
