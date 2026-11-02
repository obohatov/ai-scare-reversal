"""Run the one declared follow-up: the same models at shorter horizons.

Run from the repository root:

    python -m scripts.horizon_check

Only the training and validation periods are compared. Column names keep the
30d suffix inside this script because the code reuses the same pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from scripts.data_repo import DATA_DIR, PANEL_PATH, RAW_DIR
from scripts.train import TARGET, compare_models
from scripts.transform import add_target, build_daily, events_from_daily, frozen_members
from scripts.universe import AI_EVENT_EXTRAS, EUROPEAN_PEERS

HORIZONS = (5, 10)
REPORT_PATH = DATA_DIR.parent / "reports" / "horizon_check.json"


def summarise_horizon(events: pd.DataFrame) -> dict:
    """Compare every model at one horizon, on validation only."""
    labelled = events[events[TARGET].notna()]
    train = labelled[labelled["split"] == "train"]
    validation = labelled[labelled["split"] == "validation"]
    table = compare_models(train, validation)
    best = table.groupby("family", sort=False).head(1)
    return {
        "train_base_rate": round(float(train[TARGET].mean()), 4),
        "validation_base_rate": round(float(validation[TARGET].mean()), 4),
        "best_per_family": best.to_dict("records"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    panel = pd.read_parquet(PANEL_PATH)
    membership = pd.read_parquet(RAW_DIR / "membership.parquet")
    allowed = set(membership["ticker"]) | set(EUROPEAN_PEERS) | set(AI_EVENT_EXTRAS)
    covered = {path.stem for path in (RAW_DIR / "edgar").glob("*.parquet")}
    today = datetime.now(timezone.utc).date()
    daily = build_daily(panel, allowed, covered, today)
    frozen = frozen_members(membership)
    report = {}
    for horizon in HORIZONS:
        print(f"Horizon: {horizon} sessions", flush=True)
        events = events_from_daily(add_target(daily, horizon), frozen)
        report[str(horizon)] = summarise_horizon(events)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
