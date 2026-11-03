"""Repeat key rates with stock and sector measured over one matched span.

Run from the repository root:

    python -m scripts.robustness

This is a check added after the test period was opened. It leaves every frozen
result unchanged and uses close-to-close growth, so it does not describe a
tradable entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import pandas as pd

from scripts.ai_slice import LABELS_PATH, analyse, load_daily
from scripts.settings import REPORTS_DIR
from scripts.transform import add_target, align_target, assign_split

REPORT_PATH = REPORTS_DIR / "robustness_aligned_target.json"


def beat_rates(daily: pd.DataFrame) -> dict[str, Any]:
    """Compare ordinary days with sharp falls in every period."""
    rows = daily[daily["in_sp500"] & daily["is_positive_rel_30d"].notna()]
    splits = assign_split(rows["date"])
    report = {}
    for split in ("train", "validation", "test"):
        part = rows[splits == split]
        events = part[part["is_event"]]
        ordinary = part[~part["is_event"]]
        report[split] = {
            "ordinary_days": round(float(ordinary["is_positive_rel_30d"].mean()), 4),
            "sharp_falls": round(float(events["is_positive_rel_30d"].mean()), 4),
            "sharp_fall_count": len(events),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    labels = pd.read_csv(LABELS_PATH, encoding="utf-8-sig")
    daily, index_prices = load_daily()
    daily = align_target(add_target(daily))
    ai_report = analyse(labels, daily, index_prices)
    report = {
        "note": "Post-test check with matched close-to-close spans; not a tradable result.",
        "beat_sector": beat_rates(daily),
        "ai_events": ai_report["all_events"],
        "ai_comparison_sharp_falls": ai_report["comparison_sharp_falls"],
        "ai_by_label": ai_report["by_label"]["substitutes_paid_workflow"],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
