"""Offline tests for the AI event analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import ai_slice as ai

DATES = pd.bdate_range("2026-01-05", periods=10)


def labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "event_date": ["2026-01-05", "2026-01-10"],
            "affected_group": ["one", "two"],
            "named_tickers": ["AAA BBB", np.nan],
            "proposed_tickers": [np.nan, "CCC"],
            "substitutes_paid_workflow": ["yes", "no"],
            "depends_on_proprietary_data": ["no", "partly"],
            "product_status": ["shipped", "none"],
        }
    )


def daily() -> pd.DataFrame:
    frames = []
    for ticker, rel in (("AAA", 1.10), ("BBB", 0.90), ("CCC", 1.20), ("ZZZ", 1.05)):
        frames.append(
            pd.DataFrame(
                {
                    "date": DATES,
                    "ticker": ticker,
                    "rel_ret_1d": -0.05,
                    "is_event": ticker == "ZZZ",
                    "in_sp500": True,
                    "exit_date": DATES[-1],
                    "growth_30d": rel,
                    "rel_growth_30d": rel,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_expand_events_lists_every_ticker_with_its_source() -> None:
    expanded = ai.expand_events(labels())
    assert expanded["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert expanded["ticker_source"].tolist() == [
        "named_tickers", "named_tickers", "proposed_tickers",
    ]


def test_weekend_event_moves_to_the_next_session() -> None:
    rows = ai.locate(ai.expand_events(labels()), daily())
    row = rows[rows["ticker"] == "CCC"].iloc[0]
    assert row["date"] == pd.Timestamp("2026-01-12")


def test_each_event_counts_once() -> None:
    index = pd.Series(100.0, index=DATES)
    report = ai.analyse(labels(), daily(), index)
    assert report["all_events"]["events"] == 2
    assert report["all_events"]["mean"] == pytest.approx(1.1)


def test_groups_follow_the_hand_labels() -> None:
    report = ai.analyse(labels(), daily(), pd.Series(100.0, index=DATES))
    groups = report["by_label"]["substitutes_paid_workflow"]
    assert groups["yes"]["mean"] == pytest.approx(1.0)
    assert groups["no"]["mean"] == pytest.approx(1.2)


def test_excess_uses_the_index_over_the_same_dates() -> None:
    index = pd.Series(np.linspace(100, 110, 10), index=DATES)
    frame = pd.DataFrame(
        {"date": [DATES[0]], "exit_date": [DATES[-1]], "growth_30d": [1.2]}
    )
    assert ai.add_index_excess(frame, index)["excess_vs_index"].iloc[0] == pytest.approx(0.1)


def test_comparison_group_excludes_listed_stocks_and_earlier_periods() -> None:
    frame = daily()
    frame["date"] = frame["date"] + pd.DateOffset(years=0)
    result = ai.comparison_group(frame, {"AAA", "BBB", "CCC"})
    assert result["events"] == 10
    assert result["mean_rel_growth"] == pytest.approx(1.05)
    old = frame.assign(date=pd.Timestamp("2020-01-02"))
    assert ai.comparison_group(old, set())["events"] == 0
