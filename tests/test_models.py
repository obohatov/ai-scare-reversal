"""Offline tests for the analysis and model steps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import eda, train
from scripts.transform import DUMMIES, NUMERIC


def make_events(count: int = 900, seed: int = 3) -> pd.DataFrame:
    """Events whose outcome depends on one feature, across the three splits."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(count, len(NUMERIC))), columns=NUMERIC)
    signal = frame["rel_z"].to_numpy()
    positive = (signal + rng.normal(scale=0.8, size=count)) > 0
    frame["is_positive_rel_30d"] = positive.astype(float)
    frame["rel_growth_30d"] = 1 + np.where(positive, 0.05, -0.05)
    frame["growth_30d"] = frame["rel_growth_30d"]
    frame["sector_growth_30d"] = 1.0
    for column in DUMMIES:
        frame[column] = rng.choice(["a", "b"], size=count)
    frame["ticker"] = "AAA"
    frame["date"] = pd.bdate_range("2015-01-01", periods=count)
    frame["exit_date"] = frame["date"]
    frame["entry_open"] = 1.0
    frame["exit_close"] = 1.0
    frame["in_frozen_universe"] = True
    frame["is_peer"] = False
    frame["is_event_extra"] = False
    frame["split"] = np.select(
        [frame.index < 600, frame.index < 750], ["train", "validation"], "test"
    )
    frame["has_8k_2d"] = rng.choice([0.0, 1.0, np.nan], size=count)
    frame["has_earnings_2d"] = rng.choice([0.0, 1.0], size=count)
    return frame


def small_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        train,
        "candidates",
        lambda: [
            train.Candidate("logistic", {"C": 0.1}),
            train.Candidate("tree", {"max_depth": 3, "min_samples_leaf": 20}),
            train.Candidate(
                "forest",
                {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 20},
            ),
            train.Candidate(
                "xgboost",
                {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1},
            ),
        ],
    )


def test_news_group_separates_earnings_other_news_and_unknown() -> None:
    frame = pd.DataFrame(
        {"has_8k_2d": [0.0, 1.0, 1.0, np.nan], "has_earnings_2d": [0.0, 0.0, 1.0, np.nan]}
    )
    assert eda.news_group(frame).tolist() == ["no_8k", "other_8k", "earnings_8k", "unknown"]


def test_eda_never_reads_the_test_period() -> None:
    events = make_events()
    events.loc[events["split"] == "test", "is_positive_rel_30d"] = 1.0
    first = eda.analyse(events)
    events.loc[events["split"] == "test", "is_positive_rel_30d"] = 0.0
    assert eda.analyse(events) == first
    assert set(first) == {"train", "validation"}


def test_group_table_skips_tiny_groups() -> None:
    events = make_events(count=40)
    by = pd.Series(["big"] * 35 + ["tiny"] * 5)
    assert set(eda.group_table(events, by)) == {"big"}


def test_build_matrix_keeps_training_columns_for_new_rows() -> None:
    events = make_events()
    fit = train.build_matrix(events)
    later = events.head(5).assign(sector="never_seen")
    matrix = train.build_matrix(later, list(fit.columns))
    assert list(matrix.columns) == list(fit.columns)
    assert "sector_never_seen" not in matrix.columns


def test_run_finds_the_signal_with_every_model_family(monkeypatch) -> None:
    small_grid(monkeypatch)
    report, predictions = train.run(make_events())
    assert report["chosen"]["validation_auc"] > 0.7
    assert {item["family"] for item in report["all_candidates"]} == {
        "logistic", "tree", "forest", "xgboost",
    }
    assert set(predictions["split"]) == {"validation", "test"}
    assert predictions["score"].between(0, 1).all()


def test_model_choice_ignores_test_outcomes(monkeypatch) -> None:
    small_grid(monkeypatch)
    events = make_events()
    events.loc[events["split"] == "test", "is_positive_rel_30d"] = 1.0
    first, _ = train.run(events)
    events.loc[events["split"] == "test", "is_positive_rel_30d"] = 0.0
    second, _ = train.run(events)
    assert first["chosen"] == second["chosen"]


def test_unlabelled_recent_events_still_get_scores(monkeypatch) -> None:
    small_grid(monkeypatch)
    events = make_events()
    last = events.index[-5:]
    events.loc[last, "is_positive_rel_30d"] = np.nan
    _, predictions = train.run(events)
    assert predictions["score"].notna().all()
    assert (predictions["split"] == "test").sum() == 150


def test_top_share_uses_the_highest_scores() -> None:
    frame = pd.DataFrame(
        {"is_positive_rel_30d": [0.0, 1.0, 1.0, 0.0], "rel_growth_30d": [0.9, 1.1, 1.2, 0.8]}
    )
    result = train.top_share_metrics(np.array([0.1, 0.9, 0.8, 0.2]), frame, share=0.5)
    assert result == {
        "top_events": 2,
        "top_precision": 1.0,
        "top_median_rel_growth": 1.15,
    }
