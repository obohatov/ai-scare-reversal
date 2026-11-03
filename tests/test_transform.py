"""Offline tests for feature building, on a small synthetic panel."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import transform as tf

TICKERS = [f"T{number}" for number in range(8)]
DROP_DAY = 400
TODAY = date(2030, 1, 1)


def make_panel(sessions: int = 520, drop: float = 0.15) -> pd.DataFrame:
    """Eight stocks in one sector; T0 falls sharply on one day."""
    dates = pd.bdate_range("2021-01-04", periods=sessions)
    rng = np.random.default_rng(7)
    frames = []
    for ticker in TICKERS:
        returns = rng.normal(0.0003, 0.01, sessions)
        if ticker == "T0":
            returns[DROP_DAY] = -drop
        close = 100 * np.cumprod(1 + returns)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000.0,
                    "in_sp500": True,
                    "sector": "Technology",
                    "has_8k": False,
                    "has_earnings_8k": False,
                    "pageviews": 100.0,
                    "dgs10": 4.0,
                    "dgs2": 3.0,
                    "vixcls": 18.0,
                    "dff": 3.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run(panel: pd.DataFrame) -> pd.DataFrame:
    return tf.build_events(panel, set(TICKERS), set(TICKERS), set(TICKERS), TODAY)


def test_sharp_fall_becomes_an_event() -> None:
    events = run(make_panel())
    expected = pd.bdate_range("2021-01-04", periods=520)[DROP_DAY]
    match = events[(events["ticker"] == "T0") & (events["date"] == expected)]
    assert len(match) == 1
    assert match.iloc[0]["rel_ret_1d"] < -0.12
    assert match.iloc[0]["drop_size_bin"] == "10_to_15"


def test_sector_return_excludes_the_stock_itself() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 7),
            "ticker": list("ABCDEFG"),
            "sector": "Technology",
            "ret_1d": [-0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = tf.add_sector_features(frame.copy())
    assert result.loc[0, "sector_ret_1d"] == pytest.approx(0.0)
    assert result.loc[1, "sector_ret_1d"] == pytest.approx(-0.10)


def test_features_do_not_change_when_the_future_changes() -> None:
    base = make_panel()
    cut = base["date"].sort_values().unique()[DROP_DAY]
    altered = base.copy()
    later = altered["date"] > cut
    for column in ("open", "high", "low", "close", "adj_close"):
        altered.loc[later, column] *= 3.0
    altered.loc[later, "volume"] *= 9.0
    first = run(base).set_index(["ticker", "date"])
    second = run(altered).set_index(["ticker", "date"])
    index = (("T0", pd.Timestamp(cut)))
    for column in tf.NUMERIC:
        one, two = first.loc[index, column], second.loc[index, column]
        assert (pd.isna(one) and pd.isna(two)) or one == pytest.approx(two), column


def test_target_runs_from_next_open_to_the_exit_close() -> None:
    panel = make_panel()
    events = run(panel)
    row = events[events["ticker"] == "T0"].iloc[-1]
    stock = panel[panel["ticker"] == "T0"].reset_index(drop=True)
    position = stock.index[stock["date"] == row["date"]][0]
    expected = stock.loc[position + 31, "adj_close"] / stock.loc[position + 1, "open"]
    assert row["growth_30d"] == pytest.approx(expected)
    assert row["exit_date"] == stock.loc[position + 31, "date"]


def test_target_accepts_a_shorter_horizon() -> None:
    panel = make_panel()
    daily = tf.build_daily(panel, set(TICKERS), set(TICKERS), TODAY)
    short = tf.add_target(daily.copy(), horizon=5)
    row = short[(short["ticker"] == "T1")].iloc[100]
    stock = panel[panel["ticker"] == "T1"].reset_index(drop=True)
    position = stock.index[stock["date"] == row["date"]][0]
    expected = stock.loc[position + 6, "adj_close"] / stock.loc[position + 1, "open"]
    assert row["growth_30d"] == pytest.approx(expected)


def test_aligned_target_uses_one_span_for_stock_and_sector() -> None:
    panel = make_panel()
    daily = tf.build_daily(panel, set(TICKERS), set(TICKERS), TODAY)
    aligned = tf.align_target(daily.copy())
    row = aligned[aligned["ticker"] == "T1"].iloc[100]
    stock = panel[panel["ticker"] == "T1"].reset_index(drop=True)
    position = stock.index[stock["date"] == row["date"]][0]
    expected = stock.loc[position + 31, "adj_close"] / stock.loc[position, "adj_close"]
    assert row["growth_30d"] == pytest.approx(expected)
    assert row["rel_growth_30d"] == pytest.approx(expected / row["sector_growth_30d"])


def test_target_is_empty_without_enough_future() -> None:
    panel = make_panel(sessions=DROP_DAY + 10)
    events = run(panel)
    row = events[events["ticker"] == "T0"].iloc[-1]
    assert pd.isna(row["growth_30d"])
    assert pd.isna(row["is_positive_rel_30d"])


def test_unfinished_session_is_removed() -> None:
    panel = make_panel()
    today = pd.Timestamp(panel["date"].max()).date()
    rows = tf.select_rows(panel, set(TICKERS), today)
    assert rows["date"].max() < pd.Timestamp(today)


def test_benchmarks_and_unknown_names_are_removed() -> None:
    panel = make_panel()
    extra = panel[panel["ticker"] == "T1"].assign(ticker="^GSPC")
    rows = tf.select_rows(pd.concat([panel, extra]), set(TICKERS) | {"^GSPC"}, TODAY)
    assert "^GSPC" not in set(rows["ticker"])
    assert "T9" not in set(rows["ticker"])


def test_cooldown_keeps_only_the_first_of_a_cluster() -> None:
    frame = pd.DataFrame(
        {
            "ticker": "A",
            "rel_z": [-3.0, -3.0, 0.0, 0.0, 0.0, 0.0, -3.0],
            "rel_ret_1d": [-0.05] * 7,
        }
    )
    assert tf.find_events(frame, cooldown=5).tolist() == [
        True, False, False, False, False, False, True,
    ]


def test_split_leaves_a_gap_before_each_later_period() -> None:
    dates = pd.Series(
        pd.to_datetime(["2022-06-01", "2022-12-15", "2023-03-01", "2024-12-20", "2025-01-02"])
    )
    assert tf.assign_split(dates).tolist() == [
        "train", "embargo", "validation", "embargo", "test",
    ]


def test_missing_edgar_history_is_unknown_not_false() -> None:
    panel = make_panel()
    events = tf.build_events(panel, set(TICKERS), set(TICKERS), {"T1"}, TODAY)
    assert events.loc[events["ticker"] == "T0", "has_8k_2d"].isna().all()


def test_summary_hides_test_outcomes() -> None:
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2025-06-02"]),
            "ticker": ["A", "B"],
            "split": ["train", "test"],
            "is_positive_rel_30d": [1.0, 1.0],
            "in_frozen_universe": [True, True],
            "has_8k_2d": [0.0, 1.0],
            "pageviews_ratio_60": [1.0, 1.0],
        }
    )
    summary = tf.summarise(events)
    assert "base_rate" in summary["train"]
    assert "base_rate" not in summary["test"]


def test_daily_baseline_covers_only_train_and_validation() -> None:
    daily = tf.build_daily(make_panel(sessions=900), set(TICKERS), set(TICKERS), TODAY)
    baseline = tf.daily_baseline(daily)
    assert set(baseline) == {"train", "validation"}
    assert 0 < baseline["train"]["beat_sector"] < 1


def test_feature_lists_are_large_enough_and_unique() -> None:
    assert len(tf.NUMERIC) == len(set(tf.NUMERIC))
    assert len(tf.NUMERIC) >= 20
