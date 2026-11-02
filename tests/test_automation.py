"""Offline tests for the daily workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

import main as app
from scripts import broker, data_repo, notify, transform
from scripts.settings import Settings

KEYS = Settings(alpaca_api_key="key", alpaca_secret_key="secret")


class FakeResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("get", url, None))
        return FakeResponse({"equity": "100000"})

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("post", url, json))
        return FakeResponse()

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("delete", url, None))
        return FakeResponse()


def price_frame(dates, adjusted) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.to_datetime(dates), "ticker": "AAA", "adj_close": adjusted, "close": adjusted}
    )


def test_merge_prices_rescales_history_after_a_dividend() -> None:
    old = price_frame(["2026-01-05", "2026-01-06"], [100.0, 110.0])
    new = price_frame(["2026-01-06", "2026-01-07"], [99.0, 100.0])
    merged = data_repo.merge_prices(old, new)
    assert merged["adj_close"].tolist() == pytest.approx([90.0, 99.0, 100.0])


def test_merge_prices_without_overlap_keeps_history() -> None:
    old = price_frame(["2026-01-05"], [100.0])
    new = price_frame(["2026-01-07"], [50.0])
    assert data_repo.merge_prices(old, new)["adj_close"].tolist() == [100.0, 50.0]


def test_late_evening_run_keeps_the_finished_session() -> None:
    evening = datetime(2026, 11, 3, 22, 30, tzinfo=timezone.utc)
    noon = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    assert str(transform.session_cutoff_date(evening)) == "2026-11-04"
    assert str(transform.session_cutoff_date(noon)) == "2026-11-03"


def test_notify_stays_silent_without_a_token() -> None:
    session = FakeSession()
    assert notify.send_message(session, Settings(), "hello") is False
    assert session.calls == []


def test_notify_never_stops_the_run_on_failure() -> None:
    class Broken(FakeSession):
        def post(self, url, json=None, headers=None, timeout=None):
            raise requests.ConnectionError("offline")

    settings = Settings(telegram_bot_token="token", telegram_chat_id="1")
    assert notify.send_message(Broken(), settings, "hello") is False


def test_broker_skips_without_keys(tmp_path) -> None:
    session = FakeSession()
    result = broker.rebalance(session, Settings(), [], tmp_path / "t.json", pd.Timestamp("2026-11-03"))
    assert "skipped" in result
    assert session.calls == []


def test_broker_uses_only_the_paper_endpoint() -> None:
    assert broker.BASE_URL.startswith("https://paper-api.alpaca.markets")


def test_broker_closes_due_trades_and_fills_free_slots(tmp_path) -> None:
    path = tmp_path / "trades.json"
    path.write_text(
        json.dumps(
            [
                {"ticker": "OLD", "exit_on": "2026-11-01"},
                {"ticker": "KEEP", "exit_on": "2026-12-01"},
            ]
        )
    )
    session = FakeSession()
    signals = [{"ticker": "NEW"}, {"ticker": "KEEP"}, {"ticker": "REL.L"}]
    result = broker.rebalance(session, KEYS, signals, path, pd.Timestamp("2026-11-03"))
    assert result == {"closed": 1, "opened": 1, "open": 2}
    orders = [call for call in session.calls if call[0] == "post"]
    assert orders[0][2]["symbol"] == "NEW"
    assert orders[0][2]["notional"] == "10000.0"
    assert ("delete", f"{broker.BASE_URL}/positions/OLD", None) in session.calls


def test_broker_respects_the_position_limit() -> None:
    kept = [{"ticker": f"T{number}", "exit_on": "2027-01-01"} for number in range(10)]
    assert broker.choose_new([{"ticker": "NEW"}], kept) == []


def events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-11-03"] * 3),
            "ticker": ["AAA", "BBB", "CCC"],
            "in_sp500": [True, True, False],
            "has_8k_2d": [0.0, 1.0, 0.0],
            "has_earnings_2d": [0.0, 0.0, 0.0],
            "rel_z": [-3.0, -4.0, -5.0],
            "rel_ret_1d": [-0.05, -0.06, -0.07],
        }
    )


def test_latest_signals_apply_the_frozen_rule(monkeypatch) -> None:
    monkeypatch.setattr(app.train, "fit_and_score", lambda candidate, fit, rows: [0.4] * len(rows))
    events = events_frame().assign(is_positive_rel_30d=1.0, split="train")
    model = {"family": "tree", "params": "{}"}
    assert app.latest_signals(events, model)["ticker"].tolist() == ["AAA"]


def test_describe_handles_a_quiet_day() -> None:
    frame = events_frame().iloc[0:0].assign(score=[])
    assert "no new signals" in app.describe(frame, pd.Timestamp("2026-11-03"))


def test_pack_and_restore_round_trip(monkeypatch, tmp_path) -> None:
    for folder in ("raw", "processed"):
        (tmp_path / folder).mkdir()
    (tmp_path / "processed" / "panel.parquet").write_text("x")
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app, "ARCHIVE_PATH", tmp_path / "dataset.tar.gz")
    assert app.pack_dataset().exists()


def test_restore_explains_a_missing_link(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app.data_repo, "PANEL_PATH", tmp_path / "none.parquet")
    with pytest.raises(RuntimeError, match="DATA_URL"):
        app.restore_dataset("")
