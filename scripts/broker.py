"""Place paper trades through Alpaca. It uses the paper endpoint only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.settings import Settings

BASE_URL = "https://paper-api.alpaca.markets/v2"
MAX_POSITIONS = 10
HOLD_CALENDAR_DAYS = 44  # about 30 trading sessions
TIMEOUT = 30


def headers(settings: Settings) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def load_trades(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text()) if path.exists() else []


def split_due(
    trades: Sequence[dict[str, str]], today: pd.Timestamp
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separate trades that have reached their exit date from the rest."""
    due = [trade for trade in trades if pd.Timestamp(trade["exit_on"]) <= today]
    kept = [trade for trade in trades if pd.Timestamp(trade["exit_on"]) > today]
    return due, kept


def choose_new(
    signals: Sequence[dict[str, Any]], kept: Sequence[dict[str, str]]
) -> list[dict[str, Any]]:
    """Fill free slots with US tickers that are not already held."""
    held = {trade["ticker"] for trade in kept}
    free = MAX_POSITIONS - len(kept)
    fresh = [
        signal
        for signal in signals
        if signal["ticker"] not in held and "." not in signal["ticker"]
    ]
    return fresh[: max(free, 0)]


def rebalance(
    session: Any,
    settings: Settings,
    signals: Sequence[dict[str, Any]],
    state_path: Path,
    today: pd.Timestamp,
) -> dict[str, Any]:
    """Close finished trades, then open new ones with equal notional."""
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        return {"skipped": "Alpaca keys are not set"}
    auth = headers(settings)
    due, kept = split_due(load_trades(state_path), today)
    for trade in due:
        url = f"{BASE_URL}/positions/{trade['ticker']}"
        session.delete(url, headers=auth, timeout=TIMEOUT).raise_for_status()

    account = session.get(f"{BASE_URL}/account", headers=auth, timeout=TIMEOUT)
    account.raise_for_status()
    notional = round(float(account.json()["equity"]) / MAX_POSITIONS, 2)
    new = choose_new(signals, kept)
    exit_on = today + pd.Timedelta(days=HOLD_CALENDAR_DAYS)
    for signal in new:
        order = {
            "symbol": signal["ticker"],
            "notional": str(notional),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        session.post(f"{BASE_URL}/orders", json=order, headers=auth, timeout=TIMEOUT).raise_for_status()
        kept.append({"ticker": signal["ticker"], "exit_on": str(exit_on.date())})
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(kept, indent=1))
    return {"closed": len(due), "opened": len(new), "open": len(kept)}
