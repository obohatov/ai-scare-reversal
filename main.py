"""Run the whole workflow, from data to the latest trades.

    python main.py --mode file     # prepared dataset, no API calls for data
    python main.py --mode update   # add the newest data, for the daily schedule
    python main.py --mode full     # rebuild every source from scratch
    python main.py --pack          # write data/dataset.tar.gz for sharing

Signals follow the frozen `no_news` rule. The model score is informational: the
chosen model showed no signal on validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import warnings
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from scripts import broker, data_repo, notify, train, transform
from scripts.settings import DATA_DIR, REPORTS_DIR, Settings, load_settings

SIGNALS_PATH = DATA_DIR / "state" / "latest_signals.json"
TRADES_PATH = DATA_DIR / "state" / "paper_trades.json"
ARCHIVE_PATH = DATA_DIR / "dataset.tar.gz"
MAX_SIGNALS = 10


def restore_dataset(url: str) -> None:
    """Download and unpack the shared dataset unless it is already here."""
    if data_repo.PANEL_PATH.exists():
        return
    if not url:
        raise RuntimeError("Set DATA_URL to the shared dataset link.")
    import gdown

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(ARCHIVE_PATH), quiet=False)
    with tarfile.open(ARCHIVE_PATH) as archive:
        archive.extractall(DATA_DIR, filter="data")


def pack_dataset() -> Path:
    """Write the raw and processed data into one archive."""
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        for name in ("raw", "processed"):
            archive.add(DATA_DIR / name, arcname=name)
    return ARCHIVE_PATH


def update_data(session: requests.Session, settings: Settings) -> dict[str, Any]:
    """Refresh every source incrementally and rebuild the panel."""
    membership = data_repo.load_membership(session)
    tickers = data_repo.stock_tickers(membership)
    prices = data_repo.update_prices(
        tickers + data_repo.BENCHMARKS, data_repo.RAW_DIR / "prices"
    )
    macro = data_repo.fetch_macro(session, settings)
    edgar = data_repo.update_edgar(session, settings, sorted(set(membership["ticker"])))
    views = data_repo.update_wikimedia(session, settings)
    panel = data_repo.make_panel(session)
    return {
        "prices": prices,
        "macro_rows": len(macro),
        "edgar": edgar,
        "wikimedia": views,
        "panel": panel,
    }


def latest_signals(events: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Apply the frozen rule to the newest session and add the model score."""
    newest = events["date"].max()
    today = events[(events["date"] == newest) & events["in_sp500"]]
    chosen = today[(today["has_8k_2d"] == 0) & (today["has_earnings_2d"] == 0)]
    chosen = chosen.sort_values(["rel_z", "ticker"]).head(MAX_SIGNALS)
    if chosen.empty:
        return chosen.assign(score=pd.Series(dtype=float))
    labelled = events[events[train.TARGET].notna() & (events["split"] != "test")]
    candidate = train.Candidate(model["family"], json.loads(model["params"]))
    return chosen.assign(score=train.fit_and_score(candidate, labelled, chosen))


def describe(signals: pd.DataFrame, newest: pd.Timestamp) -> str:
    """Write the daily message."""
    day = pd.Timestamp(newest).date()
    if signals.empty:
        return f"AI Scare Reversal, {day}: no new signals."
    lines = [f"AI Scare Reversal, {day}: {len(signals)} signal(s), paper trading only."]
    for row in signals.itertuples():
        lines.append(f"{row.ticker}: {row.rel_ret_1d:+.1%} vs sector, score {row.score:.2f}")
    return "\n".join(lines)


def save_signals(signals: pd.DataFrame, newest: pd.Timestamp) -> list[dict[str, Any]]:
    """Save the signals and return them as plain records."""
    records = [
        {
            "ticker": row.ticker,
            "rel_ret_1d": round(float(row.rel_ret_1d), 4),
            "score": round(float(row.score), 4),
        }
        for row in signals.itertuples()
    ]
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"session": str(pd.Timestamp(newest).date()), "signals": records}
    SIGNALS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return records


def run(mode: str, settings: Settings, session: requests.Session) -> dict[str, Any]:
    """Run one mode from start to finish."""
    results: dict[str, Any] = {"mode": mode}
    if mode == "file":
        restore_dataset(settings.data_url)
    elif mode == "update":
        restore_dataset(settings.data_url)
        results["data"] = update_data(session, settings)
    else:
        data_repo.main(["--step", "all"])
    transform.main([])
    events = pd.read_parquet(transform.EVENTS_PATH)
    model = json.loads((REPORTS_DIR / "models.json").read_text())["chosen"]
    signals = latest_signals(events, model)
    newest = events["date"].max()
    records = save_signals(signals, newest)
    message = describe(signals, newest)
    print(message)
    results["signals"] = records
    results["telegram_sent"] = notify.send_message(session, settings, message)
    if mode == "update":
        today = pd.Timestamp(transform.session_cutoff_date())
        results["paper"] = broker.rebalance(session, settings, records, TRADES_PATH, today)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("file", "update", "full"), default="file")
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args(argv)
    # yfinance triggers this pandas deprecation notice on every download.
    warnings.filterwarnings("ignore", message="The inplace keyword")
    if args.pack:
        print(f"Archive: {pack_dataset()}")
        return 0
    settings = load_settings()
    session = requests.Session()
    try:
        results = run(args.mode, settings, session)
    except Exception as error:
        notify.send_message(session, settings, f"AI Scare Reversal failed: {type(error).__name__}")
        raise
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
