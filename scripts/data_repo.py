"""Download every data source and combine them into one table.

Run from the repository root:

    python -m scripts.data_repo --step all

Each step saves its result under data/, which stays out of Git. A repeated run
skips what is already on disk, so an interrupted download can simply be resumed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from scripts.check_sources import (
    EDGAR_PAUSE_SECONDS,
    EDGAR_SUBMISSIONS_URL,
    EDGAR_TICKERS_URL,
    FRED_CSV_URL,
    PAGEVIEWS_START,
    PRICE_START,
    WIKI_SP500_URL,
    WIKIMEDIA_URL,
    build_user_agent,
    fetch,
    parse_sp500_tables,
    to_yahoo_symbol,
)
from scripts.settings import DATA_DIR, REPORTS_DIR, Settings, load_settings
from scripts.universe import (
    AI_EVENT_EXTRAS,
    BENCHMARKS,
    EUROPEAN_ARTICLES,
    EUROPEAN_PEERS,
    FRED_SERIES,
    TICKER_RENAMES,
)

RAW_DIR = DATA_DIR / "raw"
PANEL_PATH = DATA_DIR / "processed" / "panel.parquet"

# Pinned to one commit so that every rebuild uses the same membership history.
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/"
    "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
EDGAR_OLDER_FILE_URL = "https://data.sec.gov/submissions/{name}"
MEDIAWIKI_API_URL = "https://en.wikipedia.org/w/api.php"

CHUNK_SIZE = 40
CHUNK_PAUSE_SECONDS = 2.0
RATE_LIMIT_PAUSE_SECONDS = 60.0
SECTOR_PAUSE_SECONDS = 0.2
TITLES_PER_REQUEST = 50
UPDATE_LOOKBACK_DAYS = 7
MARKET_CLOSE_HOUR_ET = 16
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
STEPS = (
    "membership",
    "prices",
    "macro",
    "sectors",
    "edgar",
    "wikimedia",
    "panel",
)


def with_date_unit(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Give date columns one resolution; merges refuse mixed ones."""
    return frame.assign(
        **{name: pd.to_datetime(frame[name]).astype("datetime64[ns]") for name in columns}
    )


def safe_name(ticker: str) -> str:
    """Make a ticker usable as a file name: ^GSPC becomes _GSPC."""
    return re.sub(r"[^A-Za-z0-9.-]", "_", ticker)


# ---------------------------------------------------------------------------
# Parsers and transformations: pure functions, covered by tests
# ---------------------------------------------------------------------------


def parse_membership(text: str, start: str = PRICE_START) -> pd.DataFrame:
    """Return date and ticker rows, keeping the last snapshot before the start."""
    frame = pd.read_csv(StringIO(text), parse_dates=["date"])
    before = frame.loc[frame["date"] <= start, "date"]
    first_date = before.max() if not before.empty else frame["date"].min()
    frame = frame[frame["date"] >= first_date]
    long = frame.assign(ticker=frame["tickers"].str.split(",")).explode("ticker")
    long["ticker"] = long["ticker"].map(to_yahoo_symbol).replace(TICKER_RENAMES)
    long = long.drop_duplicates(["date", "ticker"])
    return long[["date", "ticker"]].reset_index(drop=True)


def membership_flags(
    membership: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Mark whether each ticker was an index member on each trading day."""
    snapshots = pd.crosstab(membership["date"], membership["ticker"]).astype(bool)
    daily = snapshots.reindex(dates, method="ffill").fillna(False).astype(bool)
    daily.index.name = "date"
    daily.columns.name = "ticker"
    flags = daily.stack().rename("in_sp500").reset_index()
    return flags


def price_rows(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker from a yf.download(group_by='ticker') frame."""
    try:
        rows = frame[ticker].dropna(subset=["Adj Close"])
    except KeyError:
        return pd.DataFrame()
    if rows.empty:
        return pd.DataFrame()
    rows = rows.rename(columns=lambda name: str(name).lower().replace(" ", "_"))
    rows = rows.rename_axis("date").reset_index()
    rows["date"] = pd.to_datetime(rows["date"]).dt.tz_localize(None)
    rows["ticker"] = ticker
    return rows[["date", "ticker", *PRICE_COLUMNS]]


def extract_8k(columns: dict[str, list[Any]], ticker: str) -> pd.DataFrame:
    """Keep the 8-K rows from one block of EDGAR filing columns."""
    names = ("form", "filingDate", "acceptanceDateTime", "items")
    frame = pd.DataFrame({name: columns.get(name, []) for name in names})
    frame = frame[frame["form"].isin(["8-K", "8-K/A"])]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "filing_date": pd.to_datetime(frame["filingDate"]),
            "accepted_at": frame["acceptanceDateTime"].astype(str),
            "items": frame["items"].fillna("").astype(str),
        }
    )


def clean_article_title(security: str) -> str:
    """Turn 'Alphabet Inc. (Class A)' into a Wikipedia article title."""
    title = re.sub(r"\s*\(.*?\)\s*$", "", security).strip()
    return title.replace(" ", "_")


def resolve_titles(payload: dict[str, Any], titles: Sequence[str]) -> dict[str, str]:
    """Follow normalisation and redirects; drop missing and disambiguation pages."""
    query = payload.get("query", {})
    step_maps = [
        {item["from"]: item["to"] for item in query.get(key, [])}
        for key in ("normalized", "redirects")
    ]
    usable = {
        page["title"]
        for page in query.get("pages", {}).values()
        if "missing" not in page
        and "disambiguation" not in page.get("pageprops", {})
    }
    resolved: dict[str, str] = {}
    for title in titles:
        current = title
        for mapping in step_maps:
            current = mapping.get(current, current)
        if current in usable:
            resolved[title] = current.replace(" ", "_")
    return resolved


def news_flags(filings: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Date each 8-K by the first session that could react to it.

    EDGAR acceptance times are treated as US Eastern time, so a filing accepted
    at or after the 16:00 close counts for the next trading day.
    """
    empty = pd.DataFrame(columns=["date", "ticker", "has_8k", "has_earnings_8k"])
    if filings.empty:
        return empty
    rows = with_date_unit(filings, "filing_date")
    calendar = with_date_unit(calendar, "date")
    accepted_hour = pd.to_numeric(
        rows["accepted_at"].str.extract(r"T(\d{2}):")[0], errors="coerce"
    )
    after_close = accepted_hour.fillna(0) >= MARKET_CLOSE_HOUR_ET
    rows["ready"] = rows["filing_date"] + pd.to_timedelta(after_close.astype(int), "D")
    rows = with_date_unit(rows, "ready")
    rows["is_earnings"] = rows["items"].str.contains("2.02", regex=False)
    aligned = pd.merge_asof(
        rows.sort_values("ready"),
        calendar[["date", "ticker"]].sort_values("date"),
        left_on="ready",
        right_on="date",
        by="ticker",
        direction="forward",
    ).dropna(subset=["date"])
    flags = aligned.groupby(["date", "ticker"], as_index=False).agg(
        has_earnings_8k=("is_earnings", "max")
    )
    flags["has_8k"] = True
    return flags


def build_panel(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    macro: pd.DataFrame,
    sectors: dict[str, str],
    filings: pd.DataFrame,
    pageviews: pd.DataFrame,
) -> pd.DataFrame:
    """Combine every source into one row per ticker and trading day."""
    prices = with_date_unit(prices, "date")
    membership = with_date_unit(membership, "date")
    macro = with_date_unit(macro, "date")
    pageviews = with_date_unit(pageviews, "date")
    panel = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel = panel.merge(
        membership_flags(membership, dates), on=["date", "ticker"], how="left"
    )
    panel["in_sp500"] = panel["in_sp500"].fillna(False).astype(bool)
    panel["sector"] = panel["ticker"].map(sectors).fillna("Unknown").astype("category")

    # Macro values become usable the business day after their date.
    available = macro.sort_values("date").copy()
    available["date"] = available["date"] + pd.offsets.BDay(1)
    available = with_date_unit(available, "date")
    panel = pd.merge_asof(panel.sort_values("date"), available, on="date")

    flags = news_flags(filings, panel[["date", "ticker"]])
    panel = panel.merge(flags, on=["date", "ticker"], how="left")
    for column in ("has_8k", "has_earnings_8k"):
        panel[column] = panel[column].astype("boolean").fillna(False).astype(bool)

    if not pageviews.empty:
        panel = panel.merge(pageviews, on=["date", "ticker"], how="left")
    else:
        panel["pageviews"] = np.nan

    numeric = panel.select_dtypes("float64").columns
    panel[numeric] = panel[numeric].astype("float32")
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Download steps
# ---------------------------------------------------------------------------


def load_membership(session: requests.Session, refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "membership.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    membership = parse_membership(fetch(session, MEMBERSHIP_URL).text)
    path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(path)
    return membership


def load_constituents(session: requests.Session, settings: Settings) -> pd.DataFrame:
    path = RAW_DIR / "constituents.parquet"
    if path.exists():
        return pd.read_parquet(path)
    headers = {"User-Agent": build_user_agent(settings)}
    html = fetch(session, WIKI_SP500_URL, headers=headers).text
    constituents, _, _ = parse_sp500_tables(html)
    path.parent.mkdir(parents=True, exist_ok=True)
    constituents.to_parquet(path)
    return constituents


def stock_tickers(membership: pd.DataFrame) -> list[str]:
    """Index members since the start date, European peers and event names."""
    members = set(membership["ticker"])
    extras = [ticker for ticker in AI_EVENT_EXTRAS if ticker not in members]
    return sorted(members) + extras + EUROPEAN_PEERS


def fetch_prices(
    tickers: Sequence[str],
    output_dir: Path,
    start: str = PRICE_START,
    chunk_size: int = CHUNK_SIZE,
    pause_seconds: float = CHUNK_PAUSE_SECONDS,
    retry_pause_seconds: float = RATE_LIMIT_PAUSE_SECONDS,
) -> dict[str, Any]:
    """Save one file per ticker; tickers already on disk are skipped."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tried_path = output_dir.parent / "prices_no_data.json"
    no_data = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()
    todo = [
        ticker
        for ticker in tickers
        if ticker not in no_data
        and not (output_dir / f"{safe_name(ticker)}.parquet").exists()
    ]
    started = time.perf_counter()
    for begin in range(0, len(todo), chunk_size):
        chunk = todo[begin : begin + chunk_size]
        saved = 0
        for attempt in range(2):
            frame = yf.download(
                chunk,
                start=start,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
                timeout=30,
                multi_level_index=True,
            )
            for ticker in chunk:
                rows = price_rows(frame, ticker) if frame is not None else pd.DataFrame()
                if rows.empty:
                    continue
                rows.to_parquet(output_dir / f"{safe_name(ticker)}.parquet")
                saved += 1
            # A chunk with no data at all usually means a block, not bad tickers.
            if saved or attempt == 1:
                break
            print(f"  Empty chunk, waiting {retry_pause_seconds:.0f}s...", flush=True)
            time.sleep(retry_pause_seconds)
        if saved:
            no_data.update(
                ticker
                for ticker in chunk
                if not (output_dir / f"{safe_name(ticker)}.parquet").exists()
            )
        done = min(begin + chunk_size, len(todo))
        print(f"  Prices: {done}/{len(todo)} tickers", flush=True)
        if done < len(todo):
            time.sleep(pause_seconds)
    tried_path.write_text(json.dumps(sorted(no_data), indent=1))
    saved_total = sum(
        (output_dir / f"{safe_name(ticker)}.parquet").exists() for ticker in tickers
    )
    return {
        "requested": len(tickers),
        "saved": saved_total,
        "no_data": sorted(no_data & set(tickers)),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }


def fetch_macro(session: requests.Session, settings: Settings) -> pd.DataFrame:
    """Save the FRED series as one wide table."""
    headers = {"User-Agent": build_user_agent(settings)}
    series = []
    for name in FRED_SERIES:
        text = fetch(session, FRED_CSV_URL.format(series=name), headers=headers).text
        frame = pd.read_csv(StringIO(text))
        frame.columns = ["date", name.lower()]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[name.lower()] = pd.to_numeric(frame[name.lower()], errors="coerce")
        series.append(frame.dropna(subset=["date"]).set_index("date"))
    macro = pd.concat(series, axis=1).sort_index().ffill().reset_index()
    macro = macro[macro["date"] >= "2009-12-01"]
    path = RAW_DIR / "macro.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(path)
    return macro


def fetch_sectors(tickers: Sequence[str]) -> dict[str, Any]:
    """Save Yahoo's sector per ticker, one taxonomy for US and European names."""
    path = RAW_DIR / "sectors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sectors: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    todo = [ticker for ticker in tickers if ticker not in sectors]
    for number, ticker in enumerate(todo, start=1):
        try:
            sectors[ticker] = yf.Ticker(ticker).info.get("sector") or "Unknown"
        except Exception:  # delisted names and rate limits both land here
            sectors[ticker] = "Unknown"
        if number % 50 == 0 or number == len(todo):
            path.write_text(json.dumps(sectors, indent=1, sort_keys=True))
            print(f"  Sectors: {number}/{len(todo)} tickers", flush=True)
        time.sleep(SECTOR_PAUSE_SECONDS)
    known = sum(sectors.get(ticker, "Unknown") != "Unknown" for ticker in tickers)
    return {"requested": len(tickers), "known": known}


def fetch_edgar(
    session: requests.Session, settings: Settings, tickers: Sequence[str]
) -> dict[str, Any]:
    """Save the 8-K history of every ticker that EDGAR can map today."""
    if not settings.has_contact:
        raise RuntimeError("Set CONTACT_NAME and CONTACT_EMAIL in .env")
    headers = {"User-Agent": f"{settings.contact_name} {settings.contact_email}"}
    output_dir = RAW_DIR / "edgar"
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = fetch(session, EDGAR_TICKERS_URL, headers=headers).json()
    cik_by_ticker = {
        to_yahoo_symbol(row["ticker"]): int(row["cik_str"]) for row in mapping.values()
    }
    mapped = [ticker for ticker in tickers if ticker in cik_by_ticker]
    todo = [
        ticker
        for ticker in mapped
        if not (output_dir / f"{safe_name(ticker)}.parquet").exists()
    ]
    for number, ticker in enumerate(todo, start=1):
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik_by_ticker[ticker])
        filings = fetch(session, url, headers=headers).json()["filings"]
        time.sleep(EDGAR_PAUSE_SECONDS)
        blocks = [extract_8k(filings["recent"], ticker)]
        for page in filings.get("files", []):
            older = EDGAR_OLDER_FILE_URL.format(name=page["name"])
            blocks.append(extract_8k(fetch(session, older, headers=headers).json(), ticker))
            time.sleep(EDGAR_PAUSE_SECONDS)
        rows = pd.concat(blocks, ignore_index=True)
        rows.to_parquet(output_dir / f"{safe_name(ticker)}.parquet")
        if number % 50 == 0 or number == len(todo):
            print(f"  EDGAR: {number}/{len(todo)} tickers", flush=True)
    return {"requested": len(tickers), "mapped": len(mapped)}


def resolve_articles(
    session: requests.Session, settings: Settings, candidates: dict[str, str]
) -> dict[str, str]:
    """Map tickers to live Wikipedia articles through the MediaWiki API."""
    headers = {"User-Agent": build_user_agent(settings)}
    tickers = list(candidates)
    articles: dict[str, str] = {}
    for begin in range(0, len(tickers), TITLES_PER_REQUEST):
        batch = tickers[begin : begin + TITLES_PER_REQUEST]
        titles = [candidates[ticker] for ticker in batch]
        url = (
            f"{MEDIAWIKI_API_URL}?action=query&format=json&redirects=1"
            f"&prop=pageprops&ppprop=disambiguation&titles="
            + quote("|".join(titles), safe="|_")
        )
        resolved = resolve_titles(fetch(session, url, headers=headers).json(), titles)
        for ticker, title in zip(batch, titles):
            if title in resolved:
                articles[ticker] = resolved[title]
    return articles


def fetch_wikimedia(
    session: requests.Session, settings: Settings, constituents: pd.DataFrame
) -> dict[str, Any]:
    """Save daily page views. Only names with a known company name are covered."""
    if not settings.has_contact:
        raise RuntimeError("Set CONTACT_NAME and CONTACT_EMAIL in .env")
    candidates = dict(EUROPEAN_ARTICLES)
    for row in constituents.itertuples():
        candidates[row.symbol] = clean_article_title(row.name)
    articles = resolve_articles(session, settings, candidates)
    (RAW_DIR / "articles.json").write_text(json.dumps(articles, indent=1, sort_keys=True))

    output_dir = RAW_DIR / "pageviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": build_user_agent(settings)}
    end = f"{datetime.now(timezone.utc):%Y%m%d}00"
    todo = [
        ticker
        for ticker in articles
        if not (output_dir / f"{safe_name(ticker)}.parquet").exists()
    ]
    for number, ticker in enumerate(todo, start=1):
        url = WIKIMEDIA_URL.format(
            article=quote(articles[ticker], safe=""), start=PAGEVIEWS_START, end=end
        )
        try:
            items = fetch(session, url, headers=headers).json().get("items", [])
        except requests.HTTPError:
            continue
        rows = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [item["timestamp"][:8] for item in items], format="%Y%m%d"
                ),
                "ticker": ticker,
                "pageviews": [float(item["views"]) for item in items],
            }
        )
        rows.to_parquet(output_dir / f"{safe_name(ticker)}.parquet")
        if number % 50 == 0 or number == len(todo):
            print(f"  Wikimedia: {number}/{len(todo)} tickers", flush=True)
    return {"candidates": len(candidates), "resolved": len(articles)}


def merge_prices(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Join fresh rows to saved history.

    A dividend or split rescales all earlier adjusted closes, so the saved
    history is rescaled to match the fresh rows on their last shared day.
    """
    old = with_date_unit(old, "date").sort_values("date")
    new = with_date_unit(new, "date").sort_values("date")
    shared = old.merge(new, on="date", suffixes=("_old", "_new"))
    if not shared.empty:
        last = shared.iloc[-1]
        old = old.assign(adj_close=old["adj_close"] * last["adj_close_new"] / last["adj_close_old"])
    kept = old[~old["date"].isin(new["date"])]
    return pd.concat([kept, new], ignore_index=True).sort_values("date")


def update_prices(
    tickers: Sequence[str],
    output_dir: Path,
    lookback_days: int = UPDATE_LOOKBACK_DAYS,
    chunk_size: int = CHUNK_SIZE,
    pause_seconds: float = CHUNK_PAUSE_SECONDS,
) -> dict[str, Any]:
    """Refresh saved tickers from a few days before their last row."""
    saved = [
        ticker for ticker in tickers if (output_dir / f"{safe_name(ticker)}.parquet").exists()
    ]
    updated = 0
    if not saved:
        return {"tickers": 0, "updated": 0}
    start = pd.read_parquet(output_dir / f"{safe_name(saved[0])}.parquet", columns=["date"])
    begin_date = pd.Timestamp(start["date"].max()) - pd.Timedelta(days=lookback_days)
    for begin in range(0, len(saved), chunk_size):
        chunk = saved[begin : begin + chunk_size]
        frame = yf.download(
            chunk,
            start=str(begin_date.date()),
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
            timeout=30,
            multi_level_index=True,
        )
        for ticker in chunk:
            rows = price_rows(frame, ticker) if frame is not None else pd.DataFrame()
            if rows.empty:
                continue
            path = output_dir / f"{safe_name(ticker)}.parquet"
            merge_prices(pd.read_parquet(path), rows).to_parquet(path)
            updated += 1
        if begin + chunk_size < len(saved):
            time.sleep(pause_seconds)
    return {"tickers": len(saved), "updated": updated}


def update_edgar(
    session: requests.Session, settings: Settings, tickers: Sequence[str]
) -> dict[str, Any]:
    """Add each company's recent filings to its saved history."""
    if not settings.has_contact:
        raise RuntimeError("Set CONTACT_NAME and CONTACT_EMAIL in .env")
    headers = {"User-Agent": f"{settings.contact_name} {settings.contact_email}"}
    output_dir = RAW_DIR / "edgar"
    mapping = fetch(session, EDGAR_TICKERS_URL, headers=headers).json()
    cik_by_ticker = {
        to_yahoo_symbol(row["ticker"]): int(row["cik_str"]) for row in mapping.values()
    }
    updated = 0
    for ticker in tickers:
        path = output_dir / f"{safe_name(ticker)}.parquet"
        if ticker not in cik_by_ticker or not path.exists():
            continue
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik_by_ticker[ticker])
        filings = fetch(session, url, headers=headers).json()["filings"]
        time.sleep(EDGAR_PAUSE_SECONDS)
        rows = pd.concat([pd.read_parquet(path), extract_8k(filings["recent"], ticker)])
        keys = ["filing_date", "accepted_at", "items"]
        rows.drop_duplicates(keys).to_parquet(path)
        updated += 1
    return {"updated": updated}


def update_wikimedia(session: requests.Session, settings: Settings) -> dict[str, Any]:
    """Refresh the last days of page views for known articles."""
    articles_path = RAW_DIR / "articles.json"
    if not settings.has_contact or not articles_path.exists():
        return {"updated": 0}
    articles = json.loads(articles_path.read_text())
    headers = {"User-Agent": build_user_agent(settings)}
    now = datetime.now(timezone.utc)
    start = f"{now - pd.Timedelta(days=UPDATE_LOOKBACK_DAYS):%Y%m%d}00"
    end = f"{now:%Y%m%d}00"
    updated = 0
    for ticker, article in articles.items():
        path = RAW_DIR / "pageviews" / f"{safe_name(ticker)}.parquet"
        if not path.exists():
            continue
        url = WIKIMEDIA_URL.format(article=quote(article, safe=""), start=start, end=end)
        try:
            items = fetch(session, url, headers=headers).json().get("items", [])
        except requests.HTTPError:
            continue
        fresh = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [item["timestamp"][:8] for item in items], format="%Y%m%d"
                ),
                "ticker": ticker,
                "pageviews": [float(item["views"]) for item in items],
            }
        )
        rows = pd.concat([pd.read_parquet(path), fresh], ignore_index=True)
        rows.drop_duplicates(["date"], keep="last").to_parquet(path)
        updated += 1
    return {"updated": updated}


def read_folder(folder: Path, columns: Iterable[str]) -> pd.DataFrame:
    """Read every parquet file in a folder, or return an empty frame."""
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=list(columns))
    frames = [frame for frame in map(pd.read_parquet, files) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def make_panel(session: requests.Session) -> dict[str, Any]:
    """Build data/processed/panel.parquet from the raw files."""
    prices = read_folder(RAW_DIR / "prices", ["date", "ticker", *PRICE_COLUMNS])
    filings = read_folder(
        RAW_DIR / "edgar", ["ticker", "filing_date", "accepted_at", "items"]
    )
    pageviews = read_folder(RAW_DIR / "pageviews", ["date", "ticker", "pageviews"])
    sectors = json.loads((RAW_DIR / "sectors.json").read_text())
    panel = build_panel(
        prices,
        load_membership(session),
        pd.read_parquet(RAW_DIR / "macro.parquet"),
        sectors,
        filings,
        pageviews,
    )
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)
    return {
        "rows": len(panel),
        "tickers": int(panel["ticker"].nunique()),
        "first": str(panel["date"].min().date()),
        "last": str(panel["date"].max().date()),
        "in_sp500_rows": int(panel["in_sp500"].sum()),
        "unknown_sector_tickers": int(
            panel.loc[panel["sector"] == "Unknown", "ticker"].nunique()
        ),
        "rows_with_8k": int(panel["has_8k"].sum()),
        "rows_with_pageviews": int(panel["pageviews"].notna().sum()),
        "size_mb": round(PANEL_PATH.stat().st_size / 1e6, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", default="all", choices=("all", *STEPS))
    args = parser.parse_args(argv)
    # yfinance triggers this pandas deprecation notice on every download.
    warnings.filterwarnings("ignore", message="The inplace keyword")
    steps = STEPS if args.step == "all" else (args.step,)

    settings = load_settings()
    session = requests.Session()
    membership = load_membership(session)
    tickers = stock_tickers(membership)
    results: dict[str, Any] = {}
    for step in steps:
        print(f"Step: {step}", flush=True)
        if step == "membership":
            frozen = membership[membership["date"] <= "2024-12-31"]
            last_frozen = frozen[frozen["date"] == frozen["date"].max()]
            results[step] = {
                "tickers_since_2010": int(membership["ticker"].nunique()),
                "members_at_freeze": len(last_frozen),
                "last_snapshot": str(membership["date"].max().date()),
            }
        elif step == "prices":
            results[step] = fetch_prices(tickers + BENCHMARKS, RAW_DIR / "prices")
        elif step == "macro":
            macro = fetch_macro(session, settings)
            results[step] = {"rows": len(macro), "last": str(macro["date"].max().date())}
        elif step == "sectors":
            results[step] = fetch_sectors(tickers)
        elif step == "edgar":
            results[step] = fetch_edgar(session, settings, sorted(set(membership["ticker"])))
        elif step == "wikimedia":
            constituents = load_constituents(session, settings)
            results[step] = fetch_wikimedia(session, settings, constituents)
        else:
            results[step] = make_panel(session)
        print(f"  {json.dumps(results[step])[:600]}", flush=True)

    report = REPORTS_DIR / "data_build.json"
    report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
