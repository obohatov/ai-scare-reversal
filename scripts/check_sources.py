"""Probe every planned data source and write a short readiness report.

Run from the repository root:

    python -m scripts.check_sources --label codespace

The report goes to reports/source_check_<label>.md and .json. It contains no
credentials or contact details, so it is safe to commit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import re
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

from scripts.settings import (
    APP_NAME,
    APP_VERSION,
    REPORTS_DIR,
    Settings,
    load_settings,
)
from scripts.universe import (
    AI_SENSITIVE_US,
    BENCHMARKS,
    EUROPEAN_PEERS,
    FRED_SERIES,
    FULL_UNIVERSE_SIZE,
    SAMPLE_ARTICLES,
    SAMPLE_EARNINGS_TICKERS,
    SAMPLE_US_FILERS,
)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
WIKIMEDIA_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}"
)
TELEGRAM_GET_ME_URL = "https://api.telegram.org/bot{token}/getMe"
ALPACA_ACCOUNT_URL = "https://paper-api.alpaca.markets/v2/account"

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
REQUIRED_SOURCES = ("universe", "yahoo_prices", "fred", "edgar", "wikimedia")
ALL_SOURCES = (
    "host",
    "universe",
    "yahoo_prices",
    "yahoo_earnings",
    "fred",
    "edgar",
    "wikimedia",
    "telegram",
    "alpaca",
)

PRICE_START = "2010-01-01"
PAGEVIEWS_START = "2015070100"
YAHOO_CHUNK_SIZE = 40
YAHOO_CHUNK_PAUSE_SECONDS = 2.0
EDGAR_PAUSE_SECONDS = 0.2
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LISTED_TICKERS = 20
RANDOM_SEED = 42

# (status, one-line summary, machine-readable details)
CheckOutcome = tuple[str, str, dict[str, Any]]


@dataclass
class CheckResult:
    source: str
    status: str
    seconds: float
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class YfinanceLogCapture(logging.Handler):
    """Collect yfinance log lines, which is how it reports many failures."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    @property
    def rate_limited(self) -> bool:
        text = " ".join(self.messages).lower()
        return "rate limit" in text or "too many requests" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def redact(text: str, secrets: Iterable[str]) -> str:
    """Remove secret values and anything shaped like a Telegram bot token."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"\d{6,}:[A-Za-z0-9_-]{20,}", "[redacted]", text)


def sanitise(value: Any, secrets: Iterable[str]) -> Any:
    """Apply redact() to every string inside a nested structure."""
    secrets = tuple(secrets)
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {key: sanitise(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise(item, secrets) for item in value]
    return value


def build_user_agent(settings: Settings) -> str:
    """Identify this client as the Wikimedia policy asks."""
    contact = settings.contact_email or "contact-not-configured"
    return (
        f"{APP_NAME}/{APP_VERSION} ({contact}) "
        f"python-requests/{requests.__version__}"
    )


def fetch(
    session: requests.Session,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 2,
) -> requests.Response:
    """GET a URL, retrying short-lived failures with a growing pause."""
    for attempt in range(retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt == retries:
                raise
        else:
            if response.status_code not in RETRY_STATUS_CODES or attempt == retries:
                response.raise_for_status()
                return response
        time.sleep(2**attempt)
    raise RuntimeError("Unreachable: the retry loop always returns or raises.")


def to_yahoo_symbol(symbol: str) -> str:
    """Yahoo writes share classes with a dash: BRK.B becomes BRK-B."""
    return symbol.strip().replace(".", "-")


def article_from_href(href: Any) -> str | None:
    """Turn '/wiki/Apple_Inc.' into 'Apple_Inc.'; other links give None."""
    if isinstance(href, str) and href.startswith("/wiki/"):
        return href.removeprefix("/wiki/")
    return None


def classify_coverage(found: int, requested: int) -> str:
    """90% or more passes; at least half is usable with a warning."""
    if requested == 0:
        return FAIL
    share = found / requested
    if share >= 0.9:
        return PASS
    return WARN if share >= 0.5 else FAIL


def overall_verdict(results: Sequence[CheckResult]) -> str:
    """All required sources that were run must pass or warn."""
    required = [item for item in results if item.source in REQUIRED_SOURCES]
    if not required:
        return "PARTIAL RUN"
    ready = all(item.status in (PASS, WARN) for item in required)
    if not ready:
        return "NOT READY"
    return "READY" if len(required) == len(REQUIRED_SOURCES) else "PARTIAL RUN"


# ---------------------------------------------------------------------------
# Parsers: pure functions, covered by tests
# ---------------------------------------------------------------------------


def cell_text(cell: Any) -> Any:
    """read_html(extract_links=...) gives (text, link) pairs; keep the text."""
    return cell[0] if isinstance(cell, tuple) else cell


def parse_sp500_tables(html: str) -> tuple[pd.DataFrame, int, str | None]:
    """Return constituents, the number of change rows and the earliest change."""
    tables = pd.read_html(StringIO(html), extract_links="body")
    raw = next((t for t in tables if {"Symbol", "Security"} <= set(t.columns)), None)
    # The page no longer carries a table of index changes, so it is optional.
    changes = next((t for t in tables if "Effective Date" in str(t.columns[0])), None)
    if raw is None:
        raise ValueError("The S&P 500 page no longer has the expected tables.")
    text = raw.map(cell_text)
    links = raw["Security"].map(lambda cell: cell[1] if isinstance(cell, tuple) else None)
    constituents = pd.DataFrame(
        {
            "symbol": text["Symbol"].map(to_yahoo_symbol),
            "name": text["Security"],
            "sector": text["GICS Sector"],
            "sub_industry": text["GICS Sub-Industry"],
            "cik": pd.to_numeric(text["CIK"], errors="coerce"),
            "article": links.map(article_from_href),
        }
    )
    if changes is None:
        return constituents, 0, None
    change_dates = pd.to_datetime(
        changes.iloc[:, 0].map(cell_text), errors="coerce", format="mixed"
    ).dropna()
    earliest = str(change_dates.min().date()) if not change_dates.empty else None
    return constituents, len(changes), earliest


def summarise_prices(
    frame: pd.DataFrame | None, tickers: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Summarise Adjusted Close per ticker from a yf.download() frame."""
    summary: dict[str, dict[str, Any]] = {}
    if frame is None or frame.empty:
        return summary
    for ticker in tickers:
        try:
            series = frame[ticker]["Adj Close"].dropna()
        except KeyError:
            continue
        if series.empty:
            continue
        summary[ticker] = {
            "rows": int(series.size),
            "first": str(series.index.min().date()),
            "last": str(series.index.max().date()),
        }
    return summary


def parse_fred_csv(text: str) -> dict[str, Any]:
    """Summarise a FRED CSV. The first column is the date, the second the value."""
    frame = pd.read_csv(StringIO(text))
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    valid = dates[values.notna() & dates.notna()]
    if valid.empty:
        raise ValueError("The FRED response holds no numeric observations.")
    return {
        "rows": int(valid.size),
        "first": str(valid.min().date()),
        "last": str(valid.max().date()),
    }


def parse_edgar_submissions(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarise what one company's EDGAR submissions file offers."""
    filings = payload["filings"]
    recent = filings["recent"]
    columns = ("form", "filingDate", "acceptanceDateTime", "items")
    frame = pd.DataFrame({name: recent.get(name, []) for name in columns})
    eight_k = frame[frame["form"].isin(["8-K", "8-K/A"])]
    earnings = eight_k[eight_k["items"].fillna("").str.contains("2.02", regex=False)]
    older_pages = filings.get("files", [])
    all_dates = list(frame["filingDate"])
    all_dates += [page["filingFrom"] for page in older_pages if page.get("filingFrom")]
    return {
        "recent_filings": len(frame),
        "recent_8k": len(eight_k),
        "recent_earnings_8k": len(earnings),
        "older_pages": len(older_pages),
        "earliest_filing": min(all_dates) if all_dates else None,
        "has_acceptance_time": bool(frame["acceptanceDateTime"].astype(bool).any()),
    }


def parse_pageviews(payload: dict[str, Any], today: date) -> dict[str, Any]:
    """Summarise a Wikimedia per-article daily response."""
    items = payload.get("items", [])
    if not items:
        raise ValueError("The Wikimedia response holds no daily rows.")
    days = [datetime.strptime(item["timestamp"][:8], "%Y%m%d").date() for item in items]
    return {
        "rows": len(days),
        "first": str(min(days)),
        "last": str(max(days)),
        "lag_days": (today - max(days)).days,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_host() -> CheckOutcome:
    """Record the machine, because memory limits shaped Module 4."""
    try:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, OSError, ValueError):
        memory_gb = None
    free_disk_gb = shutil.disk_usage(".").free / 1e9
    details = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "yfinance": yf.__version__,
        "cpu_count": os.cpu_count(),
        "memory_gb": round(memory_gb, 1) if memory_gb else None,
        "free_disk_gb": round(free_disk_gb, 1),
    }
    status = WARN if memory_gb is not None and memory_gb < 3.5 else PASS
    memory_text = f"{memory_gb:.1f} GB RAM" if memory_gb else "RAM unknown"
    summary = (
        f"Python {details['python']}, {details['cpu_count']} CPUs, "
        f"{memory_text}, {free_disk_gb:.0f} GB free disk"
    )
    return status, summary, details


def check_universe(
    session: requests.Session, user_agent: str
) -> tuple[CheckOutcome, list[str]]:
    """Load the S&P 500 table and test what later stages will need from it."""
    html = fetch(session, WIKI_SP500_URL, headers={"User-Agent": user_agent}).text
    constituents, change_rows, earliest_change = parse_sp500_tables(html)
    count = len(constituents)
    with_cik = int(constituents["cik"].notna().sum())
    with_article = int(constituents["article"].notna().sum())
    details = {
        "constituents": count,
        "with_cik": with_cik,
        "with_article_link": with_article,
        "sectors": int(constituents["sector"].nunique()),
        "change_rows": change_rows,
        "earliest_change": earliest_change,
    }
    status = PASS if count >= 480 and change_rows > 0 else WARN
    changes_text = (
        f"{change_rows} change rows from {earliest_change}"
        if change_rows
        else "no table of index changes on the page"
    )
    summary = (
        f"{count} members, {with_cik} with CIK, {with_article} with article links, "
        f"{changes_text}"
    )
    return (status, summary, details), constituents["symbol"].tolist()


def build_yahoo_sample(symbols: Sequence[str], sample_size: int) -> list[str]:
    """Mix named AI-sensitive groups, peers, benchmarks and random members."""
    named = list(dict.fromkeys(AI_SENSITIVE_US + EUROPEAN_PEERS + BENCHMARKS))
    others = [symbol for symbol in symbols if symbol not in named]
    room = max(sample_size - len(named), 0)
    extras = random.Random(RANDOM_SEED).sample(others, min(room, len(others)))
    return named + extras


def check_yahoo_prices(
    tickers: Sequence[str],
    chunk_size: int = YAHOO_CHUNK_SIZE,
    pause_seconds: float = YAHOO_CHUNK_PAUSE_SECONDS,
) -> CheckOutcome:
    """Download daily history in chunks and project the full download."""
    capture = YfinanceLogCapture()
    logger = logging.getLogger("yfinance")
    logger.addHandler(capture)
    logger.propagate = False
    summary_by_ticker: dict[str, dict[str, Any]] = {}
    chunk_errors = 0
    started = time.perf_counter()
    try:
        for start in range(0, len(tickers), chunk_size):
            chunk = list(tickers[start : start + chunk_size])
            try:
                frame = yf.download(
                    chunk,
                    start=PRICE_START,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                    timeout=30,
                )
            except Exception as error:  # yfinance raises several private types
                chunk_errors += 1
                capture.messages.append(str(error))
                continue
            summary_by_ticker.update(summarise_prices(frame, chunk))
            if start + chunk_size < len(tickers):
                time.sleep(pause_seconds)
    finally:
        logger.removeHandler(capture)
    elapsed = time.perf_counter() - started

    requested = len(tickers)
    found = len(summary_by_ticker)
    rows = sum(item["rows"] for item in summary_by_ticker.values())
    missing = [ticker for ticker in tickers if ticker not in summary_by_ticker]
    seconds_per_ticker = elapsed / requested if requested else 0
    details = {
        "requested": requested,
        "with_data": found,
        "rows": rows,
        "missing": missing[:MAX_LISTED_TICKERS],
        "missing_total": len(missing),
        "chunk_errors": chunk_errors,
        "rate_limited": capture.rate_limited,
        "seconds_per_ticker": round(seconds_per_ticker, 3),
        "estimated_full_minutes": round(seconds_per_ticker * FULL_UNIVERSE_SIZE / 60, 1),
        "estimated_full_rows": int(rows / found * FULL_UNIVERSE_SIZE) if found else 0,
        "earliest_first_date": min(
            (item["first"] for item in summary_by_ticker.values()), default=None
        ),
        "latest_last_date": max(
            (item["last"] for item in summary_by_ticker.values()), default=None
        ),
    }
    status = classify_coverage(found, requested)
    if capture.rate_limited and status == PASS:
        status = WARN
    rate_note = ", rate limit seen" if capture.rate_limited else ""
    summary = (
        f"{found}/{requested} tickers, {rows:,} rows in {elapsed:.0f}s; "
        f"projected {details['estimated_full_rows']:,} rows in "
        f"{details['estimated_full_minutes']} min{rate_note}"
    )
    return status, summary, details


def check_yahoo_earnings(tickers: Sequence[str]) -> CheckOutcome:
    """See how far back Yahoo earnings dates go; EDGAR is the fallback."""
    per_ticker: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        try:
            frame = yf.Ticker(ticker).get_earnings_dates(limit=100)
        except Exception as error:  # yfinance raises several private types
            per_ticker[ticker] = {"error": type(error).__name__}
            continue
        if frame is None or frame.empty:
            per_ticker[ticker] = {"rows": 0}
            continue
        per_ticker[ticker] = {
            "rows": len(frame),
            "earliest": str(frame.index.min().date()),
        }
    with_rows = sum(1 for item in per_ticker.values() if item.get("rows"))
    status = classify_coverage(with_rows, len(tickers))
    earliest = min(
        (item["earliest"] for item in per_ticker.values() if "earliest" in item),
        default=None,
    )
    summary = f"{with_rows}/{len(tickers)} tickers, earliest date {earliest}"
    return status, summary, {"tickers": per_ticker}


def check_fred(session: requests.Session, user_agent: str) -> CheckOutcome:
    """Load a few macro series through FRED's public CSV endpoint."""
    series: dict[str, dict[str, Any]] = {}
    for name in FRED_SERIES:
        url = FRED_CSV_URL.format(series=name)
        response = fetch(session, url, headers={"User-Agent": user_agent})
        series[name] = parse_fred_csv(response.text)
    latest = min(item["last"] for item in series.values())
    summary = f"{len(series)} series, all current to at least {latest}"
    return PASS, summary, {"series": series}


def check_edgar(session: requests.Session, settings: Settings) -> CheckOutcome:
    """Map tickers to CIKs and inspect sample filing histories."""
    if not settings.has_contact:
        return SKIP, "Set CONTACT_NAME and CONTACT_EMAIL in .env", {}
    headers = {"User-Agent": f"{settings.contact_name} {settings.contact_email}"}
    mapping = fetch(session, EDGAR_TICKERS_URL, headers=headers).json()
    cik_by_ticker = {row["ticker"]: int(row["cik_str"]) for row in mapping.values()}
    filers: dict[str, dict[str, Any]] = {}
    for ticker in SAMPLE_US_FILERS:
        if ticker not in cik_by_ticker:
            filers[ticker] = {"error": "ticker not in the EDGAR map"}
            continue
        time.sleep(EDGAR_PAUSE_SECONDS)
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik_by_ticker[ticker])
        filers[ticker] = parse_edgar_submissions(fetch(session, url, headers=headers).json())
    usable = [item for item in filers.values() if "error" not in item]
    status = classify_coverage(len(usable), len(SAMPLE_US_FILERS))
    earliest = min((item["earliest_filing"] for item in usable if item["earliest_filing"]), default=None)
    summary = (
        f"{len(cik_by_ticker):,} tickers mapped; {len(usable)}/{len(SAMPLE_US_FILERS)} "
        f"filers read, earliest filing {earliest}"
    )
    return status, summary, {"mapped_tickers": len(cik_by_ticker), "filers": filers}


def check_wikimedia(session: requests.Session, settings: Settings) -> CheckOutcome:
    """Read daily page views for sample company articles, one at a time."""
    if not settings.has_contact:
        return SKIP, "Set CONTACT_NAME and CONTACT_EMAIL in .env", {}
    headers = {"User-Agent": build_user_agent(settings)}
    today = datetime.now(timezone.utc).date()
    end = f"{today:%Y%m%d}00"
    articles: dict[str, dict[str, Any]] = {}
    for article in SAMPLE_ARTICLES:
        url = WIKIMEDIA_URL.format(
            article=quote(article, safe=""), start=PAGEVIEWS_START, end=end
        )
        articles[article] = parse_pageviews(fetch(session, url, headers=headers).json(), today)
    earliest = min(item["first"] for item in articles.values())
    worst_lag = max(item["lag_days"] for item in articles.values())
    summary = (
        f"{len(articles)} articles, data from {earliest}, "
        f"published with up to {worst_lag} day(s) of lag"
    )
    return PASS, summary, {"articles": articles}


def check_telegram(session: requests.Session, settings: Settings) -> CheckOutcome:
    """Optional: confirm the bot token works."""
    if not settings.telegram_bot_token:
        return SKIP, "Optional: TELEGRAM_BOT_TOKEN is empty", {}
    url = TELEGRAM_GET_ME_URL.format(token=settings.telegram_bot_token)
    is_ok = bool(fetch(session, url).json().get("ok"))
    return (PASS, "Bot token accepted", {}) if is_ok else (FAIL, "Bot token refused", {})


def check_alpaca(session: requests.Session, settings: Settings) -> CheckOutcome:
    """Optional: confirm the paper account can be reached."""
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        return SKIP, "Optional: Alpaca keys are empty", {}
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    account = fetch(session, ALPACA_ACCOUNT_URL, headers=headers).json()
    status_text = str(account.get("status", "unknown"))
    status = PASS if status_text == "ACTIVE" else WARN
    return status, f"Paper account status: {status_text}", {"status": status_text}


# ---------------------------------------------------------------------------
# Running and reporting
# ---------------------------------------------------------------------------


def run_check(
    source: str, check: Callable[[], CheckOutcome], secrets: Sequence[str]
) -> CheckResult:
    """Run one check so that a failure in it cannot stop the others."""
    started = time.perf_counter()
    try:
        status, summary, details = check()
    except Exception as error:  # every failure must end up in the report
        status = FAIL
        summary = f"{type(error).__name__}: {error}"[:300]
        details = {}
    seconds = round(time.perf_counter() - started, 1)
    return CheckResult(
        source=source,
        status=status,
        seconds=seconds,
        summary=redact(summary, secrets),
        details=sanitise(details, secrets),
    )


def render_markdown(results: Sequence[CheckResult], label: str, run_at: str) -> str:
    """Render the report for reading or pasting into a conversation."""
    lines = [
        f"# Source check: {label}",
        "",
        f"Run at {run_at}. Verdict: **{overall_verdict(results)}**.",
        "",
        "| Source | Status | Seconds | Summary |",
        "|---|---|---|---|",
    ]
    for item in results:
        cell = item.summary.replace("|", "/")
        lines.append(f"| {item.source} | {item.status} | {item.seconds} | {cell} |")
    return "\n".join(lines) + "\n"


def write_report(
    results: Sequence[CheckResult],
    label: str,
    output_dir: Path,
    secrets: Sequence[str],
) -> tuple[Path, Path]:
    """Write the Markdown and JSON reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label)
    markdown_path = output_dir / f"source_check_{safe_label}.md"
    json_path = output_dir / f"source_check_{safe_label}.json"
    markdown_path.write_text(
        redact(render_markdown(results, label, run_at), secrets), encoding="utf-8"
    )
    payload = {
        "label": label,
        "run_at": run_at,
        "verdict": overall_verdict(results),
        "results": [asdict(item) for item in results],
    }
    json_path.write_text(
        json.dumps(sanitise(payload, secrets), indent=2) + "\n", encoding="utf-8"
    )
    return markdown_path, json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--label", default="local", help="Name of this machine in the report."
    )
    parser.add_argument(
        "--only",
        default="",
        help=f"Comma-separated subset of: {', '.join(ALL_SOURCES)}.",
    )
    parser.add_argument(
        "--yahoo-sample",
        type=int,
        default=80,
        help=(
            "Target ticker count for the Yahoo test (default: 80). Named groups "
            "are always included; random S&P 500 members need the universe check."
        ),
    )
    args = parser.parse_args(argv)
    args.selected = [item for item in args.only.split(",") if item] or list(ALL_SOURCES)
    unknown = sorted(set(args.selected) - set(ALL_SOURCES))
    if unknown:
        parser.error(f"Unknown sources: {', '.join(unknown)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # yfinance triggers this pandas deprecation notice on every download.
    warnings.filterwarnings("ignore", message="The inplace keyword")
    settings = load_settings()
    session = requests.Session()
    user_agent = build_user_agent(settings)
    secrets = settings.secrets
    symbols: list[str] = []

    def universe_outcome() -> CheckOutcome:
        nonlocal symbols
        outcome, symbols = check_universe(session, user_agent)
        return outcome

    checks: dict[str, Callable[[], CheckOutcome]] = {
        "host": check_host,
        "universe": universe_outcome,
        "yahoo_prices": lambda: check_yahoo_prices(
            build_yahoo_sample(symbols, args.yahoo_sample)
        ),
        "yahoo_earnings": lambda: check_yahoo_earnings(SAMPLE_EARNINGS_TICKERS),
        "fred": lambda: check_fred(session, user_agent),
        "edgar": lambda: check_edgar(session, settings),
        "wikimedia": lambda: check_wikimedia(session, settings),
        "telegram": lambda: check_telegram(session, settings),
        "alpaca": lambda: check_alpaca(session, settings),
    }

    results: list[CheckResult] = []
    for source in ALL_SOURCES:
        if source not in args.selected:
            continue
        print(f"Checking {source}...", flush=True)
        result = run_check(source, checks[source], secrets)
        results.append(result)
        print(f"  {result.status}: {result.summary}", flush=True)

    markdown_path, json_path = write_report(results, args.label, REPORTS_DIR, secrets)
    verdict = overall_verdict(results)
    print(f"\nVerdict: {verdict}")
    print(f"Report: {markdown_path}")
    print(f"Details: {json_path}")
    return 1 if verdict == "NOT READY" else 0


if __name__ == "__main__":
    sys.exit(main())
