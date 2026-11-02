"""Offline tests for the source check. No test touches the network."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
import requests

from scripts import check_sources as cs
from scripts.settings import Settings

SECRET_EMAIL = "someone@example.org"
SECRET_TOKEN = "123456789:AAAbbbCCCdddEEEfffGGGhhhIIIjjj"

WIKI_HTML = """
<html><body>
<table id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
<th>GICS Sub-Industry</th><th>CIK</th></tr>
<tr><td><a href="https://www.nyse.com/quote/XNYS:BRK.B">BRK.B</a></td>
<td><a href="/wiki/Berkshire_Hathaway">Berkshire Hathaway</a></td>
<td>Financials</td><td>Multi-Sector Holdings</td><td>0001067983</td></tr>
<tr><td><a href="https://www.nasdaq.com/market-activity/stocks/aapl">AAPL</a></td>
<td><a href="/wiki/Apple_Inc.">Apple Inc.</a></td>
<td>Information Technology</td><td>Technology Hardware</td><td>0000320193</td></tr>
</table>
<table id="changes">
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th>
<th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>December 22, 2025</td><td>AAA</td><td>Alpha</td><td>BBB</td>
<td>Beta</td><td>Size</td></tr>
<tr><td>March 3, 1997</td><td>CCC</td><td>Gamma</td><td>DDD</td>
<td>Delta</td><td>Merger</td></tr>
</table>
</body></html>
"""


def make_price_frame(tickers: list[str]) -> pd.DataFrame:
    """Build a frame shaped like yf.download(..., group_by='ticker')."""
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    columns = pd.MultiIndex.from_product([tickers, ["Close", "Adj Close"]])
    return pd.DataFrame(1.0, index=index, columns=columns)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.text = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def get(self, url: str, headers=None, timeout=None) -> FakeResponse:
        self.calls += 1
        return self.responses.pop(0)


def test_to_yahoo_symbol_converts_share_classes() -> None:
    assert cs.to_yahoo_symbol(" BRK.B ") == "BRK-B"


def test_parse_sp500_tables_reads_members_links_and_changes() -> None:
    members, change_rows, earliest = cs.parse_sp500_tables(WIKI_HTML)
    assert members["symbol"].tolist() == ["BRK-B", "AAPL"]
    assert members["article"].tolist() == ["Berkshire_Hathaway", "Apple_Inc."]
    assert members["cik"].tolist() == [1067983, 320193]
    assert change_rows == 2
    assert earliest == "1997-03-03"


def test_summarise_prices_skips_absent_and_empty_tickers() -> None:
    frame = make_price_frame(["AAA", "EMPTY"])
    frame.loc[:, ("EMPTY", "Adj Close")] = float("nan")
    summary = cs.summarise_prices(frame, ["AAA", "EMPTY", "ABSENT"])
    assert summary == {"AAA": {"rows": 3, "first": "2024-01-02", "last": "2024-01-04"}}
    assert cs.summarise_prices(None, ["AAA"]) == {}


@pytest.mark.parametrize("date_column", ["observation_date", "DATE"])
def test_parse_fred_csv_handles_both_headers_and_missing_values(date_column: str) -> None:
    text = f"{date_column},DGS10\n2024-01-01,\n2024-01-02,3.95\n2024-01-03,3.91\n"
    assert cs.parse_fred_csv(text) == {
        "rows": 2,
        "first": "2024-01-02",
        "last": "2024-01-03",
    }


def test_parse_fred_csv_rejects_a_response_without_numbers() -> None:
    with pytest.raises(ValueError):
        cs.parse_fred_csv("DATE,DGS10\n2024-01-01,.\n")


def test_parse_edgar_submissions_counts_earnings_filings() -> None:
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "8-K"],
                "filingDate": ["2026-01-08", "2025-12-30", "2025-10-02"],
                "acceptanceDateTime": ["2026-01-08T16:05:00.000Z", "", ""],
                "items": ["2.02,9.01", "", "5.02"],
            },
            "files": [{"filingFrom": "2001-03-01"}],
        }
    }
    assert cs.parse_edgar_submissions(payload) == {
        "recent_filings": 3,
        "recent_8k": 2,
        "recent_earnings_8k": 1,
        "older_pages": 1,
        "earliest_filing": "2001-03-01",
        "has_acceptance_time": True,
    }


def test_parse_pageviews_reports_range_and_lag() -> None:
    payload = {"items": [{"timestamp": "2015070100"}, {"timestamp": "2026110100"}]}
    summary = cs.parse_pageviews(payload, date(2026, 11, 2))
    assert summary == {"rows": 2, "first": "2015-07-01", "last": "2026-11-01", "lag_days": 1}


def test_parse_pageviews_rejects_empty_response() -> None:
    with pytest.raises(ValueError):
        cs.parse_pageviews({"items": []}, date(2026, 11, 2))


@pytest.mark.parametrize(
    ("found", "requested", "expected"),
    [(9, 10, cs.PASS), (5, 10, cs.WARN), (4, 10, cs.FAIL), (0, 0, cs.FAIL)],
)
def test_classify_coverage(found: int, requested: int, expected: str) -> None:
    assert cs.classify_coverage(found, requested) == expected


def test_build_yahoo_sample_keeps_named_groups_and_is_repeatable() -> None:
    symbols = [f"T{number}" for number in range(100)]
    first = cs.build_yahoo_sample(symbols, 80)
    assert first == cs.build_yahoo_sample(symbols, 80)
    assert len(first) == 80
    assert "WKL.AS" in first
    assert "^GSPC" in first
    assert len(set(first)) == len(first)


def test_check_yahoo_prices_lists_missing_tickers(monkeypatch) -> None:
    def fake_download(chunk, **kwargs):
        return make_price_frame([ticker for ticker in chunk if ticker != "BAD"])

    monkeypatch.setattr(cs.yf, "download", fake_download)
    status, summary, details = cs.check_yahoo_prices(
        ["AAA", "BBB", "BAD"], chunk_size=2, pause_seconds=0
    )
    assert status == cs.WARN
    assert details["with_data"] == 2
    assert details["missing"] == ["BAD"]
    assert details["estimated_full_rows"] == 3 * cs.FULL_UNIVERSE_SIZE
    assert "2/3 tickers" in summary


def test_check_yahoo_prices_turns_a_rate_limit_into_a_warning(monkeypatch) -> None:
    def fake_download(chunk, **kwargs):
        cs.logging.getLogger("yfinance").error("Too Many Requests. Rate limited.")
        return make_price_frame(list(chunk))

    monkeypatch.setattr(cs.yf, "download", fake_download)
    status, summary, details = cs.check_yahoo_prices(["AAA", "BBB"], pause_seconds=0)
    assert status == cs.WARN
    assert details["rate_limited"] is True
    assert "rate limit seen" in summary


def test_check_yahoo_prices_survives_a_failing_chunk(monkeypatch) -> None:
    def fake_download(chunk, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cs.yf, "download", fake_download)
    status, _, details = cs.check_yahoo_prices(["AAA", "BBB"], pause_seconds=0)
    assert status == cs.FAIL
    assert details["chunk_errors"] == 1


def test_fetch_retries_a_rate_limit_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(cs.time, "sleep", lambda seconds: None)
    session = FakeSession([FakeResponse(429), FakeResponse(200, {"ok": True})])
    assert cs.fetch(session, "https://example.org").json() == {"ok": True}
    assert session.calls == 2


def test_fetch_does_not_retry_a_permanent_error(monkeypatch) -> None:
    monkeypatch.setattr(cs.time, "sleep", lambda seconds: None)
    session = FakeSession([FakeResponse(403)])
    with pytest.raises(requests.HTTPError):
        cs.fetch(session, "https://example.org")
    assert session.calls == 1


def test_contact_checks_skip_without_contact_details() -> None:
    session = FakeSession([])
    assert cs.check_edgar(session, Settings())[0] == cs.SKIP
    assert cs.check_wikimedia(session, Settings())[0] == cs.SKIP
    assert session.calls == 0


def test_optional_checks_skip_without_tokens() -> None:
    session = FakeSession([])
    assert cs.check_telegram(session, Settings())[0] == cs.SKIP
    assert cs.check_alpaca(session, Settings())[0] == cs.SKIP


def test_run_check_reports_a_failure_without_leaking_secrets() -> None:
    def failing_check():
        url = f"https://api.telegram.org/bot{SECRET_TOKEN}/getMe for {SECRET_EMAIL}"
        raise requests.ConnectionError(url)

    result = cs.run_check("telegram", failing_check, (SECRET_EMAIL, SECRET_TOKEN))
    assert result.status == cs.FAIL
    assert SECRET_TOKEN not in result.summary
    assert SECRET_EMAIL not in result.summary


def test_redact_removes_a_token_even_when_it_is_not_listed() -> None:
    assert SECRET_TOKEN not in cs.redact(f"bot{SECRET_TOKEN}/getMe", ())


def result(source: str, status: str) -> cs.CheckResult:
    return cs.CheckResult(source, status, 0.0, "summary")


def test_verdict_is_ready_only_when_every_required_source_was_usable() -> None:
    ready = [result(source, cs.PASS) for source in cs.REQUIRED_SOURCES]
    assert cs.overall_verdict(ready) == "READY"
    ready[0] = result(ready[0].source, cs.WARN)
    assert cs.overall_verdict(ready) == "READY"


def test_verdict_is_not_ready_when_a_required_source_fails_or_skips() -> None:
    failed = [result(source, cs.PASS) for source in cs.REQUIRED_SOURCES]
    failed[1] = result(failed[1].source, cs.FAIL)
    assert cs.overall_verdict(failed) == "NOT READY"
    skipped = [result(source, cs.PASS) for source in cs.REQUIRED_SOURCES]
    skipped[3] = result(skipped[3].source, cs.SKIP)
    assert cs.overall_verdict(skipped) == "NOT READY"


def test_verdict_marks_a_subset_as_partial() -> None:
    assert cs.overall_verdict([result("fred", cs.PASS)]) == "PARTIAL RUN"
    assert cs.overall_verdict([result("host", cs.PASS)]) == "PARTIAL RUN"


def test_write_report_creates_both_files_without_secrets(tmp_path) -> None:
    item = cs.CheckResult(
        "edgar", cs.PASS, 1.2, f"Mapped | for {SECRET_EMAIL}", {"note": SECRET_EMAIL}
    )
    markdown_path, json_path = cs.write_report(
        [item], "code space", tmp_path, (SECRET_EMAIL,)
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert markdown_path.name == "source_check_code-space.md"
    assert SECRET_EMAIL not in markdown + json.dumps(payload)
    assert "| edgar | PASS | 1.2 |" in markdown
    assert payload["results"][0]["source"] == "edgar"


def test_parse_sp500_tables_accepts_a_page_without_changes() -> None:
    html = WIKI_HTML.split('<table id="changes">')[0] + "</body></html>"
    members, change_rows, earliest = cs.parse_sp500_tables(html)
    assert len(members) == 2
    assert (change_rows, earliest) == (0, None)


def test_parse_sp500_tables_explains_a_changed_page() -> None:
    with pytest.raises(ValueError, match="expected tables"):
        cs.parse_sp500_tables("<table><tr><th>Other</th></tr><tr><td>1</td></tr></table>")


def test_main_passes_universe_symbols_to_the_yahoo_check(monkeypatch, tmp_path) -> None:
    seen: dict[str, list[str]] = {}

    def fake_universe(session, user_agent):
        return (cs.PASS, "1 member", {}), ["ZZZZ"]

    def fake_yahoo(tickers):
        seen["tickers"] = list(tickers)
        return cs.PASS, "ok", {}

    monkeypatch.setattr(cs, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(cs, "load_settings", Settings)
    monkeypatch.setattr(cs, "check_universe", fake_universe)
    monkeypatch.setattr(cs, "check_yahoo_prices", fake_yahoo)
    exit_code = cs.main(["--label", "test", "--only", "universe,yahoo_prices"])
    assert exit_code == 0
    assert "ZZZZ" in seen["tickers"]
    assert (tmp_path / "source_check_test.md").exists()


def test_main_returns_an_error_when_a_required_source_fails(monkeypatch, tmp_path) -> None:
    def failing_fred(session, user_agent):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(cs, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(cs, "load_settings", Settings)
    monkeypatch.setattr(cs, "check_fred", failing_fred)
    assert cs.main(["--label", "test", "--only", "fred"]) == 1


def test_parse_args_rejects_an_unknown_source() -> None:
    with pytest.raises(SystemExit):
        cs.parse_args(["--only", "yahoo"])


def test_parse_args_accepts_a_subset() -> None:
    assert cs.parse_args(["--only", "yahoo_prices,fred"]).selected == [
        "yahoo_prices",
        "fred",
    ]
