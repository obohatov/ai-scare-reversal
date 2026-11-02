"""Offline tests for the data build. No test touches the network."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import data_repo as dr

MEMBERSHIP_CSV = (
    "date,tickers\n"
    '2009-06-01,"OLD,BRK.B"\n'
    '2009-12-01,"AAA,BRK.B"\n'
    '2024-06-03,"AAA,BBB"\n'
)


def yahoo_frame(tickers: list[str]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=3, freq="B", name="Date")
    names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([tickers, names])
    return pd.DataFrame(1.0, index=index, columns=columns)


def test_parse_membership_keeps_last_snapshot_before_start() -> None:
    membership = dr.parse_membership(MEMBERSHIP_CSV, start="2010-01-01")
    assert sorted(membership["ticker"].unique()) == ["AAA", "BBB", "BRK-B"]
    assert membership["date"].min() == pd.Timestamp("2009-12-01")


def test_membership_flags_show_entry_and_exit() -> None:
    membership = dr.parse_membership(MEMBERSHIP_CSV, start="2010-01-01")
    dates = pd.DatetimeIndex(["2010-01-04", "2024-06-03", "2026-01-05"])
    flags = dr.membership_flags(membership, dates).set_index(["date", "ticker"])
    assert flags.loc[("2010-01-04", "BRK-B"), "in_sp500"]
    assert not flags.loc[("2024-06-03", "BRK-B"), "in_sp500"]
    assert not flags.loc[("2010-01-04", "BBB"), "in_sp500"]
    assert flags.loc[("2026-01-05", "BBB"), "in_sp500"]


def test_price_rows_returns_long_columns_and_skips_absent() -> None:
    frame = yahoo_frame(["AAA"])
    rows = dr.price_rows(frame, "AAA")
    assert list(rows.columns) == ["date", "ticker", *dr.PRICE_COLUMNS]
    assert len(rows) == 3
    assert dr.price_rows(frame, "ZZZ").empty


def test_extract_8k_keeps_only_8k_forms() -> None:
    columns = {
        "form": ["8-K", "10-Q", "8-K/A"],
        "filingDate": ["2026-01-08", "2026-01-02", "2025-01-02"],
        "acceptanceDateTime": ["2026-01-08T16:05:00.000Z", "", ""],
        "items": ["2.02,9.01", "", "5.02"],
    }
    rows = dr.extract_8k(columns, "AAA")
    assert rows["items"].tolist() == ["2.02,9.01", "5.02"]
    assert set(rows["ticker"]) == {"AAA"}


@pytest.mark.parametrize(
    ("security", "expected"),
    [("Alphabet Inc. (Class A)", "Alphabet_Inc."), ("3M", "3M"), ("Eli Lilly", "Eli_Lilly")],
)
def test_clean_article_title(security: str, expected: str) -> None:
    assert dr.clean_article_title(security) == expected


def test_resolve_titles_follows_redirects_and_drops_bad_pages() -> None:
    payload = {
        "query": {
            "normalized": [{"from": "Apple_Inc.", "to": "Apple Inc."}],
            "redirects": [{"from": "Old Name", "to": "New Name"}],
            "pages": {
                "1": {"title": "Apple Inc."},
                "2": {"title": "New Name"},
                "-1": {"title": "Gone", "missing": ""},
                "3": {"title": "Ambiguous", "pageprops": {"disambiguation": ""}},
            },
        }
    }
    titles = ["Apple_Inc.", "Old Name", "Gone", "Ambiguous"]
    assert dr.resolve_titles(payload, titles) == {
        "Apple_Inc.": "Apple_Inc.",
        "Old Name": "New_Name",
    }


def test_news_flags_move_after_close_filings_to_next_session() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-12"]),
            "ticker": "AAA",
        }
    )
    filings = pd.DataFrame(
        {
            "ticker": "AAA",
            "filing_date": pd.to_datetime(["2026-01-08", "2026-01-09"]),
            "accepted_at": ["2026-01-08T09:00:00.000Z", "2026-01-09T16:30:00.000Z"],
            "items": ["5.02", "2.02"],
        }
    )
    flags = dr.news_flags(filings, calendar).sort_values("date")
    assert flags["date"].tolist() == list(pd.to_datetime(["2026-01-08", "2026-01-12"]))
    assert flags["has_earnings_8k"].tolist() == [False, True]


def test_news_flags_accepts_no_filings() -> None:
    calendar = pd.DataFrame({"date": pd.to_datetime(["2026-01-08"]), "ticker": "AAA"})
    empty = pd.DataFrame(columns=["ticker", "filing_date", "accepted_at", "items"])
    assert dr.news_flags(empty, calendar).empty


def make_inputs() -> dict:
    dates = pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"])
    prices = pd.concat(
        pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                **{name: 1.0 for name in dr.PRICE_COLUMNS},
            }
        )
        for ticker in ("AAA", "REL.L")
    )
    return {
        "prices": prices,
        "membership": dr.parse_membership(MEMBERSHIP_CSV, start="2010-01-01"),
        "macro": pd.DataFrame(
            {"date": pd.to_datetime(["2024-06-03", "2024-06-04"]), "dgs10": [4.0, 5.0]}
        ),
        "sectors": {"AAA": "Technology"},
        "filings": pd.DataFrame(
            {
                "ticker": ["AAA"],
                "filing_date": pd.to_datetime(["2024-06-04"]),
                "accepted_at": ["2024-06-04T09:00:00.000Z"],
                "items": ["2.02"],
            }
        ),
        "pageviews": pd.DataFrame(
            {"date": dates[:1], "ticker": ["AAA"], "pageviews": [10.0]}
        ),
    }


def test_build_panel_combines_sources_without_duplicates() -> None:
    panel = dr.build_panel(**make_inputs())
    assert len(panel) == 6
    assert not panel.duplicated(["date", "ticker"]).any()
    assert panel.loc[panel["ticker"] == "AAA", "in_sp500"].all()
    assert not panel.loc[panel["ticker"] == "REL.L", "in_sp500"].any()
    assert set(panel.loc[panel["ticker"] == "REL.L", "sector"]) == {"Unknown"}
    assert panel["has_8k"].sum() == 1
    assert panel["pageviews"].notna().sum() == 1


def test_build_panel_makes_macro_available_one_business_day_later() -> None:
    panel = dr.build_panel(**make_inputs())
    rows = panel[panel["ticker"] == "AAA"].set_index("date")["dgs10"]
    assert pd.isna(rows.loc["2024-06-03"])
    assert rows.loc["2024-06-04"] == 4.0
    assert rows.loc["2024-06-05"] == 5.0


def test_fetch_prices_resumes_and_remembers_empty_tickers(monkeypatch, tmp_path) -> None:
    requested: list[list[str]] = []

    def fake_download(chunk, **kwargs):
        requested.append(list(chunk))
        return yahoo_frame([ticker for ticker in chunk if ticker != "DEAD"])

    monkeypatch.setattr(dr.yf, "download", fake_download)
    output = tmp_path / "prices"
    first = dr.fetch_prices(["AAA", "DEAD"], output, pause_seconds=0, retry_pause_seconds=0)
    second = dr.fetch_prices(
        ["AAA", "BBB", "DEAD"], output, pause_seconds=0, retry_pause_seconds=0
    )
    assert first["saved"] == 1
    assert first["no_data"] == ["DEAD"]
    assert requested[-1] == ["BBB"]
    assert second["saved"] == 2


def test_fetch_prices_retries_an_empty_chunk_and_does_not_blacklist(
    monkeypatch, tmp_path
) -> None:
    calls = {"count": 0}

    def fake_download(chunk, **kwargs):
        calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(dr.yf, "download", fake_download)
    result = dr.fetch_prices(
        ["AAA"], tmp_path / "prices", pause_seconds=0, retry_pause_seconds=0
    )
    assert calls["count"] == 2
    assert result["no_data"] == []
    assert result["saved"] == 0
