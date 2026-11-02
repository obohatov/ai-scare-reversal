"""Offline tests for the trading simulation, with hand-checked numbers."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import simulate as sm

DATES = pd.bdate_range("2024-01-01", periods=8)


def make_prices(stock: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {"SPY": [100.0] * 8, "AAA": stock or [10.0] * 8, "BBB": [20.0] * 8}, index=DATES
    )


def signal(ticker: str = "AAA", entry: float = 10.0, day: int = 0, **extra) -> dict:
    row = {
        "date": DATES[day],
        "ticker": ticker,
        "entry_open": entry,
        "rel_z": -3.0,
        "score": 0.5,
        "has_8k_2d": 0.0,
        "has_earnings_2d": 0.0,
        "growth_30d": 1.1,
        "exit_date": DATES[-1],
    }
    return {**row, **extra}


def fee_net() -> float:
    return 1 - sm.STOCK_FEE - sm.ETF_FEE


def test_no_signals_means_holding_the_index() -> None:
    frame = pd.DataFrame([signal()]).iloc[0:0]
    equity, stats = sm.exact_equity(frame, make_prices(), sm.Strategy("x", "all_drops"))
    assert equity.iloc[-1] == pytest.approx(100_000 * (1 - sm.ETF_FEE))
    assert stats["trades"] == 0


def test_one_trade_earns_on_a_tenth_of_capital() -> None:
    prices = make_prices([10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
    frame = pd.DataFrame([signal()])
    strategy = sm.Strategy("x", "all_drops", max_positions=10)
    equity, stats = sm.exact_equity(frame, prices, strategy, horizon=3)
    start = 100_000 * (1 - sm.ETF_FEE)
    size = start / 10
    sold = (size * fee_net() / 10.0) * 11.0 * fee_net()
    assert stats["trades"] == 1
    assert equity.iloc[-1] == pytest.approx(start - size + sold)


def test_position_limit_and_duplicate_tickers_are_respected() -> None:
    frame = pd.DataFrame([signal("AAA"), signal("AAA"), signal("BBB", entry=20.0)])
    strategy = sm.Strategy("x", "all_drops", max_positions=1)
    _, stats = sm.exact_equity(frame, make_prices(), strategy, horizon=20)
    assert stats["trades"] == 1


def test_stop_loss_closes_the_position_early() -> None:
    prices = make_prices([10.0, 10.0, 8.0, 12.0, 12.0, 12.0, 12.0, 12.0])
    frame = pd.DataFrame([signal()])
    strategy = sm.Strategy("x", "all_drops", stop_loss=0.15)
    equity, stats = sm.exact_equity(frame, prices, strategy, horizon=5)
    assert stats["stopped_out"] == 1
    assert equity.iloc[-1] == pytest.approx(equity.iloc[2])


def test_entry_uses_the_next_session_not_the_signal_day() -> None:
    prices = make_prices([10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    frame = pd.DataFrame([signal(entry=20.0)])
    equity, _ = sm.exact_equity(frame, prices, sm.Strategy("x", "all_drops"), horizon=3)
    assert equity.iloc[-1] < equity.iloc[0]


def test_missing_price_keeps_the_last_known_value() -> None:
    prices = make_prices()
    prices.loc[DATES[3]:, "AAA"] = float("nan")
    equity, stats = sm.exact_equity(pd.DataFrame([signal()]), prices, sm.Strategy("x", "all_drops"), horizon=3)
    assert stats["trades"] == 1
    assert equity.notna().all()


def test_no_news_rule_rejects_earnings_and_unknown_news() -> None:
    frame = pd.DataFrame(
        [
            signal("AAA"),
            signal("BBB", has_8k_2d=1.0),
            signal("CCC", has_earnings_2d=1.0),
            signal("DDD", has_8k_2d=float("nan")),
        ]
    )
    chosen = sm.select_signals(frame, sm.Strategy("x", "no_news"), 0.0)
    assert chosen["ticker"].tolist() == ["AAA"]


def test_top_score_rule_ranks_best_first() -> None:
    frame = pd.DataFrame(
        [signal("AAA", score=0.6), signal("BBB", score=0.9), signal("CCC", score=0.1)]
    )
    chosen = sm.select_signals(frame, sm.Strategy("x", "top_score"), 0.5)
    assert chosen["ticker"].tolist() == ["BBB", "AAA"]


def test_vector_results_subtract_fees_and_the_index() -> None:
    frame = pd.DataFrame([signal(growth_30d=1.1)])
    prices = pd.Series([100.0] * 7 + [105.0], index=DATES)
    result = sm.vector_results(frame, prices)
    assert result["mean_net_return"] == pytest.approx(0.098)
    assert result["mean_excess_vs_index"] == pytest.approx(0.048)


def test_performance_on_a_known_curve() -> None:
    equity = pd.Series([100.0 * 1.001**day for day in range(253)])
    result = sm.performance(equity)
    assert result["cagr"] == pytest.approx(1.001**252 - 1, abs=1e-4)
    assert result["max_drawdown"] == 0


def test_run_reads_saved_files_end_to_end(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=60)
    panel = pd.concat(
        pd.DataFrame(
            {"date": dates, "ticker": ticker, "adj_close": 100.0 + index,
             "in_sp500": True, "dff": 5.0}
        )
        for index, ticker in enumerate(["SPY", "AAA"])
    )
    panel_path, events_path, scores_path = (
        tmp_path / "panel.parquet", tmp_path / "events.parquet", tmp_path / "scores.parquet",
    )
    panel.to_parquet(panel_path)
    event = signal(date=dates[5], exit_date=dates[40], entry_open=101.0)
    event.update(
        split="validation", in_sp500=True, in_frozen_universe=True, rel_growth_30d=1.02
    )
    pd.DataFrame([event]).drop(columns=["score"]).to_parquet(events_path)
    pd.DataFrame([event])[["date", "ticker", "score"]].to_parquet(scores_path)
    monkeypatch.setattr(sm, "PANEL_PATH", panel_path)
    monkeypatch.setattr(sm, "EVENTS_PATH", events_path)
    monkeypatch.setattr(sm, "PREDICTIONS_PATH", scores_path)
    report = sm.run("validation")
    assert report["events"] == 1
    assert report["strategies"]["no_news"]["exact"]["trades"] == 1


def test_core_position_uses_a_fixed_share_of_equity() -> None:
    prices = make_prices([10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
    strategy = sm.Strategy("x", "all_drops", position_share=0.02)
    equity, _ = sm.exact_equity(pd.DataFrame([signal()]), prices, strategy, horizon=3)
    start = 100_000 * (1 - sm.ETF_FEE)
    size = start * 0.02
    sold = (size * fee_net() / 10.0) * 11.0 * fee_net()
    assert equity.iloc[-1] == pytest.approx(start - size + sold)


def test_spread_short_gains_when_the_price_falls() -> None:
    prices = make_prices([10.0, 10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    shorts = pd.DataFrame([signal()])
    rates = pd.Series(0.0, index=DATES)
    equity, stats = sm.spread_equity(shorts.iloc[0:0], shorts, prices, rates, side_positions=1, horizon=3)
    size = 50_000.0
    fees = size * sm.STOCK_FEE + 0.8 * size * sm.STOCK_FEE
    borrow = 0.8 * size * sm.SHORT_BORROW_RATE / sm.TRADING_DAYS * 2 + size * sm.SHORT_BORROW_RATE / sm.TRADING_DAYS
    assert stats == {"long_trades": 0, "short_trades": 1}
    assert equity.iloc[-1] == pytest.approx(100_000 + 0.2 * size - fees - borrow)


def test_spread_earns_interest_on_idle_cash() -> None:
    frame = pd.DataFrame([signal()]).iloc[0:0]
    rates = pd.Series(25.2, index=DATES)
    equity, _ = sm.spread_equity(frame, frame, make_prices(), rates)
    assert equity.iloc[0] == pytest.approx(100_100.0)


def test_spread_signals_split_news_groups() -> None:
    frame = pd.DataFrame(
        [
            signal("AAA"),
            signal("BBB", has_8k_2d=1.0, has_earnings_2d=1.0),
            signal("CCC", has_8k_2d=1.0),
        ]
    )
    longs, shorts = sm.spread_signals(frame)
    assert longs["ticker"].tolist() == ["AAA"]
    assert shorts["ticker"].tolist() == ["BBB"]


def test_spread_gap_reports_difference_and_error() -> None:
    longs = pd.DataFrame({"rel_growth_30d": [1.02, 1.04]})
    shorts = pd.DataFrame({"rel_growth_30d": [0.98, 1.00]})
    result = sm.spread_gap(longs, shorts)
    assert result["mean_gap"] == pytest.approx(0.04)
    assert result["std_error"] > 0


def test_equal_weight_curve_averages_members_only() -> None:
    dates = DATES[:2]
    panel = pd.DataFrame(
        {
            "date": list(dates) * 3,
            "ticker": ["A", "A", "B", "B", "C", "C"],
            "adj_close": [10.0, 11.0, 10.0, 13.0, 10.0, 20.0],
            "in_sp500": [True, True, True, True, False, False],
        }
    )
    curve = sm.equal_weight_curve(panel, dates)
    assert curve.iloc[-1] == pytest.approx(120_000.0)


def test_cash_curve_compounds_the_rate() -> None:
    curve = sm.cash_curve(pd.Series(25.2, index=DATES[:2]), DATES[:2])
    assert curve.iloc[-1] == pytest.approx(100_100.0)


def test_prices_stop_at_the_last_protocol_date() -> None:
    dates = pd.to_datetime(["2026-10-30", "2026-11-02"])
    panel = pd.DataFrame(
        {"date": list(dates) * 2, "ticker": ["SPY", "SPY", "AAA", "AAA"], "adj_close": 1.0}
    )
    events = pd.DataFrame({"ticker": ["AAA"], "exit_date": [pd.NaT]})
    assert sm.load_prices(events, panel).index.max() == dates[0]


def test_test_period_needs_explicit_confirmation() -> None:
    with pytest.raises(SystemExit):
        sm.main(["--period", "test"])
