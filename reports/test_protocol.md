# Test protocol

Fixed on 2026-11-02, before any test-period outcome was computed. The test period is opened once. No strategy, threshold, feature or model changes after that.

## Period and universe

- Signals from 2025-01-03 to 2026-10-30. Prices end on 2026-10-30, because later rows may come from an unfinished session.
- Universe: S&P 500 members at the end of 2024. European peers and event extras are excluded from the strategy test because they were picked after the AI scares.
- Open positions at the end are valued at their last price.

## Strategies

- Primary: `no_news`, 10 positions, chosen from the theory in Savor (2012), not from its validation result.
- Control: `all_drops`.
- Also reported, unselected: `no_news_5_positions`, `no_news_stop_15`, `no_news_core_20`, `top_score`, `news_spread`.

The best test result among these must not be presented as the strategy.

## Benchmarks

SPY, the equal-weight index of S&P 500 members and cash at the Fed Funds rate.

## Model

The chosen Decision Tree showed no signal on validation (AUC 0.509). `top_score` does not hold the top 10% of events, because tree scores tie.

## AI events

- Inputs: the 11 labels in `events/ai_events.csv`, committed on 2026-11-02 before outcomes were viewed.
- Primary measure: `rel_growth_30d`, from the next open to the close 30 sessions later, against the sector.
- Secondary measure: the same at 90 sessions.
- All listed tickers count, whether or not they passed the sharp-fall rule.
- Each event counts once, as the mean of its stocks. Event means are compared by `substitutes_paid_workflow`.
- Comparison group: sharp falls of index members in the test period, outside the AI lists.
- E01 lies before the test period and is reported with the others, marked as such.
- With 11 overlapping events selected on their fall, results are descriptive, not a statistical proof.

## Note added after the test

An outside review found that `rel_growth_30d` measures the stock from the next open and the sector from the event-day close. The frozen results were not changed. `scripts/robustness.py` is a later check with matched spans, not part of the original test.
