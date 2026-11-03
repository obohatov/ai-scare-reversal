# AI Scare Reversal

Final project for the [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) by DataTalks.Club. An end-to-end system that detects sharp stock falls, scores them, simulates trading and runs daily.

> **Main result:** the strategy did not beat its benchmarks. Over the test period the primary strategy made 10.3% a year against 15.3% for SPY. The tested models found no useful signal to separate falls that recover from falls that continue. The 11 selected AI events averaged +1.9% against their sectors after 30 sessions, compared with -0.9% for ordinary sharp falls. The two samples differ in selection, period and aggregation, so this is descriptive and cannot establish an AI-specific reversal.

This is a research project, not investment advice.

## Problem

New AI products have pushed whole groups of listed companies down within a day. According to a [Jefferies table cited by Yahoo Finance](https://finance.yahoo.com/news/stock-market-reflecting-fears-ai-133000491.html), legal technology stocks fell 16% on average on 3 February 2026, and freight logistics stocks fell 19% on 12 February 2026.

[Savor (2012)](https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:3:p:635-659) found that large price moves with firm-specific information tend to continue, while moves without it tend to reverse. An AI scare is a third case: there is news, but it concerns another company's product.

**Research question:** after a sharp fall relative to its sector, can a model tell which stocks will recover over the next 30 trading days, and do AI-driven falls behave like falls with no news or like falls with company news?

The result would serve an investor deciding whether to buy after a scare or stay out. This differs from the course setup in three ways: it trades events instead of every day, measures stocks against their sector, and adds hand-labelled AI events.

## Results

The periods were separated in advance: train to 2022, validation 2023-2024, test from January 2025 to October 2026. Rules were fixed in [`reports/test_protocol.md`](reports/test_protocol.md) before the test period was opened once. [`notebooks/research.ipynb`](notebooks/research.ipynb) holds the tables and charts.

| Strategy | Validation CAGR | Test CAGR | Test return/volatility | Test max drawdown |
|---|---|---|---|---|
| SPY | 25.8% | 15.3% | 0.93 | -18.8% |
| Equal-weight S&P 500 | 14.2% | 15.5% | 1.02 | -15.7% |
| **`no_news`, primary** | **12.0%** | **10.3%** | 0.56 | -17.9% |
| `all_drops`, control | 7.1% | 8.1% | 0.47 | -19.0% |
| `no_news_core_20` | 22.9% | 14.5% | 0.89 | -18.0% |
| `news_spread`, long-short | -1.5% | 8.9% | 1.10 | -7.4% |
| `no_news_stop_15` | 7.7% | 8.0% | 0.46 | -20.3% |
| `no_news_5_positions` | 13.9% | 26.5% | 0.98 | -16.3% |
| `top_score` | 3.4% | 19.1% | 1.09 | -18.8% |
| Cash at Fed Funds | 5.2% | 4.0% | | |

The return/volatility column is a Sharpe ratio divided by volatility with the risk-free rate assumed to be zero.

- **No broad reversal advantage.** After a sharp fall, the chance of beating the sector was no higher than on ordinary days: 46.0% in training and 44.9% in validation, against about 46.8%. This concerns relative performance. Fallen stocks could still rise in absolute terms.
- **No useful predictive signal found in the tested models.** The best validation AUC was 0.509 at 30 sessions and 0.522 at 5 sessions, across 60 fits of a limited set of models and features. This does not show that no signal exists. Tuning stopped there, as declared.
- **Savor's direction appears, weakly.** This is not a replication: Savor used analyst reports, and this project uses SEC 8-K filings as a proxy. Falls after earnings did worse than falls without an 8-K, but the gap was 0.27 points (standard error 0.40) on validation and 0.77 (0.48) on test.
- **Do not read the best test rows as the strategy.** `no_news_5_positions` is the primary rule with 75 trades and high volatility, and its 10-position version made 10.3%. `top_score` made 20 trades. Its threshold passed 1,376 of 2,345 validation signals but 132 of 2,890 test signals, because the model was refitted and tree scores tie, so the rule selects very different shares in the two periods.
- **`news_spread` deserves more study, nothing more.** It lost to cash on validation and beat it on test.

### AI events

| Group | Change against sector, 30 sessions |
|---|---|
| 11 AI events | +1.9% (standard error 1.9) |
| 2,584 ordinary sharp falls | -0.9% |

The AI group counts every listed stock, whether or not it passed the sharp-fall rule, and averages by event. The reference group holds single sharp falls of other index members. The comparison is a rough guide only.

Seven of the 11 events beat their sector. The results do not show the expected pattern, in which falls reverse when a product replaces no paid work: events labelled `no` changed -1.0%, `partly` +2.5% and `yes` +6.0%. With 3, 7 and 1 events, and `yes` resting on Chegg alone, this cannot test the thesis statistically. At 90 sessions the average rests on two cybersecurity events; the other nine average about -8%.

## Limits

- `no_news` means no 8-K within two days, not an absence of information. Executive remarks, sector news, media reports and analyst changes are not observed.
- The sector-relative target compares a stock from the next open with its sector from the event-day close, so the sector span also holds the overnight move before entry. This was found after the test period was opened. The frozen results stay as they are, and `reports/robustness_aligned_target.json` shows a matched close-to-close version. Strategy returns use actual prices and are not affected. The conclusions hold under that version: sharp falls beat their sector 45.7%, 45.3% and 45.5% of the time in training, validation and test, against 47.0%, 47.0% and 45.0% on ordinary days. AI events average +2.9% against -1.0% for ordinary sharp falls, but this span includes the overnight move before any possible entry.
- Standard errors treat falls as independent, but falls in one sector or market wave are correlated, so uncertainty is understated. Cluster-robust estimates were not run.
- AI labels come from one person, with written reasons but no second coder, and `no` against `partly` is a hard line to draw.
- Yahoo serves no history for most acquired or bankrupt companies. Index members with prices cover about 72% of the 2010 index and about 97% of the end-2024 index, before symbol renames. Training leans towards survivors; the frozen test universe is much less affected, though not free of the bias.
- AI events entered the list because stocks fell, so the sample cannot show how often similar announcements caused no fall.
- Six of the 11 events fall within two weeks, so they are not independent. Tickers in `proposed_tickers` were chosen by sector, without a documented market reaction.
- The list stops at March 2026 because web searches found no later matching events, and this search was not systematic. There may be others.
- The simulation uses the next open and fixed costs. It ignores slippage, bid-ask spreads, market impact and limits on short selling, and the ETF sleeve moves close to close.
- Page views start in July 2015 and cover current members and peers only, reaching 64.7% of events. EDGAR history covers tickers that it can map today, reaching 92.9% of events.

## Reproduce

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add CONTACT_NAME, CONTACT_EMAIL and DATA_URL
pytest
```

There are two ways to run it.

**From the prepared file**, with no data calls. Set `DATA_URL` to the [shared dataset](https://drive.google.com/file/d/1cqTS1uVbbhd7WkchzR0qGluzyGH7DUI6/view?usp=sharing):

```bash
python main.py --mode file
```

**From the sources:**

```bash
python main.py --mode full
```

`notebooks/workflow.ipynb` shows the same run step by step. Each stage can also run alone:

```bash
python -m scripts.check_sources --label local
python -m scripts.data_repo --step all
python -m scripts.transform
python -m scripts.eda
python -m scripts.train
python -m scripts.horizon_check
python -m scripts.simulate
python -m scripts.simulate --period test --confirm-frozen
python -m scripts.ai_slice
python -m scripts.robustness
```

## How it works

1. **Data.** Yahoo prices since 2010, FRED macro series, S&P 500 history from [fja05680/sp500](https://github.com/fja05680/sp500), SEC EDGAR 8-K filings and Wikimedia page views, combined in `data/processed/panel.parquet`: 2.59 million rows and 681 tickers.
2. **Events.** A daily return at least 2.5 standard deviations and 3 percentage points below the sector. Sector returns leave out the stock itself, and repeats within 5 sessions are ignored. This gave 17,957 events.
3. **Features.** 38 numeric features and 5 dummy sets, described in [`docs/features.md`](docs/features.md). Every feature uses data from the close of the event day or earlier, and a test checks this.
4. **Target.** Whether the stock, from the next open to the close 30 sessions later, beats its sector from the event-day close to that same close.
5. **Models.** Logistic regression, Decision Tree, Random Forest and XGBoost, with 20 settings compared on validation.
6. **Simulation.** Vector and exact versions, with up to 10 positions, reinvestment, 0.1% cost per side and capital held in SPY between trades.
7. **Automation.** `python main.py --mode update` adds new data incrementally, finds signals, sends a Telegram message and places paper orders. A Railway cron service runs it from the `Dockerfile` on weekdays at 22:30 UTC (`30 22 * * 1-5`, set in the service settings), with data on a volume set by `DATA_DIR`. Telegram and Alpaca run only when their keys are set, and Alpaca uses the paper endpoint only.

## Repository layout

```text
.
├── Dockerfile
├── main.py          # whole workflow
├── data/            # local only, never committed
├── docs/            # feature descriptions
├── events/          # hand-labelled AI events
├── notebooks/       # workflow and research
├── reports/         # results and the test protocol
├── scripts/         # one module per stage
└── tests/
```

## Data policy

Data files stay out of Git. Reviewers can use the shared dataset link above or rebuild it from the APIs.
