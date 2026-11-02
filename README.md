# AI Scare Reversal

Final project for the [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) by DataTalks.Club.

> **Status:** stage 5 of 9, models. No strategy has been tested yet, and no result here claims profitability.

## Problem

New AI products have pushed whole groups of listed companies down within a day. According to a [Jefferies table cited by Yahoo Finance](https://finance.yahoo.com/news/stock-market-reflecting-fears-ai-133000491.html), legal technology stocks fell 16% on average on 3 February 2026, and freight logistics stocks fell 19% on 12 February 2026.

[Savor (2012)](https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:3:p:635-659) found that large price moves with firm-specific information tend to continue, while moves without it tend to reverse. An AI scare is a third case: there is news, but it concerns another company's product.

**Research question:** after a sharp fall relative to its sector, can a model tell which stocks will recover over the next 30 trading days, and do AI-driven falls behave like falls with no news or like falls with company news?

## Planned approach

1. Build one table of daily data for S&P 500 members and European peers, 2010–2026.
2. Find sharp falls relative to each stock's sector.
3. Train Decision Tree, Random Forest and XGBoost models to predict whether a stock beats its sector over 30 trading days.
4. Test on 2025–2026 with the stock list frozen at the end of 2024, which limits survivorship bias.
5. Simulate strategies against the S&P 500. Idle capital stays in an S&P 500 ETF between signals.
6. Analyse hand-labelled AI events as a separate slice of the test period. Labels are fixed before any later return is viewed.
7. Automate the workflow with daily runs and trade notifications.

## Quick start

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in CONTACT_NAME and CONTACT_EMAIL
pytest
python -m scripts.check_sources --label codespace
```

The check writes `reports/source_check_<label>.md` and `.json`. Both are free of credentials and contact details.

To test one source on another machine, for example Yahoo on a server:

```bash
python -m scripts.check_sources --label server --only host,universe,yahoo_prices
```

## Build the dataset

```bash
python -m scripts.data_repo --step all
```

This writes `data/processed/panel.parquet`, with one row per ticker and trading day, and `reports/data_build.json`. A repeated run resumes from what is already saved. One step can be run alone with, for example, `--step prices`.

## Build the features

```bash
python -m scripts.transform
```

This writes `data/processed/events.parquet`, with one row per sharp fall, plus `reports/feature_sets.json` and `reports/features_summary.json`.

- **Event:** a daily return at least 2.5 standard deviations and 3 percentage points below the sector. Sector returns leave out the stock itself, and repeats within 5 sessions are ignored.
- **Target:** whether the stock beats its sector from the next open to the close 30 sessions later.
- **Splits:** train to 2022, validation 2023-2024, test from 2025, with a 45-day gap before each later period.
- **Universe:** the main test uses S&P 500 members frozen at the end of 2024. European peers and event extras were chosen because of AI scares, so they are flagged and reported separately.

## Explore and train

```bash
python -m scripts.eda
python -m scripts.train
```

`scripts.eda` writes `reports/eda.json` from the training and validation periods. `scripts.train` compares logistic regression, Decision Tree, Random Forest and XGBoost on validation, then refits the chosen model on both periods to score the test period. Test metrics are not computed at this stage.

The first model round found no signal at the 30-session horizon: validation AUC stayed near 0.5. One declared follow-up tests horizons of 5 and 10 sessions with the same features:

```bash
python -m scripts.horizon_check
```

No further tuning follows from this round, whatever it shows.

## Data sources

| Source | Purpose | Access |
|---|---|---|
| Yahoo Finance | Daily prices, benchmarks, earnings dates | `yfinance` |
| FRED | Macro series | Public CSV endpoint |
| Wikipedia | Current S&P 500 members and company names | Public page |
| [fja05680/sp500](https://github.com/fja05680/sp500) | Historical S&P 500 membership since 1996 (MIT licence) | CSV pinned to one commit |
| SEC EDGAR | 8-K filings as a firm-news flag | Free; clients declare a name and email |
| Wikimedia | Daily page views as an attention signal | Free; clients declare a contact |

## Repository layout

```text
.
├── data/          # local only, never committed
├── events/        # hand-labelled AI events
├── notebooks/     # research notebooks
├── reports/       # source check reports
├── scripts/
│   ├── check_sources.py
│   ├── data_repo.py
│   ├── eda.py
│   ├── settings.py
│   ├── train.py
│   ├── transform.py
│   └── universe.py
└── tests/
```

## Known data limits

Yahoo serves no history for most companies that were acquired or went bankrupt. Index members with prices cover about 72% of the 2010 index and about 97% of the end-2024 index, before symbol renames are applied. Training data therefore leans towards survivors, and the test period is much less affected.

## Data policy

Data files stay out of Git. Reviewers will be able to download the prepared dataset from a shared drive link or rebuild it from the APIs.
