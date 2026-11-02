# AI Scare Reversal

Final project for the [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) by DataTalks.Club.

> **Status:** stage 2 of 9, data build. No strategy has been tested yet, and no result here claims profitability.

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
├── notebooks/     # research notebooks
├── reports/       # source check reports
├── scripts/
│   ├── check_sources.py
│   ├── data_repo.py
│   ├── settings.py
│   └── universe.py
└── tests/
```

## Data policy

Data files stay out of Git. Reviewers will be able to download the prepared dataset from a shared drive link or rebuild it from the APIs.
