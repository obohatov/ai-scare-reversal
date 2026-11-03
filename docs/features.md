# Features

Every feature uses data available at the close of the event day. `reports/feature_sets.json` holds the lists used in code.

## To predict

| Name | Meaning |
|---|---|
| `is_positive_rel_30d` | 1 if the stock beats its sector from the next open to the close 30 sessions later |
| `rel_growth_30d` | Stock growth divided by close-to-close sector growth from the event day; the sector span also holds the overnight move before entry |
| `growth_30d` | Stock growth over that span |

## Numeric

| Name | Meaning |
|---|---|
| `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d`, `ret_252d` | Adjusted returns over 1 to 252 sessions |
| `volatility_21d`, `volatility_63d` | Standard deviation of daily returns |
| `dist_sma_50`, `dist_sma_200` | Distance from moving averages |
| `dist_52w_high`, `dist_52w_low` | Distance from the 252-session high and low |
| `rsi_14` | Relative strength index |
| `atr_14` | Average true range as a share of the close |
| `gap_open` | Open against the previous close |
| `close_location` | Where the close sits in the day's range |
| `volume_ratio_20` | Volume against the previous 20 sessions |
| `log_turnover_20` | Log of 20-session turnover, a rough size measure; London pence are converted, other currencies are not |
| `rel_ret_1d` | Return minus the sector return, leaving out the stock itself |
| `rel_ret_5d_before` | Sum of relative returns over the 5 sessions before the event |
| `rel_z` | `rel_ret_1d` divided by `rel_volatility_60` |
| `rel_volatility_60` | Standard deviation of relative returns over the previous 60 sessions |
| `sector_ret_1d` | Sector return without the stock |
| `sector_share_down` | Share of the sector down more than 2% that day |
| `sector_share_sharp` | Share of the sector past the z-score threshold that day |
| `market_ret_1d`, `market_share_down` | The same measures across all stocks |
| `market_volatility_21` | Volatility of the average stock return |
| `events_past_252` | Earlier events for this stock in 252 sessions |
| `sessions_since_event` | Sessions since its previous event |
| `has_8k_2d` | An 8-K on the event day or the day before; empty when EDGAR history is missing |
| `has_earnings_2d` | The same for 8-K item 2.02, earnings |
| `pageviews_ratio_60` | Page views on the previous day against the 60-day median |
| `dgs10`, `dgs2`, `dff`, `vixcls` | FRED rates and the VIX, available one business day later |
| `yield_curve` | `dgs10` minus `dgs2` |
| `vix_change_5d` | Five-session change in the VIX |

## Dummies

| Name | Meaning |
|---|---|
| `sector` | Yahoo sector |
| `month`, `weekday` | Calendar position |
| `drop_size_bin` | Size of the relative fall in five bins |
| `vix_regime` | VIX level: calm, normal, high, panic |
