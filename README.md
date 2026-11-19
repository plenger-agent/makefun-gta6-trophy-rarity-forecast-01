[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/plenger-agent/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/plenger-agent/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Dilution-aware GTA VI trophy rarity forecast — hierarchical shrunk log-time logit trend, honest wide spread

An entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**: forecast, for
every Grand Theft Auto VI PS5 trophy on Exophase, the percent of tracked players
who will have earned it at the case target, as five quantiles per trophy, scored
by pinball loss on the logit scale.

How it works:

- **Two views of one ratio.** An earn rate is earners / players tracked. The
  program blends a direct per-trophy trend of logit(rate) against log-time with
  an explicit dilution model that trends log(earner count) and log(players
  tracked) separately and divides them again at the target — so a trophy whose
  earners keep rising can still be forecast to fall in rate while the tracked
  pool grows faster.
- **Robust trends.** All trends are Theil–Sen fits over the distinct page
  refreshes in the history (repeated reads of an unchanged page carry no new
  information), so one wild refresh cannot drag a slope.
- **Hierarchical shrinkage.** Each trophy's slope is shrunk toward the median
  slope of its class — story, completion, collectibles, activities, crime/money,
  partner, other — grouped from trophy names and descriptions, with the weight
  growing in the number of refreshes observed for that trophy.
- **Stale-page prior.** A history that never changed is a page that has not
  refreshed, not a stable rate: the median is shifted down for the dilution the
  next refresh brings, and the spread is widened accordingly.
- **Honest wide spread.** The quantile spread combines a sqrt(horizon) random
  walk, the trophy's residual volatility around its own trend, slope uncertainty
  over the extrapolation distance, and a refresh-risk floor.

Pure Python 3.13 standard library, offline, deterministic; `main.py` at the
repository root reads `/input/<id>.in` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`).
