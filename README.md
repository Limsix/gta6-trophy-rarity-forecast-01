# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Limsix/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Limsix/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for each of the game's 55 PlayStation 5 trophies, forecast the percent
of Exophase-tracked players who will have earned it at
2026-11-26T00:00:00Z, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 /
0.95), scored by pinball loss on the logit scale.

## Approach

Everything happens in logit space. Consecutive reads that print
identical rates are collapsed first, because Exophase's page is a
step function frozen between its own refreshes. Each trophy then
takes one of three paths:

- **Launch dilution** — when the history is a single frozen page and
  the game is hours old, the next refresh floods the tracked
  population with new players who have barely played, so rates fall
  hard: the median shifts down by log(0.19), with a 0.35-logit spread
  for the dilution uncertainty.
- **Dormant** — a trophy that has not moved in a mature history
  follows the cross-trophy median slope, anchored at its last rate.
- **Moving** — a trophy with a real slope gets a least-squares line
  of logit(rate) against log(1 + hours since release) over the last
  day of distinct refreshes, shrunk toward the pack median and damped
  because progress saturates; the spread combines the slope's
  standard error, residual variance, and a random-walk floor.

The trophy already at 100% stays at ~100%. Python standard library
only, no network, deterministic.

## Run

```sh
MAKEFUN_INPUT_DIR=./cases MAKEFUN_OUTPUT_DIR=./out python3 main.py
```

Reads every `<id>.in` in the input directory and writes
`<id>.out` (one `trophies` row per trophy, five non-decreasing
quantiles each) to the output directory.
