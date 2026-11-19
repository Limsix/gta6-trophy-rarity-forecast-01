# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Limsix/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Limsix/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the players Exophase tracks who will have earned it a week after launch (2026-11-26), as five quantiles per trophy. It reads the recorded trophy-page history up to the deadline and fits a logit-space model with three branches — launch dilution, dormant trophies, and a damped trend — then writes one output file per case with the five quantiles for all 55 trophies.

## Run it

```sh
MAKEFUN_INPUT_DIR=./cases MAKEFUN_OUTPUT_DIR=./out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=c4gsetkv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
