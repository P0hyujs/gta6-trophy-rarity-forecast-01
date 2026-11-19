# GTA VI trophy rarity forecaster

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at the case
target, as five quantiles per trophy, scored by pinball loss on the logit
scale. Competition: `gta6-trophy-rarity-forecast-01` ($250, 20 paid places).

## Model

Exophase's page only changes when Exophase refreshes it, so a trophy's rate
is earners / players-tracked, and the two move for different reasons:

- **N(t)** — players tracked: geometric mean of two recency-weighted fits
  over the last ~36 h of reads: a power law (log N linear in ln(1 + hours
  since the first read), decelerating) and an exponential (log N linear in
  time). A launch base grows explosively then saturates; neither shape is
  known right a week out, and the geometric mean minimises worst-case log
  error. Floored at the last observed N.
- **E_i(t)** — earners of trophy i: recency-weighted least-squares slope of
  earners against cumulative player-hours H(t) (trapezoid integral of N),
  fitted on the recent ~12 h of reads (E vs H bends as the base widens, so
  a whole-window slope overshoots rare trophies). E_i(target) = E_i(last) +
  slope · (H(target) − H(last)), floored at E_i(last), capped at N(target).
  Flat history (fewer than 3 moving reads) falls back to persistence.
- **Spread** — walk-forward backtest inside the case history: the median
  forecaster is replayed from earlier reads at lags of 3/6/12/24 h, its RMS
  logit error scaled by sqrt(horizon / lag) to the target horizon, per
  trophy with a global floor; sd floored at 0.20 logit, capped at 1.20.
  Quantiles = median logit + z_tau · sd.

Rate 100 stays 100; null rates are skipped per trophy; a missing trophy
keeps its last known value; an empty history forecasts persistence at 0.01.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13 stdlib only, deterministic, no network. See `STATUS.md` for
validation results.
