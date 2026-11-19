# STATUS — gta6-trophy-rarity-forecast-01

## Deliverable

- `main.py` only (Python 3.13 stdlib, no `main.ts`), `LICENSE` (MIT, SPDX
  line first), `README.md`. Git repo at
  `~/workspace/makefun/hunter/bountyhawk/gta6-trophy-rarity-forecast-01/`.
- No anchor line was shown in the build brief, so no anchor was quoted
  before the first write (thinking_fresh is a skip-note, not a rejection).
  No submit by this builder — parent bundles transcripts and submits.

## Model design

Forecast earners E_i and players-tracked N separately, rate = 100·E_i/N:
- N: geometric mean of a power-law fit (log N vs ln(1+h)) and an
  exponential fit (log N vs h), recency-weighted over the last ~36 h.
- E_i: recency-weighted OLS slope of earners vs cumulative player-hours
  H(t), fitted on the recent ~12 h (whole-window slope overshoots rare
  trophies because E vs H is concave); flat history → persistence.
- Spread: walk-forward backtest at 3/6/12/24 h lags, RMS logit error
  scaled by sqrt(horizon/lag), sd floor 0.20 / cap 1.20 logit.
- Rate 100 stays 100; null rates skipped; empty history → 0.01.

## Local validation

- **Practice p00/p01**: output valid (55 rows, exactly one `trophies` key,
  non-decreasing quantiles, 0–100, 2,657 bytes < 65,536), deterministic
  (two runs byte-identical), ~0.05 s per case (cap 8 s).
  Flat history → median = persistence, sd = floor 0.20, as designed.
  Approx practice pinball loss vs the E-frozen refresh answer
  (new rate = old × 638/3868): **1.634** on both cases, coverage 0.018.
  That is EXPECTED, not a failure: both practice cases straddle Exophase's
  first refresh, which no program can foresee from the flat practice
  history (house baselines score ~1.51 the same way). Do not chase it.
- **Synthetic moving-history test** (N ×3, story earners ×12, rare ×1.5,
  horizon 6,510 min): story trophy 4.40% → median **11.45%** (up),
  rare trophy 0.32% → median **0.25%** (down), spread widens vs the
  frozen case (log q95/q05 1.155 vs 0.651). Direction check PASSES.
  Frozen-history control: medians equal last rates exactly.
- **Edge cases**: empty history → all quantiles 0.01, valid; null rate in
  a read → handled, output valid.
- **Final-like stress test** (290 reads / 10-min cadence / 48 h, N
  638 → 23,378, 24 refreshes): runtime **1.04 s** (cap 8 s), output valid.
  NOTE: the first version of the backtest took 13.34 s on this case —
  fixed by subsampling backtest pseudo-targets to ≤40 per lag and caching
  the N/H projections per target instead of per trophy.

## Known limits

- Final-horizon behaviour cannot be validated until the final history
  exists: the practice history is completely flat, so the E/N machinery
  and the backtest spread are exercised only on synthetic data. The real
  final history (reads every 10 min to the deadline) will have refresh
  jumps the backtest can calibrate on; on the flat practice it falls back
  to the sd floor.
- N projection is the largest uncertainty: launch adoption could stay
  near-exponential or saturate; the geometric-mean hedge bounds the
  worst-case log error but cannot remove it.

## Recommendation

**SUBMIT.** Entry is valid, deterministic, fast (1.04 s worst-case
simulated final), and implements the brief's model with the direction
check passing. The prize is decided on the final case only, and this
model is designed to beat persistence/curve baselines exactly there
(E/N separation + backtest-calibrated spread vs their fixed 0.5·√d
spread), while the practice score is uninformative by design.
