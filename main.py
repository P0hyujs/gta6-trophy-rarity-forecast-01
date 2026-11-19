"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Model, in one paragraph: Exophase's page is a snapshot that only changes when
Exophase refreshes it, so a trophy's earn rate is earners / players-tracked,
and the two move for different reasons.  Players-tracked N(t) only jumps at
refreshes, while earners E_i(t) grow as players play.  I therefore forecast
N and E separately and divide: rate_i(target) = 100 * E_i(target) / N(target).

  * N(t): two growth models are fitted (recency-weighted) over the last
    ~36 h of reads and their projections combined by geometric mean: a
    power law, log N linear in ln(1 + hours since the first read), which
    decelerates, and an exponential, log N linear in time, which does not.
    A launch's tracked base grows explosively early and then saturates;
    neither shape is known to be right a week out, and the geometric mean
    minimises the worst-case log error of the two.  Floored at the last N.
  * E_i(t): earners are fitted against cumulative player-hours H(t), the
    trapezoid integral of N over the reads, with a recency-weighted least
    squares slope over the last ~36 h of reads.  E_i(target) is the last
    observed E_i plus slope * (H(target) - H(last)), floored at E_i(last)
    and capped at N(target).  A flat history (fewer than 3 moving reads, or
    a slope near zero) falls back to persistence for the median.
  * Spread: the model's own walk-forward backtest inside the case history.
    For several lags the median forecaster is replayed from earlier reads
    only and its logit error measured; the RMS error per lag is scaled by
    sqrt(target horizon / lag) to the target horizon, and the largest scaled
    value (per trophy, with a global floor) is the logit sd.  Quantiles are
    median logit + z_tau * sd, z = [-1.645, -0.674, 0, 0.674, 1.645].

Edge cases: a rate of 100 stays 100; null rates are skipped for that trophy;
a trophy missing from a read keeps its last known value; an empty history
forecasts persistence at 0.01.  Standard library only, deterministic, no
network, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Backtest lags (minutes) used to calibrate the spread.
BACKTEST_LAGS_MIN = (180, 360, 720, 1440)
# Window (hours) of reads the N and E fits use.
FIT_WINDOW_HOURS = 36.0
SD_FLOOR = 0.20
SD_CAP = 1.20


def parse_time(text):
    """ISO-8601 UTC timestamp -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round_half_up_4dp(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def weighted_line(xs, ys, ws):
    """Weighted least-squares line y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


class CaseHistory:
    """Parsed reads of one case, with per-trophy earner series."""

    def __init__(self, case):
        self.times = []          # epoch seconds per read
        self.players = []        # N per read
        self.earners = []        # per read: list of E_i (float) or None
        self.n_trophies = len(case["trophies"])
        for read in case["history"]:
            rates = read["rates"]
            n = float(read["players_tracked"])
            row = []
            for r in rates:
                row.append(None if r is None else float(r) * n / 100.0)
            self.times.append(parse_time(read["at"]))
            self.players.append(n)
            self.earners.append(row)
        # Forward-fill missing earner values from the last known read.
        for j in range(1, len(self.times)):
            for i in range(self.n_trophies):
                if self.earners[j][i] is None:
                    self.earners[j][i] = self.earners[j - 1][i]
        # Cumulative player-hours H(t): trapezoid integral of N over time.
        self.player_hours = [0.0]
        for j in range(1, len(self.times)):
            dt_h = (self.times[j] - self.times[j - 1]) / 3600.0
            avg_n = 0.5 * (self.players[j] + self.players[j - 1])
            self.player_hours.append(self.player_hours[-1] + avg_n * dt_h)

    def __len__(self):
        return len(self.times)

    def window(self, upto_idx, hours):
        """Indices of reads within `hours` before read `upto_idx` (inclusive)."""
        t_end = self.times[upto_idx]
        return [j for j in range(upto_idx + 1)
                if (t_end - self.times[j]) / 3600.0 <= hours]

    def project_players(self, upto_idx, target_time):
        """Forecast N(target) from reads up to `upto_idx`.

        Geometric mean of a decelerating power-law projection (log N linear
        in ln(1 + hours since the first read)) and an exponential one
        (log N linear in time), both recency-weighted over the fit window.
        """
        last_n = self.players[upto_idx]
        idx = self.window(upto_idx, FIT_WINDOW_HOURS)
        if len(idx) < 2:
            return last_n
        t_end = self.times[upto_idx]
        t_first = self.times[0]
        ws = [0.25 + 0.75 * (self.times[j] - self.times[idx[0]])
              / max(1.0, t_end - self.times[idx[0]]) for j in idx]
        log_n = [math.log(max(1.0, self.players[j])) for j in idx]
        # Power law: log N ~ a + b * ln(1 + hours since first read).
        xs_pow = [math.log1p((self.times[j] - t_first) / 3600.0) for j in idx]
        fit_pow = weighted_line(xs_pow, log_n, ws)
        n_pow = None
        if fit_pow is not None and fit_pow[1] > 0:
            x_t = math.log1p((target_time - t_first) / 3600.0)
            n_pow = math.exp(fit_pow[0] + fit_pow[1] * x_t)
        # Exponential: log N ~ a + b * hours since first read.
        xs_exp = [(self.times[j] - t_first) / 3600.0 for j in idx]
        fit_exp = weighted_line(xs_exp, log_n, ws)
        n_exp = None
        if fit_exp is not None and fit_exp[1] > 0:
            x_t = (target_time - t_first) / 3600.0
            n_exp = math.exp(fit_exp[0] + fit_exp[1] * x_t)
            n_exp = min(n_exp, last_n * 50.0)
        if n_pow is None and n_exp is None:
            return last_n
        if n_pow is None:
            return max(last_n, n_exp)
        if n_exp is None:
            return max(last_n, n_pow)
        return max(last_n, math.sqrt(n_pow * n_exp))

    def project_player_hours(self, upto_idx, target_time, n_target):
        """H(target): integrate projected N from the last read to target.

        N is assumed log-linear in ln(1 + hours since release) between the
        last read's N and N(target); the integral is done in small steps.
        """
        h_last = self.player_hours[upto_idx]
        t_last = self.times[upto_idx]
        n_last = self.players[upto_idx]
        if target_time <= t_last or n_target <= n_last:
            return h_last + n_last * (target_time - t_last) / 3600.0
        steps = 24
        total = 0.0
        prev_t, prev_n = t_last, n_last
        t_first = self.times[0]
        x0 = math.log1p(max(0.0, (t_last - t_first) / 3600.0))
        x1 = math.log1p(max(0.0, (target_time - t_first) / 3600.0))
        log_n0, log_n1 = math.log(n_last), math.log(n_target)
        for s in range(1, steps + 1):
            frac = s / steps
            t = t_last + frac * (target_time - t_last)
            x = x0 + frac * (x1 - x0)
            n = math.exp(log_n0 + (log_n1 - log_n0) * (x - x0) / max(1e-9, x1 - x0))
            total += 0.5 * (prev_n + n) * (t - prev_t) / 3600.0
            prev_t, prev_n = t, n
        return h_last + total

    def median_rate(self, trophy_idx, upto_idx, target_time,
                    n_target=None, h_target=None):
        """Median forecast rate (%) for one trophy from reads up to upto_idx."""
        last_e = self.earners[upto_idx][trophy_idx]
        last_n = self.players[upto_idx]
        if last_e is None:
            return 0.01
        if last_e >= last_n - 1e-9:
            return 100.0
        if n_target is None:
            n_target = self.project_players(upto_idx, target_time)
        if h_target is None:
            h_target = self.project_player_hours(upto_idx, target_time, n_target)
        idx = self.window(upto_idx, FIT_WINDOW_HOURS)
        t_end = self.times[upto_idx]
        # (player-hours, earners, read time) triples for this trophy.
        triples = [(self.player_hours[j], self.earners[j][trophy_idx], self.times[j])
                   for j in idx if self.earners[j][trophy_idx] is not None]
        pts = [(h, e) for h, e, _t in triples]
        # Need at least 3 reads and real movement to trust a slope.
        moving = sum(1 for j in range(1, len(pts))
                     if abs(pts[j][1] - pts[j - 1][1]) > 1e-9)
        if len(pts) < 3 or moving < 2:
            return 100.0 * last_e / last_n
        # Slope from the RECENT reads only (last ~12 h): E vs H bends
        # (dE/dH falls as the tracked base widens with casual players), so
        # a slope fitted over the whole window is stale and overshoots rare
        # trophies.  Fall back to the full window when the recent stretch
        # has too few moving reads.
        recent = [(h, e) for h, e, t in triples if t_end - t <= 12 * 3600]
        recent_moves = sum(1 for k in range(1, len(recent))
                           if abs(recent[k][1] - recent[k - 1][1]) > 1e-9)
        use = recent if (len(recent) >= 3 and recent_moves >= 2) else pts
        t_start = self.times[idx[0]]
        ws = [0.25 + 0.75 * (t - t_start) / max(1.0, t_end - t_start)
              for _h, _e, t in triples
              if (t_end - t <= 12 * 3600 if use is recent else True)]
        fit = weighted_line([p[0] for p in use], [p[1] for p in use], ws)
        if fit is None:
            return 100.0 * last_e / last_n
        _intercept, slope = fit
        if slope <= 0:
            return 100.0 * last_e / last_n
        e_target = last_e + slope * (h_target - self.player_hours[upto_idx])
        e_target = min(max(e_target, last_e), n_target)
        return 100.0 * e_target / n_target


def calibrate_spread(hist, case, target_time):
    """Walk-forward backtest: per-trophy logit sd, scaled to the horizon.

    For each lag, every read that has enough earlier history is forecast
    from those earlier reads only; the RMS logit error of the median is
    scaled by sqrt(horizon / lag).  Returns a list of sd per trophy.
    """
    horizon_min = case["horizon_minutes"]
    n = len(hist)
    per_trophy_err = [[] for _ in range(hist.n_trophies)]
    global_err = []
    for lag in BACKTEST_LAGS_MIN:
        lag_sec = lag * 60.0
        errs = [[] for _ in range(hist.n_trophies)]
        # Candidate pseudo-targets: every read with enough earlier history.
        # Subsample to at most ~40 per lag: the backtest is a spread
        # calibration, not a score, and a full sweep is O(reads^2) — on a
        # final-like 290-read history it blows the 8 s run cap.
        candidates = []
        for j in range(1, n):
            t_j = hist.times[j]
            upto = None
            for k in range(j - 1, -1, -1):
                if t_j - hist.times[k] >= lag_sec - 1.0:
                    upto = k
                    break
            if upto is not None and upto >= 2:
                candidates.append((j, upto))
        stride = max(1, len(candidates) // 40)
        for j, upto in candidates[::stride]:
            t_j = hist.times[j]
            # N and H projections depend on the target, not the trophy:
            # compute once per pseudo-target, not once per trophy.
            n_t = hist.project_players(upto, t_j)
            h_t = hist.project_player_hours(upto, t_j, n_t)
            for i in range(hist.n_trophies):
                actual = hist.earners[j][i]
                if actual is None:
                    continue
                pred = hist.median_rate(i, upto, t_j, n_t, h_t)
                if pred >= 100.0 or actual >= hist.players[j] - 1e-9:
                    continue
                err = logit_of(100.0 * actual / hist.players[j]) - logit_of(pred)
                errs[i].append(err)
        scale = math.sqrt(horizon_min / lag)
        for i in range(hist.n_trophies):
            if errs[i]:
                rms = math.sqrt(sum(e * e for e in errs[i]) / len(errs[i]))
                per_trophy_err[i].append(rms * scale)
                global_err.extend([e * scale for e in errs[i]])
    global_sd = 0.0
    if global_err:
        global_sd = math.sqrt(sum(e * e for e in global_err) / len(global_err))
    sds = []
    for i in range(hist.n_trophies):
        own = max(per_trophy_err[i]) if per_trophy_err[i] else 0.0
        sd = max(own, 0.5 * global_sd, SD_FLOOR)
        sds.append(min(sd, SD_CAP))
    return sds


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the expected five")
    trophies = case["trophies"]
    target_time = parse_time(case["target"])
    if not case["history"]:
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                             for t in trophies]}
    hist = CaseHistory(case)
    last_idx = len(hist) - 1
    sds = calibrate_spread(hist, case, target_time)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = case["history"][-1]["rates"][i]
        if last_rate is not None and last_rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        median = hist.median_rate(i, last_idx, target_time)
        med_logit = logit_of(median)
        sd = sds[i]
        qs = []
        for z in Z_SCORES:
            q = inv_logit(med_logit + z * sd)
            qs.append(round_half_up_4dp(min(99.99, max(0.01, q))))
        # Rounding can only keep order (inv_logit is monotone in z), but
        # enforce non-decreasing defensively.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
