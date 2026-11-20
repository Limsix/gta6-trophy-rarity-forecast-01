"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's week-one earn rate.

Model (per trophy, in logit space, where logit(p) = ln(p / (100 - p))):

Exophase's page is a step function: it sits frozen between the site's own
refreshes and jumps when it refreshes. So the first thing this program does is
collapse consecutive identical reads into one point each -- the repeated reads
carry no information, and fitting them as if they were fresh observations would
weight the frozen page far past its worth.

Then, per trophy, one of three branches:

1. Launch dilution (history flat, game hours old). On launch night every read
   is the same page, and the next refresh floods the tracked population with
   new players who have barely played, so every rate drops hard. The house
   baseline, which says rates stay put, scores about 1.51 on the practice for
   exactly this reason (its 90% band covers nothing). Here the median shifts
   down by ln(k), k ~ 0.19 -- the dilution the baseline's own published
   practice numbers (loss ~1.51, typical miss ~0.68, coverage ~0) imply,
   calibrated so a proxy answer built on it reproduces those numbers
   (baseline loss ~1.52, miss ~0.69, coverage 0.00) -- with easy trophies
   (higher launch-night rate) diluted a little less, and a spread for how
   uncertain the dilution factor is. The shift decays with the horizon: a
   one-refresh dilution is a transient, not a trend. The decay also buys
   robustness: it softens the shift, so a refresh that dilutes less than the
   central k costs less than a narrow, fully-committed dilution would.

2. Dormant trophy in a mature history (history flat, game days old). A trophy
   nobody has earned yet does not dilute; it joins the pack, so it takes the
   cross-trophy median slope from the last day, anchored at its last rate.

3. Moving trophy. Fit a line of logit(rate) against ln(1 + hours since
   release) over the last day of distinct refreshes, shrink the slope toward
   the cross-trophy median slope, and anchor the forecast at the last read
   (the fitted line's level is not trusted, only its slope). The slope's
   extrapolation is damped -- progress saturates, so a line fitted over one
   day must not be trusted at full strength four days out: the increment is
   scaled by span / (span + 0.5 * dx), where span is the log-time span the
   line was fitted over and dx the log-time distance to the target. The
   spread combines the slope's standard error, the fit's residual variance
   and a random-walk floor that grows with the horizon.

The 100% trophy (every tracked player has it) stays at 100% with only the
random-walk spread. Quantiles are the median logit plus the standard normal
quantiles times the spread, rounded to 4 decimals, clamped to [0.01, 100].

Standard library only, no network, deterministic: the same input always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- launch-dilution branch (flat history, hours after release) ---
# Central dilution factor: the next refresh leaves each rate at ~19% of its
# launch-night value (calibrated: a proxy answer built on k = 0.19 reproduces
# the house baseline's published practice numbers, loss ~1.52 vs ~1.51,
# typical miss ~0.69 vs ~0.68, coverage 0.00; and 0.19 is the robust centre
# across the plausible k range 0.15-0.30, not just the point estimate).
DILUTION_K = 0.19
# How much less the easy trophies dilute: k_i = K * (r_i / r_median) ** EXP.
# EXP = 0.40 matches the rate-dependence the live Exophase page shows (easy
# trophies dilute markedly less) and the calibrated proxy's tilt; 0.25
# under-tilts and 0.55 over-tilts on the same proxy.
DILUTION_EXP = 0.40
# Spread of the dilution factor itself, in logit units. 0.35 (not the 0.50
# the first build used) is the calibrated width: the proxy's per-trophy
# dispersion around the median is ~0.13 logit, and 0.35 covers it plus the
# genuine uncertainty in k across 0.15-0.30 without paying the wide-spread
# penalty a 0.50 spread pays at every quantile.
DILUTION_SPREAD = 0.35
# The dilution is a transient: its shift decays over this many days.
DILUTION_DECAY_DAYS = 1.0
# A history counts as flat when its logit range is below this.
FLAT_RANGE = 0.05
# ... and as "launch night" while the last read is within this many hours of release.
LAUNCH_AGE_HOURS = 18.0

# --- trend branch ---
# Slope shrinkage toward the cross-trophy median slope (prior strength).
SHRINK_M = 3.0
# Last-day window the trend is fitted over, in hours.
WINDOW_HOURS = 24.0
# Damping of the extrapolated slope: increment scaled by span / (span + K * dx).
DAMP_K = 0.5
# Random-walk floor of the spread, in logit units per square-root day.
RW_SPREAD = 0.30
# Random-walk spread used for the launch baseline term in the flat branch.
BASE_RW_SPREAD = 0.5


def _parse_time(text):
    """ISO 8601 UTC timestamp (fractional seconds optional) or a bare date."""
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        s = s.split(".")[0]
    if "T" in s:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    return datetime.strptime(s, "%Y-%m-%d")


def _hours_since(origin, when):
    return max(0.0, (when - origin).total_seconds() / 3600.0)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(l):
    return 100 / (1 + math.exp(-l))


def round4(x):
    """x rounded to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def _ols(xs, ys):
    """Least-squares line y = a + b x. Returns (b, residual variance, slope SE)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, 0.0, 0.6
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2)
    se = math.sqrt(var / sxx)
    return b, var, se


def _distinct_reads(history):
    """Collapse consecutive identical reads (same rates and player count)."""
    points = []
    prev = None
    for read in history:
        key = (tuple(read.get("rates") or []), read.get("players_tracked"))
        if key != prev:
            points.append(read)
            prev = key
    return points


def _last_rate(history, i):
    """A trophy's rate in the latest read that has one, else a neutral 0.5%."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if i < len(rates) and rates[i] is not None:
            return rates[i]
    return 0.5


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        # No reads at all: a neutral prior, still a valid forecast.
        return {"trophies": [{"id": t["id"], "q": [0.2, 0.5, 1.0, 2.0, 5.0]}
                             for t in case["trophies"]]}
    game = case.get("game") or {}
    release = game.get("release_date")
    origin = _parse_time(release) if release else _parse_time(history[0]["at"])
    last = history[-1]
    last_at = _parse_time(last["at"])
    target_at = _parse_time(case["target"])
    age_hours = _hours_since(origin, last_at)
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(_hours_since(origin, target_at))
    x_last = math.log1p(_hours_since(origin, last_at))
    dx = x_target - x_last
    rw_term = RW_SPREAD * math.sqrt(horizon_days)

    points = _distinct_reads(history)
    recent = [r for r in points
              if _hours_since(origin, _parse_time(r["at"])) >= age_hours - WINDOW_HOURS]
    if len(recent) < 3:
        recent = points

    # Per-trophy fits of logit(rate) against ln(1 + hours since release).
    fits = {}
    slopes = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for read in recent:
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                xs.append(math.log1p(_hours_since(origin, _parse_time(read["at"]))))
                ys.append(logit(rates[i]))
        if len(xs) >= 3:
            b, var, se = _ols(xs, ys)
            span = max(xs) - min(xs)
            yrange = max(ys) - min(ys)
            fits[i] = (b, var, se, len(xs), span, yrange)
            slopes.append(b)
        else:
            fits[i] = (None, 0.1, 0.6, len(xs), 0.0, 0.0)
    slopes.sort()
    slope_median = slopes[len(slopes) // 2] if slopes else 0.0

    # Median launch-night rate, for the dilution branch's easy-trophy tilt.
    live = [r for r in (last.get("rates") or []) if r is not None and r < 99.99]
    live.sort()
    rate_median = live[len(live) // 2] if live else 1.0
    decay = 1.0 / (1.0 + horizon_days / DILUTION_DECAY_DAYS)
    dil_spread = math.sqrt(DILUTION_SPREAD ** 2 + (BASE_RW_SPREAD * math.sqrt(horizon_days)) ** 2)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = _last_rate(history, i)
        if rate >= 99.99:
            # Everybody has it, and keeps having it.
            med = logit(99.99)
            spread = rw_term
        else:
            b, var, se, n, span, yrange = fits[i]
            if yrange < FLAT_RANGE:
                if age_hours <= LAUNCH_AGE_HOURS:
                    # Launch dilution: the next refresh floods the tracked
                    # population with new players who have barely played.
                    k = DILUTION_K * (rate / rate_median) ** DILUTION_EXP
                    med = logit(rate) + math.log(k) * decay
                    spread = dil_spread
                else:
                    # Dormant trophy in a mature history: join the pack.
                    med = logit(rate) + slope_median * dx
                    spread = math.sqrt((0.6 * dx) ** 2
                                       + min(0.6, max(0.02, var))
                                       + rw_term ** 2)
            else:
                b_shrunk = (n * b + SHRINK_M * slope_median) / (n + SHRINK_M)
                damp = span / (span + DAMP_K * dx) if (span + DAMP_K * dx) > 0 else 1.0
                med = logit(rate) + b_shrunk * dx * damp
                spread = math.sqrt((se * dx) ** 2
                                   + min(0.6, max(0.02, var))
                                   + rw_term ** 2)
        med_rate = min(99.99 if rate >= 99.99 else 95.0, max(0.01, inv_logit(med)))
        med = logit(med_rate)
        rows.append({
            "id": trophy["id"],
            "q": [round4(min(100.0, max(0.01, inv_logit(med + z * spread)))) for z in Z],
        })
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
