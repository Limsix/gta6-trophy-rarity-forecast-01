"""Forecast GTA VI trophy earn rates at week one, as five quantiles each.

Method (all in logit space, where a rate p in percent becomes
log(p / (100 - p))):

* The house's page is a step function frozen between Exophase's own
  refreshes, so consecutive reads that print identical rates are
  collapsed into a single observation first.  Fitting the repeats as
  fresh data would overweight one frozen page.
* Each trophy then takes one of three paths, chosen from its own
  history and the game's age at the checkpoint:
  - LAUNCH DILUTION: the history is a single frozen page and the game
    is hours old.  The next refresh floods the tracked population
    with new players who have barely played, so rates fall hard: the
    median moves down by log(K) on the logit scale, K = 0.19, with
    trophies that already had a higher launch-night rate diluted a
    touch less, and a wide 0.35-logit spread for the dilution
    uncertainty.  The shift is a transient, so it fades a little as
    the horizon grows.
  - DORMANT: a trophy that has not moved in a mature history joins
    the pack: it follows the cross-trophy median slope, anchored at
    its last rate.
  - MOVING: a trophy with a real slope gets an ordinary least squares
    line of logit(rate) against log(1 + hours since release) over the
    last day of distinct refreshes, its slope shrunk toward the
    cross-trophy median, anchored at the last read, with the
    extrapolated slope damped because progress saturates.  The spread
    combines the slope's standard error, the residual variance, and a
    random-walk floor that grows with the horizon.
* The trophy already at 100% stays at ~100%.

Python standard library only, no network, deterministic: the same
case always produces byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at the five levels above.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- Recalibrated model constants ---------------------------------------
DILUTE_K = 0.19        # launch-dilution strength (median rate multiplier)
DILUTE_EXP = 0.40      # how fast the dilution shift fades with horizon
DILUTE_SPREAD = 0.35   # logit spread of the dilution branch
DILUTE_MAX_AGE_H = 18.0  # dilution branch only while the game is this young
DECAY_DAYS = 1.0       # trend damping scale (days)
SHRINK_M = 3.0         # slope shrinkage weight toward the pack median
RW_FLOOR = 0.30        # logit random-walk floor per sqrt(day)


def clamp_pct(x):
    """Clamp a percent into the range the logit can represent."""
    return min(99.99, max(0.01, x))


def logit_of(pct):
    """Logit of a percent (0..100)."""
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(l):
    """Inverse logit, back to a percent."""
    if l >= 0:
        e = math.exp(-l)
        return 100.0 / (1.0 + e)
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_ts(text):
    """Parse an ISO timestamp or a bare date into an aware datetime."""
    text = text.strip()
    if "T" not in text:
        text = text + "T00:00:00+00:00"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def median_of(values):
    """Median of a non-empty list of floats."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def collapse_history(history):
    """Collapse consecutive identical reads into distinct observations.

    Returns a list of (datetime, rates) with one entry per distinct
    page the house saw, oldest first.
    """
    points = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        points.append((parse_ts(read["at"]), rates))
    return points


def ols(xs, ys):
    """Least-squares line through (xs, ys).

    Returns (slope, intercept, resid_var, slope_se).  With fewer than
    three points the variance terms fall back to wide defaults.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0 or n < 2:
        return 0.0, my, 0.25, 0.5
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = sum((y - (intercept + slope * x)) ** 2
                    for x, y in zip(xs, ys))
        var = resid / (n - 2)
        se = math.sqrt(var / sxx)
    else:
        var, se = 0.25, 0.5
    return slope, intercept, var, se


def quantiles_from(median_logit, spread):
    """Five non-decreasing quantiles (percent) from a logit median."""
    out = []
    for z in Z:
        q = inv_logit_pct(median_logit + z * spread)
        out.append(round(min(99.99, max(0.01, q)), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    """Forecast one case dict into the output dict."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    release = parse_ts(case["game"]["release_date"])
    checkpoint = parse_ts(case["at"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = horizon_h / 24.0
    age_h = (checkpoint - release).total_seconds() / 3600.0

    points = collapse_history(case["history"])
    last_rates = points[-1][1] if points else [None] * n_tr

    # Per-trophy series in (time, x, logit) space, x = log(1 + hours
    # since release); only reads where this trophy had a rate.  The
    # time is kept so the last-day window below stays correct even
    # when some reads lack this trophy.
    series = []
    for i in range(n_tr):
        pts = []
        for dt, rates in points:
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            hrs = (dt - release).total_seconds() / 3600.0
            pts.append((dt, math.log1p(max(0.0, hrs)), logit_of(r)))
        series.append(pts)

    # Pack statistics from the trophies that actually move: their OLS
    # slopes and their spans, for shrinkage and for the dormant path.
    slopes = []
    spans = []
    for pts in series:
        if len(pts) >= 2:
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            b, _, _, _ = ols(xs, ys)
            slopes.append(b)
            spans.append(max(xs) - min(xs))
    pack_slope = median_of(slopes) if slopes else 0.0
    pack_span = median_of(spans) if spans else 1.0

    # How far the trend may run: from the checkpoint's log-age to the
    # target's, damped because progress saturates.
    x_now = math.log1p(max(0.0, age_h))
    x_tgt = math.log1p(max(0.0, age_h + horizon_h))
    dx = x_tgt - x_now

    def damped(dx_raw, span):
        return dx_raw * span / (span + 0.5 * dx_raw) if dx_raw > 0 else 0.0

    rows = []
    for i, tr in enumerate(trophies):
        last = last_rates[i] if i < len(last_rates) else None
        if last is None:
            # No read ever listed this trophy: sit at the floor.
            last = 0.01
        if last >= 99.99:
            # Already earned by everyone tracked: it stays ~100%.
            rows.append({"id": tr["id"],
                         "q": [99.0, 99.5, 99.9, 99.99, 100.0]})
            continue

        pts = series[i]
        last_logit = logit_of(last)

        if len(pts) <= 1 and age_h <= DILUTE_MAX_AGE_H:
            # --- Launch dilution -------------------------------------
            ease = last / 100.0
            k_eff = DILUTE_K * (1.0 + 0.8 * ease)
            fade = (1.0 + horizon_h / 24.0) ** (-DILUTE_EXP)
            med = last_logit + math.log(k_eff) * fade
            rows.append({"id": tr["id"],
                         "q": quantiles_from(med, DILUTE_SPREAD)})
            continue

        if len(pts) <= 1:
            # --- Dormant in a mature history: follow the pack --------
            step = damped(dx, pack_span)
            med = last_logit + pack_slope * step
            spread = 0.35 + 0.15 * math.sqrt(horizon_d)
            rows.append({"id": tr["id"],
                         "q": quantiles_from(med, min(1.5, spread))})
            continue

        # --- Moving trophy -------------------------------------------
        # Fit over the last day of distinct refreshes (falling back
        # to the whole series when that window holds fewer than two
        # points), using each point's own timestamp.
        cutoff = checkpoint.timestamp() - 24.0 * 3600.0
        win = [(x, y) for (dt, x, y) in pts if dt.timestamp() >= cutoff]
        if len(win) < 2:
            win = [(x, y) for (_, x, y) in pts]
        xs = [p[0] for p in win]
        ys = [p[1] for p in win]
        b, _, var, se = ols(xs, ys)
        if len(pts) >= 2 and max(ys) - min(ys) < 1e-9:
            # Flat line in disguise: dormant.
            step = damped(dx, pack_span)
            med = last_logit + pack_slope * step
            spread = 0.35 + 0.15 * math.sqrt(horizon_d)
            rows.append({"id": tr["id"],
                         "q": quantiles_from(med, min(1.5, spread))})
            continue
        n_win = len(win)
        b_shrunk = (n_win * b + SHRINK_M * pack_slope) / (n_win + SHRINK_M)
        span = max(xs) - min(xs) if len(xs) > 1 else pack_span
        step = damped(dx, span)
        med = last_logit + b_shrunk * step
        spread = math.sqrt((se * step) ** 2 + var +
                           (RW_FLOOR * math.sqrt(horizon_d)) ** 2)
        spread = min(2.0, max(0.15, spread))
        rows.append({"id": tr["id"],
                     "q": quantiles_from(med, spread)})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        with open(in_path) as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (in_path.stem + ".out")
        with open(out_path, "w") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
