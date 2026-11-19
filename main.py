"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Model, all in logit space (logit(p) = ln(p / (100 - p)), p in percent):

Exophase's page is a step function: it sits frozen between the page's own
refreshes, so consecutive identical reads are collapsed into one point each
before any fitting, or the frozen page would count as many fresh observations.

One of three branches per trophy:

1. Launch dilution. The history is flat (a single distinct read) and the game
   is only hours old. The next refresh floods the tracked population with new
   players who have barely played, so rates fall hard: the median shifts down
   by ln(K) with K = 0.20, easy trophies (higher launch-night rate) diluted
   slightly less, spread 0.50 logit. The shift decays with the horizon --
   1 / (1 + horizon_days) -- because a one-refresh dilution is a transient,
   not a trend: over days the newcomers play and earn, and rates climb back.

2. Dormant trophy in a mature history. It has too few distinct reads to fit,
   so it joins the pack: the cross-trophy median slope, anchored at its last
   read, with a spread from the pack's slope dispersion plus a random-walk
   floor.

3. Moving trophy. Ordinary least squares of logit(rate) against
   ln(1 + hours since release) over the last day of distinct refreshes, the
   slope shrunk toward the cross-trophy median slope (pseudo-count m = 3),
   anchored at the last read, the extrapolated slope damped by
   span / (span + 0.5 * dx) because progress saturates -- a one-day line must
   not be trusted at full strength several days out. The spread combines the
   slope's standard error, the residual variance and a 0.30 * sqrt(days)
   random-walk floor.

A trophy at ~100% stays at ~100%: everyone tracked has earned it, and the
next wave of players earns it first too.

A free quadratic (concave) fit was tried and dropped: it looked good on clean
synthetics but failed badly under refresh noise, so the damped linear form
above is used instead.

Standard library only, no network, deterministic (no randomness anywhere).
Reads every *.in file in the input directory and writes one *.out per case.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- Model constants (see the module docstring for what each one does) ---
K_DILUTE = 0.20        # central launch-dilution factor: rates fall to ~20%
BETA_EASY = 0.40       # easy (high-rate) trophies are diluted less, per logit
SPREAD_DILUTE = 0.50   # logit spread of the dilution branch
RW_SIGMA = 0.30        # random-walk floor, logit per sqrt(day)
SHRINK_M = 3.0         # pseudo-count shrinking a trophy's slope to the pack
DAMP_HALF = 0.5        # slope damping: span / (span + DAMP_HALF * dx)
MATURE_HOURS = 36.0    # older than this, a flat history is dormant, not launch
WINDOW_HOURS = 24.0    # moving-trophy fit window: the last day of reads
SPREAD_NEAR100 = 0.10  # tight spread for a trophy at ~100%
MIN_SPREAD = 0.15      # floors keep a forecast honest, never degenerate
MAX_SPREAD = 2.50
LOGIT_CAP = 9.210340371976184  # logit(99.99): the page prints 2 decimals


def parse_time(text):
    """ISO 8601 timestamp (with a trailing Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    lg = min(LOGIT_CAP, max(-LOGIT_CAP, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p <= phigh:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def ols(points):
    """Least squares of y on x for [(x, y), ...]; returns (slope, se, resid_var).

    Returns None when the fit is degenerate (fewer than 2 points, no x spread).
    """
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0.0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    residuals = [p[1] - (my + slope * (p[0] - mx)) for p in points]
    if n > 2:
        resid_var = sum(r * r for r in residuals) / (n - 2)
    else:
        resid_var = 0.04
    se = math.sqrt(resid_var / sxx) if resid_var > 0 else 0.0
    return slope, se, resid_var


def distinct_points(reads, index, release):
    """One point per distinct rate for trophy `index`, oldest first.

    Each point is (hours_since_release, x, logit_rate) with
    x = ln(1 + hours). Consecutive identical reads collapse to one point.
    """
    points = []
    last_rate = None
    for read in reads:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        rate = rates[index]
        if last_rate is not None and rate == last_rate:
            continue
        last_rate = rate
        hours = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        hours = max(0.0, hours)
        points.append((hours, math.log1p(hours), logit(rate)))
    return points


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    release = datetime.fromisoformat(
        case["game"]["release_date"] + "T00:00:00+00:00")
    z = [norm_ppf(level) for level in case.get("levels", LEVELS)]

    if not history:
        # No reads at all: fall back to a wide, centred forecast, never crash.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(lg + zi * 1.0))
                                  for zi in z]}
            for t in trophies for lg in [logit(1.0)]]}

    # Last non-null rate per trophy, and the pack's median launch-night rate.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = rates[i]
                break
        last_rates.append(1.0 if rate is None else rate)
    pack_median_rate = median([r for r in last_rates if r < 99.99])
    pack_median_logit = logit(pack_median_rate)
    last_read_time = parse_time(history[-1]["at"])
    game_age_hours = max(
        0.0, (last_read_time - release).total_seconds() / 3600.0)
    x_last_read = math.log1p(game_age_hours)
    x_target = math.log1p(max(
        0.0, (target - release).total_seconds() / 3600.0))

    # Per-trophy distinct-point series, then the pack's moving slopes.
    series = [distinct_points(history, i, release)
              for i in range(len(trophies))]
    slopes = []
    resid_vars = []
    for points in series:
        window = [p for p in points
                  if (checkpoint - release).total_seconds() / 3600.0
                  - p[0] <= WINDOW_HOURS]
        fit = window if len(window) >= 3 else points
        if len(fit) >= 3:
            result = ols([(p[1], p[2]) for p in fit])
            if result is not None:
                slopes.append(result[0])
                resid_vars.append(result[2])
    pack_slope = median(slopes) if slopes else 0.0
    pack_resid = median(resid_vars) if resid_vars else 0.09
    if slopes:
        mean_slope = sum(slopes) / len(slopes)
        slope_disp = math.sqrt(
            sum((s - mean_slope) ** 2 for s in slopes) / len(slopes))
    else:
        slope_disp = 0.25

    decay = 1.0 / (1.0 + horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        lg_last = logit(rate)
        points = series[i]

        if rate >= 99.99:
            # Everyone tracked has earned it; the next wave earns it first.
            med, spread = LOGIT_CAP, SPREAD_NEAR100
        elif len(points) <= 1 and game_age_hours <= MATURE_HOURS:
            # Branch 1: launch dilution. Flat history, game hours old.
            shift = (math.log(K_DILUTE)
                     + BETA_EASY * (lg_last - pack_median_logit)) * decay
            med = lg_last + shift
            spread = SPREAD_DILUTE
        elif len(points) >= 3:
            # Branch 3: moving trophy, damped shrunk OLS in ln(1 + hours).
            window = [p for p in points
                      if (checkpoint - release).total_seconds() / 3600.0
                      - p[0] <= WINDOW_HOURS]
            fit = window if len(window) >= 3 else points
            result = ols([(p[1], p[2]) for p in fit])
            # The forecast is anchored at the LAST READ, not at the last
            # distinct point: for a trophy whose rate has sat still, the
            # last distinct point can be days old, and extrapolating the
            # pack slope from there would invent movement that never
            # happened.
            dx = x_target - x_last_read
            if result is None:
                med = lg_last + pack_slope * dx
                spread = math.sqrt((slope_disp * dx) ** 2
                                   + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                   + 0.04)
            else:
                slope, se, resid_var = result
                n = len(fit)
                shrunk = (n * slope + SHRINK_M * pack_slope) / (n + SHRINK_M)
                span = fit[-1][1] - fit[0][1]
                damped = shrunk * span / (span + DAMP_HALF * dx) if dx > 0 else shrunk
                med = lg_last + damped * dx
                spread = math.sqrt((se * dx) ** 2 + resid_var
                                   + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
        else:
            # Branch 2: dormant in a mature history; join the pack's slope.
            dx = x_target - x_last_read
            med = lg_last + pack_slope * dx
            spread = math.sqrt((slope_disp * dx) ** 2
                               + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                               + 0.04)

        spread = min(MAX_SPREAD, max(MIN_SPREAD, spread))
        med = min(LOGIT_CAP, max(-LOGIT_CAP, med))
        quantiles = [round4(clamp_pct(inv_logit(med + zi * spread)))
                     for zi in z]
        # Guard the non-decreasing invariant against rounding edge cases.
        for j in range(1, len(quantiles)):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
