"""Dilution-aware GTA VI trophy rarity forecast for make.fun gta6-trophy-rarity-forecast-01.

Model, in one paragraph: Exophase earn rates are a ratio -- earners over players
tracked -- and the denominator grows explosively after launch, so a rate can fall
while its earner count rises. This program therefore forecasts the ratio two ways
and blends them on the logit scale: (1) a direct per-trophy robust Theil-Sen trend
of logit(rate) against log-time, and (2) an explicit dilution model that trends
log(earner count) and log(players tracked) separately against log-time and divides
them again at the target. Per-trophy slopes are shrunk hierarchically toward the
median slope of the trophy's class (story, completion, collectibles, activities,
crime/money, partner, other), with the shrinkage weight growing with the number of
distinct page refreshes observed for that trophy. The quantile spread is honest
and wide: it combines a random-walk term in sqrt(horizon days), the trophy's own
residual volatility around its trend, slope uncertainty over the extrapolation
distance, and a refresh-risk floor, because Exophase refreshes its counts on its
own schedule and a stale page can jump between the checkpoint and the target.

Runs offline, standard library only, deterministic. Reads every <id>.in in the
input directory and writes <id>.out to the output directory, per the kit contract:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]} in percent.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped
# beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Spread of a rate's move, in logit units per square-root day (random-walk term).
# A trend-following model's residual uncertainty is smaller than a pure random
# walk's from the last value, so this sits a little below the house baseline's
# 0.5; the refresh-risk and volatility terms below add back what is real.
SIGMA_PER_SQRT_DAY = 0.42
# A stale page (few distinct refreshes seen) may jump at the next refresh; this is
# the extra logit-scale spread charged when the history shows at most two states.
REFRESH_RISK_FLAT = 1.05
REFRESH_RISK_SEEN = 0.35
# Bounds that keep an extrapolated trend honest rather than heroic.
MAX_LOGIT_RISE = 2.2
MAX_LOGIT_FALL = -3.2
MAX_SLOPE = 3.0
# Theil-Sen is quadratic in the number of points; thin long series to this many.
MAX_FIT_POINTS = 44


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, matching JavaScript's Math.round behaviour."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> float:
    """ISO-8601 UTC timestamp -> epoch seconds. Tolerates a trailing Z."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Robust line fit: (slope, intercept at x of first point, slope IQR).

    The slope is the median of all pairwise slopes, so a single wild refresh
    cannot drag the trend the way a least-squares fit would let it. The slope
    IQR is returned as a cheap, honest measure of how much the pairs disagree.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, ys[0], 0.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx > 1e-9:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return 0.0, ys[-1], 0.0
    slope = median(slopes)
    ordered = sorted(slopes)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    # Intercept anchored so the line passes through the last observed point's
    # neighbourhood: use the median residual against the final x.
    intercept = median([y - slope * x for x, y in zip(xs, ys)])
    return slope, intercept, max(0.0, q3 - q1)


def thin(points: list[tuple[float, float]], cap: int) -> list[tuple[float, float]]:
    """Evenly thin a series to at most `cap` points, always keeping the last."""
    if len(points) <= cap:
        return points
    step = (len(points) - 1) / (cap - 1)
    picked = [points[round(k * step)] for k in range(cap - 1)]
    picked.append(points[-1])
    # De-duplicate x values introduced by rounding, keeping order.
    out = []
    last_x = None
    for x, y in picked:
        if last_x is None or x > last_x + 1e-12:
            out.append((x, y))
            last_x = x
    return out


def classify(trophy: dict) -> str:
    """Group a trophy into a shrinkage class from its name and description."""
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    text = name + " " + desc
    if tier == "platinum" or "100% completion" in desc or "legend" in desc:
        return "completion"
    if "partner" in text:
        return "partner"
    if any(k in text for k in ("collectible", "discover", "explore all", "every spot",
                               "every location", "purchase all", "register",
                               "meditate", "card sets", "100 locations")):
        return "collectible"
    if any(k in text for k in ("rob", "laundering", "earn $", "spend $",
                               "smash", "take down", "bank", "heist")):
        return "crime"
    if any(k in text for k in ("stunt", "race", "challenge", "medal", "win ",
                               "arcade", "fishing", "hunt", "bowl", "activities")):
        return "activity"
    if desc.startswith("complete"):
        return "story"
    return "other"


def distinct_state_indices(history: list[dict]) -> list[int]:
    """Indices of the reads at which the page actually changed.

    Exophase refreshes on its own schedule, so consecutive reads are often the
    same page repeated. Trend-fitting the repeats would mistake the recorder's
    clock for information; only a read whose players count or rates differ from
    the previous kept read is a new observation.
    """
    idx = []
    for i, read in enumerate(history):
        if idx:
            prev = history[idx[-1]]
            if prev.get("players_tracked") == read.get("players_tracked") and \
                    prev.get("rates") == read.get("rates"):
                continue
        idx.append(i)
    return idx


def shrink(slope: float, class_slope: float, n_obs: int) -> float:
    """Hierarchical shrinkage of a per-trophy slope toward its class median."""
    w = n_obs / (n_obs + 5.0)
    blended = w * slope + (1.0 - w) * class_slope
    return max(-MAX_SLOPE, min(MAX_SLOPE, blended))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    times = [parse_time(r["slot"]) for r in history]
    t0 = times[0]
    hours = [(t - t0) / 3600.0 for t in times]
    xs_all = [math.log1p(h) for h in hours]
    x_last = xs_all[-1]
    x_target = math.log1p(hours[-1] + case["horizon_minutes"] / 60.0)
    dx = max(0.0, x_target - x_last)

    state_idx = distinct_state_indices(history)
    state_xs = [xs_all[i] for i in state_idx]
    state_reads = [history[i] for i in state_idx]
    n_states = len(state_reads)
    span_hours = hours[-1] - hours[0] if len(hours) > 1 else 0.0

    last = history[-1]
    players_last = last.get("players_tracked") or 0
    for read in reversed(history):
        if read.get("players_tracked"):
            players_last = read["players_tracked"]
            break

    # --- Global denominator trend: log(players tracked) against log-time. ---
    p_points = [(x, math.log(r["players_tracked"]))
                for x, r in zip(state_xs, state_reads) if r.get("players_tracked")]
    if len(p_points) >= 2 and players_last > 0:
        p_fit = thin(p_points, MAX_FIT_POINTS)
        p_slope, _, _ = theil_sen([p[0] for p in p_fit], [p[1] for p in p_fit])
        p_slope = max(0.0, min(2.5, p_slope))
    else:
        p_slope = 0.0
    log_p_hat = math.log(players_last) + p_slope * dx if players_last > 0 else 0.0
    if players_last > 0:
        # The tracked base cannot shrink, and a >20x jump in one horizon is not
        # a forecast, it is a fantasy: cap the growth factor.
        log_p_hat = min(log_p_hat, math.log(players_last) + math.log(20.0))
        log_p_hat = max(log_p_hat, math.log(players_last * 0.98))
    p_hat = math.exp(log_p_hat) if players_last > 0 else 0.0

    # --- Per-trophy raw fits over the distinct refresh states. ---
    classes = [classify(t) for t in trophies]
    raw = []
    for i in range(len(trophies)):
        pts_logit = []
        pts_earn = []
        last_rate = None
        for x, read in zip(state_xs, state_reads):
            rate = read["rates"][i] if i < len(read.get("rates") or []) else None
            if rate is None:
                continue
            last_rate = rate
            pts_logit.append((x, logit(rate)))
            players = read.get("players_tracked")
            if players:
                pts_earn.append((x, math.log(rate / 100.0 * players + 0.5)))
        if last_rate is None:
            for read in reversed(history):
                rate = read["rates"][i] if i < len(read.get("rates") or []) else None
                if rate is not None:
                    last_rate = rate
                    break
        fit_pts = thin(pts_logit, MAX_FIT_POINTS)
        if len(fit_pts) >= 2:
            slope, intercept, slope_iqr = theil_sen([p[0] for p in fit_pts],
                                                    [p[1] for p in fit_pts])
            # Log-time adoption curves bend: the recent half's slope is the
            # better guide to the next stretch than the whole-history average.
            mid_x = 0.5 * (fit_pts[0][0] + fit_pts[-1][0])
            recent = [p for p in fit_pts if p[0] >= mid_x]
            if len(recent) >= 3:
                r_slope, _, _ = theil_sen([p[0] for p in recent], [p[1] for p in recent])
                slope = 0.55 * slope + 0.45 * r_slope
            resid = [y - (intercept + slope * x) for x, y in fit_pts]
            resid_sd = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
        else:
            slope, slope_iqr, resid_sd = 0.0, 0.0, 0.0
        earn_pts = thin(pts_earn, MAX_FIT_POINTS)
        if len(earn_pts) >= 2:
            e_slope, _, _ = theil_sen([p[0] for p in earn_pts], [p[1] for p in earn_pts])
            mid_x = 0.5 * (earn_pts[0][0] + earn_pts[-1][0])
            recent = [p for p in earn_pts if p[0] >= mid_x]
            if len(recent) >= 3:
                r_slope, _, _ = theil_sen([p[0] for p in recent], [p[1] for p in recent])
                e_slope = 0.55 * e_slope + 0.45 * r_slope
        else:
            e_slope = 0.0
        raw.append({
            "last_rate": last_rate, "slope": slope, "slope_iqr": slope_iqr,
            "resid_sd": resid_sd, "earn_slope": e_slope, "n": len(pts_logit),
        })

    # Class medians of the raw slopes: the hierarchical prior each trophy's own
    # slope is shrunk toward.
    class_slope: dict[str, float] = {}
    class_earn: dict[str, float] = {}
    for cls in set(classes):
        members = [raw[i] for i in range(len(trophies)) if classes[i] == cls and raw[i]["n"] >= 2]
        class_slope[cls] = median([m["slope"] for m in members]) if members else 0.0
        class_earn[cls] = median([m["earn_slope"] for m in members]) if members else 0.0
    all_last = [r["last_rate"] for r in raw if r["last_rate"] is not None]
    fallback_rate = median(all_last) if all_last else 1.0

    # A page that never changed in the whole history is stale, not stable: the
    # next refresh replaces a pre-launch-scale tracked pool with the launch
    # crowd, and earn rates dilute hard. Price that in as a prior shift.
    if n_states <= 1:
        stale = min(1.0, 0.3 + span_hours / 8.0)
        h_factor = min(1.0, 0.5 + 0.9 * math.sqrt(horizon_days))
        flat_shift = -1.35 * stale * h_factor
    else:
        flat_shift = 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        info = raw[i]
        r_last = info["last_rate"] if info["last_rate"] is not None else fallback_rate
        base_logit = logit(r_last)
        cls = classes[i]

        if r_last >= 99.9:
            # A rate pinned at the top has nowhere to go but (mildly) down; the
            # page's 100% rows behaved as fixtures through the first refresh.
            med_logit = base_logit
            sigma = min(0.6, 0.3 + SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days))
        elif n_states <= 1:
            med_logit = base_logit + flat_shift
            sigma = math.sqrt((SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                              + REFRESH_RISK_FLAT ** 2)
        else:
            slope_dir = shrink(info["slope"], class_slope[cls], info["n"])
            logit_direct = base_logit + slope_dir * dx
            # Dilution model: trend earners and the denominator separately.
            logit_earn = None
            if players_last > 0 and p_hat > 0:
                e_slope = shrink(info["earn_slope"], class_earn[cls], info["n"])
                e_last = r_last / 100.0 * players_last
                e_hat = (e_last + 0.5) * math.exp(max(-1.0, min(3.0, e_slope)) * dx) - 0.5
                e_hat = max(e_hat, e_last * 0.85)          # counts do not fall
                e_hat = min(e_hat, p_hat * 0.999)          # nor exceed the pool
                if e_hat > 0:
                    logit_earn = logit(100.0 * e_hat / p_hat)
            if logit_earn is not None:
                med_logit = 0.5 * logit_direct + 0.5 * logit_earn
            else:
                med_logit = logit_direct
            med_logit = base_logit + max(MAX_LOGIT_FALL,
                                         min(MAX_LOGIT_RISE, med_logit - base_logit))
            # Each uncertainty component is capped: on a bending log-time curve
            # the line residuals and the pairwise-slope spread measure the
            # bend as much as the noise, and uncapped they would swamp the
            # random-walk term and pay pinball at every trophy for spread the
            # model does not actually need.
            slope_unc = min(0.4, 0.35 * info["slope_iqr"] * dx)
            resid_term = min(0.45, 0.6 * info["resid_sd"])
            refresh = REFRESH_RISK_FLAT if n_states <= 2 else REFRESH_RISK_SEEN
            sigma = math.sqrt((SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                              + resid_term ** 2 + slope_unc ** 2 + refresh ** 2)
        sigma = max(0.28, min(1.45, sigma))
        qs = [round4(inv_logit(med_logit + z * sigma)) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
