"""Weather-driven disease progression model.

Severity S is the fraction of leaf area infected (0..1). It follows logistic growth:

    dS/dt = r_eff * S * (1 - S)

r_eff changes every day with the weather:

    r_eff = r_base * f_temp(T) * f_humidity(RH) * f_rain(rain)

    f_temp      bell curve around the disease's optimum temperature
    f_humidity  S-curve that switches on above the disease's humidity threshold
    f_rain      1 + rain_boost * (rain, capped at 20 mm) / 20   (splash and leaf wetness)

With r held constant over one day the logistic equation has an exact solution, so we
step day by day with no numerical-stability issues:

    S(t+1) = 1 / (1 + ((1 - S) / S) * exp(-r))

Statuses: healthy, low_risk, act_by (spray by a given day), act_now (even spraying today
won't keep damage under the limit), spread_warning (damage will pass the limit but there
is no spray data for this disease yet).

Treatment: while a spray is protecting the crop, r_eff is multiplied by (1 - efficacy).
Protection lasts `protection_days`; sprays repeat at that interval (as product labels
advise). Non-rainfast products are washed off by a day with >= WASHOFF_MM of rain.

All constants live in data/diseases.csv and data/pesticides.csv so the team can tune
them without touching code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

WASHOFF_MM = 10.0
S_MIN, S_MAX = 1e-6, 1 - 1e-6


@dataclass
class DiseaseParams:
    r_base: float
    t_opt: float
    t_width: float
    h_thresh: float
    h_scale: float
    rain_boost: float
    economic_threshold: float

    @classmethod
    def from_row(cls, row: dict) -> "DiseaseParams":
        return cls(**{k: float(row[k]) for k in cls.__dataclass_fields__})


@dataclass
class Treatment:
    efficacy: float          # 0..1 reduction of the growth rate while protected
    protection_days: int
    rainfast: bool

    @classmethod
    def from_product(cls, p: dict) -> "Treatment":
        return cls(float(p["efficacy"]), int(p["protection_days"]), bool(p["rainfast"]))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def growth_rate(p: DiseaseParams, temp: float, humidity: float, rain_mm: float = 0.0) -> float:
    """Daily logistic growth rate for the given weather. Inputs are clamped to sane ranges."""
    if p.r_base <= 0:
        return 0.0
    temp = _clamp(float(temp), -10.0, 50.0)
    humidity = _clamp(float(humidity), 0.0, 100.0)
    rain_mm = _clamp(float(rain_mm), 0.0, 500.0)

    f_temp = math.exp(-(((temp - p.t_opt) / max(p.t_width, 0.1)) ** 2))
    f_hum = 0.05 + 0.95 / (1.0 + math.exp(-(humidity - p.h_thresh) / max(p.h_scale, 0.1)))
    f_rain = 1.0 + p.rain_boost * min(rain_mm, 20.0) / 20.0
    return p.r_base * f_temp * f_hum * f_rain


def logistic_step(s: float, r: float) -> float:
    s = _clamp(s, S_MIN, S_MAX)
    return 1.0 / (1.0 + ((1.0 - s) / s) * math.exp(-r))


def simulate(
    p: DiseaseParams,
    weather_days: list[dict],
    s0: float,
    treatment: Treatment | None = None,
    spray_day: int | None = None,
    repeat: bool = True,
) -> list[float]:
    """Severity at the start of each day: len(weather_days) + 1 values (day 0..N)."""
    s = _clamp(s0, 0.0, S_MAX)
    severities = [s]
    protected_until = -1  # exclusive day index
    next_spray = spray_day if treatment is not None else None

    for d, w in enumerate(weather_days):
        if next_spray is not None and d == next_spray:
            protected_until = d + treatment.protection_days
            next_spray = d + treatment.protection_days if repeat else None

        r = growth_rate(p, w["temp"], w["humidity"], w.get("rain_mm", 0.0))
        if treatment is not None and d < protected_until:
            r *= 1.0 - treatment.efficacy
            if not treatment.rainfast and w.get("rain_mm", 0.0) >= WASHOFF_MM:
                protected_until = d + 1  # washed off: unprotected from tomorrow

        s = logistic_step(s, r) if s > 0 else 0.0
        severities.append(s)
    return severities


def risk_level(r: float) -> str:
    if r < 0.08:
        return "low"
    if r < 0.20:
        return "moderate"
    return "high"


def run_scenarios(
    p: DiseaseParams,
    weather_days: list[dict],
    s0: float,
    treatment: Treatment | None,
    max_delay: int = 10,
) -> dict:
    """Everything the chart needs: no-action curve, a treated curve for every spray day
    0..max_delay (so the UI slider is instant), and the action window."""
    horizon = len(weather_days)
    max_delay = max(0, min(max_delay, horizon - 1))
    threshold = p.economic_threshold

    no_action = simulate(p, weather_days, s0)
    treated: dict[str, list[float]] = {}
    if treatment is not None:
        for d in range(max_delay + 1):
            treated[str(d)] = simulate(p, weather_days, s0, treatment, spray_day=d)

    daily_r = [growth_rate(p, w["temp"], w["humidity"], w.get("rain_mm", 0.0)) for w in weather_days]
    first_cross = next((i for i, s in enumerate(no_action) if s >= threshold), None)

    latest_ok = None
    for d in range(max_delay + 1):
        curve = treated.get(str(d))
        if curve is not None and max(curve) <= threshold:
            latest_ok = d

    if p.r_base <= 0:
        status = "healthy"
    elif max(no_action) <= threshold:
        status = "low_risk"
    elif treatment is None:
        status = "spread_warning"  # forecast known, but no spray data for this disease yet
    elif latest_ok is not None:
        status = "act_by"
    else:
        status = "act_now"

    def r4(xs):
        return [round(x, 4) for x in xs]

    return {
        "horizon_days": horizon,
        "threshold": threshold,
        "start_severity": round(s0, 4),
        "no_action": r4(no_action),
        "treated": {k: r4(v) for k, v in treated.items()},
        "daily_growth_rate": [round(r, 4) for r in daily_r],
        "risk_today": risk_level(daily_r[0]) if daily_r else "low",
        "action": {
            "status": status,
            "latest_spray_day": latest_ok,
            "untreated_crosses_threshold_on_day": first_cross,
            "final_severity_no_action": round(no_action[-1], 4),
            "final_severity_spray_today": round(treated["0"][-1], 4) if "0" in treated else None,
        },
    }
