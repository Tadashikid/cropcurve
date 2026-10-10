"""Weather for the simulation: live OpenWeatherMap forecast, or manual values.

The free OpenWeatherMap tier gives a 5-day forecast in 3-hour steps. We average it into
daily temperature / humidity and sum rainfall. Days beyond the forecast are filled with
the forecast average and marked `extrapolated` so the UI can say so honestly.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone

from . import config

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"

_cache: dict[tuple, tuple[float, dict]] = {}


class WeatherError(Exception):
    """Raised with a message that is safe to show to the user."""


def _cache_get(key):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < config.WEATHER_CACHE_MINUTES * 60:
        return hit[1]
    return None


async def _get_json(url: str, params: dict) -> dict | list:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params)
    except httpx.HTTPError as e:
        raise WeatherError(f"Couldn't reach the weather service ({type(e).__name__}).") from e
    if resp.status_code == 401:
        raise WeatherError("The weather API key was rejected. New keys can take up to 2 hours to activate.")
    if resp.status_code == 429:
        raise WeatherError("Weather API daily limit reached.")
    if resp.status_code != 200:
        raise WeatherError(f"Weather service returned an error ({resp.status_code}).")
    return resp.json()


async def geocode(city: str, api_key: str) -> tuple[float, float, str]:
    key = ("geo", city.strip().lower())
    cached = _cache_get(key)
    if cached:
        return cached["lat"], cached["lon"], cached["name"]
    data = await _get_json(GEOCODE_URL, {"q": city, "limit": 1, "appid": api_key})
    if not data:
        raise WeatherError(f"Couldn't find a place called '{city}'. Try adding the state, e.g. 'Nashik, Maharashtra'.")
    top = data[0]
    name = ", ".join(x for x in [top.get("name"), top.get("state"), top.get("country")] if x)
    _cache[key] = (time.time(), {"lat": top["lat"], "lon": top["lon"], "name": name})
    return top["lat"], top["lon"], name


def aggregate_daily(raw: dict, horizon: int) -> list[dict]:
    """Turn a /forecast response into `horizon` daily records (local dates)."""
    tz_offset = timedelta(seconds=raw.get("city", {}).get("timezone", 0))
    buckets: "OrderedDict[date, list]" = OrderedDict()
    for item in raw.get("list", []):
        local = datetime.fromtimestamp(item["dt"], tz=timezone.utc) + tz_offset
        buckets.setdefault(local.date(), []).append(item)

    days = []
    for d, items in buckets.items():
        temps = [i["main"]["temp"] for i in items]
        hums = [i["main"]["humidity"] for i in items]
        rain = sum((i.get("rain") or {}).get("3h", 0.0) for i in items)
        days.append({
            "date": d.isoformat(),
            "temp": round(sum(temps) / len(temps), 1),
            "temp_max": round(max(i["main"].get("temp_max", i["main"]["temp"]) for i in items), 1),
            "humidity": round(sum(hums) / len(hums)),
            "rain_mm": round(rain, 1),
            "extrapolated": False,
        })
    if not days:
        raise WeatherError("The weather service returned no forecast data.")
    return extend_days(days, horizon)


def extend_days(days: list[dict], horizon: int) -> list[dict]:
    days = days[:horizon]
    if len(days) >= horizon:
        return days
    n = len(days)
    avg = {
        "temp": round(sum(d["temp"] for d in days) / n, 1),
        "humidity": round(sum(d["humidity"] for d in days) / n),
        "rain_mm": round(sum(d["rain_mm"] for d in days) / n, 1),
    }
    last = date.fromisoformat(days[-1]["date"])
    for i in range(1, horizon - n + 1):
        days.append({"date": (last + timedelta(days=i)).isoformat(), **avg,
                     "temp_max": avg["temp"], "extrapolated": True})
    return days


def manual_days(temp: float, humidity: float, rain_mm: float, horizon: int) -> list[dict]:
    today = date.today()
    return [{
        "date": (today + timedelta(days=i)).isoformat(),
        "temp": round(float(temp), 1), "temp_max": round(float(temp), 1),
        "humidity": round(float(humidity)), "rain_mm": round(float(rain_mm), 1),
        "extrapolated": False,
    } for i in range(horizon)]


async def live_weather(lat: float | None, lon: float | None, city: str | None, horizon: int) -> dict:
    key = config.OPENWEATHER_API_KEY
    if not key:
        raise WeatherError("No weather API key is set on the server.")
    name = None
    if lat is None or lon is None:
        if not city:
            raise WeatherError("Share your location or type a village or town name.")
        lat, lon, name = await geocode(city, key)

    ck = ("fc", round(lat, 2), round(lon, 2))
    raw = _cache_get(ck)
    if raw is None:
        raw = await _get_json(FORECAST_URL, {"lat": lat, "lon": lon, "appid": key, "units": "metric"})
        _cache[ck] = (time.time(), raw)

    days = aggregate_daily(raw, horizon)
    c = raw.get("city", {})
    name = name or ", ".join(x for x in [c.get("name"), c.get("country")] if x) or f"{lat:.2f}, {lon:.2f}"
    return {"source": "live", "location": name, "lat": lat, "lon": lon, "days": days,
            "live_days": sum(1 for d in days if not d["extrapolated"])}
