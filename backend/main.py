"""CropCurve API.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, data_store
from .classifier import Classifier, ImageError
from .recommend import recommend
from .simulation import DiseaseParams, Treatment, growth_rate, risk_level, run_scenarios
from .weather import WeatherError, live_weather, manual_days

log = logging.getLogger("uvicorn.error")
classifier = Classifier()

Extent = Literal["few", "several", "widespread"]


def _startup_checks():
    for w in data_store.check_consistency(classifier.classes if classifier.loaded else None):
        log.warning("Data check: %s", w)
    if not classifier.loaded:
        log.warning("No trained model found at %s. Running in DEMO MODE. %s",
                    config.MODEL_PATH, classifier.load_error or "")
    if not config.OPENWEATHER_API_KEY:
        log.warning("OPENWEATHER_API_KEY not set. Live weather disabled; manual weather only.")


@asynccontextmanager
async def lifespan(_app):
    _startup_checks()
    yield


app = FastAPI(title="CropCurve API", version="0.1.0", lifespan=lifespan)


def build_forecast(disease_id: str, extent: str, weather: dict) -> dict:
    """Disease + weather -> recommendation + simulated curves. Shared by /analyze and /simulate.
    `simulation` is None for healthy leaves and for diseases without forecast parameters yet."""
    disease = data_store.get_disease(disease_id)
    days = weather["days"]

    sim, risk = None, "unknown"
    params = DiseaseParams.from_row(disease) if disease["has_forecast"] else None
    if params:
        risk = risk_level(growth_rate(params, days[0]["temp"], days[0]["humidity"], days[0].get("rain_mm", 0)))
    rec = recommend(disease_id, extent, risk, days)
    if params:
        treatment = Treatment.from_product(rec["primary"]) if rec["primary"] else None
        sim = run_scenarios(params, days, config.EXTENT_SEVERITY[extent], treatment,
                            max_delay=config.MAX_SPRAY_DELAY)

    return {
        "disease": {
            **data_store.summary(disease),
            "pathogen": disease["pathogen"], "symptoms": disease["symptoms"],
            "favoured_by": disease["favoured_by"],
        },
        "extent": extent,
        "recommendation": rec,
        "simulation": sim,
    }


async def resolve_weather(mode, lat, lon, city, temp, humidity, rain_mm) -> dict:
    horizon = config.FORECAST_DAYS
    warning = None
    if mode == "auto":
        try:
            return await live_weather(lat, lon, city, horizon)
        except WeatherError as e:
            warning = f"{e} Using manual weather instead."
    t = temp if temp is not None else config.MANUAL_DEFAULTS["temp"]
    h = humidity if humidity is not None else config.MANUAL_DEFAULTS["humidity"]
    r = rain_mm if rain_mm is not None else config.MANUAL_DEFAULTS["rain_mm"]
    return {"source": "manual", "location": None, "days": manual_days(t, h, r, horizon),
            "live_days": 0, "warning": warning}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier.loaded,
        "model_error": classifier.load_error,
        "n_classes": len(classifier.classes),
        "weather_key_configured": bool(config.OPENWEATHER_API_KEY),
        "forecast_days": config.FORECAST_DAYS,
        "max_spray_delay": config.MAX_SPRAY_DELAY,
    }


@app.get("/api/diseases")
def list_diseases():
    return [data_store.summary(d) for d in data_store.diseases().values()]


@app.get("/api/samples")
def list_samples():
    """Reference leaf photos written by ml/train.py into models/samples/ (named <disease_id>__<n>.jpg)."""
    out = []
    if config.SAMPLES_DIR.is_dir():
        for f in sorted(config.SAMPLES_DIR.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            did = f.stem.split("__")[0]
            d = data_store.diseases().get(did)
            if d:
                out.append({"url": f"/samples/{f.name}", "file": f.name, **data_store.summary(d)})
    return out


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    extent: Extent = Form("few"),
    weather_mode: Literal["auto", "manual"] = Form("auto"),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    city: str | None = Form(None),
    temp: float | None = Form(None),
    humidity: float | None = Form(None),
    rain_mm: float | None = Form(None),
):
    data = await image.read()
    if not data:
        raise HTTPException(400, "No photo received. Choose a photo of the affected leaf.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The photo is larger than 10 MB. Use a smaller photo.")
    try:
        diagnosis = classifier.predict(data, image.filename or "")
    except ImageError as e:
        raise HTTPException(400, str(e))

    weather = await resolve_weather(weather_mode, lat, lon, (city or "").strip() or None, temp, humidity, rain_mm)
    try:
        forecast = build_forecast(diagnosis["disease_id"], extent, weather)
    except KeyError as e:
        raise HTTPException(500, f"Diagnosis returned a disease with no parameter data: {e}")
    return {"diagnosis": diagnosis, "weather": weather, **forecast}


class WeatherDay(BaseModel):
    date: str
    temp: float
    humidity: float
    rain_mm: float = 0.0
    temp_max: float | None = None
    extrapolated: bool = False


class SimulateRequest(BaseModel):
    disease_id: str
    extent: Extent = "few"
    weather_days: list[WeatherDay] = Field(min_length=2)


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Re-run the forecast for another disease (e.g. the farmer picks the second guess)
    without re-uploading the photo."""
    try:
        weather = {"days": [d.model_dump() for d in req.weather_days]}
        return build_forecast(req.disease_id, req.extent, weather)
    except KeyError as e:
        raise HTTPException(404, str(e))


# Sample leaf photos and the frontend (mounted last so /api routes win)
config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=config.SAMPLES_DIR), name="samples")
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
