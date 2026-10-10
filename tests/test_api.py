"""End-to-end API test (needs fastapi + httpx installed). Run: pytest -q"""
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)


def _jpeg():
    a = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG")
    return buf.getvalue()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and "model_loaded" in r.json()


def test_analyze_manual_weather():
    r = client.post("/api/analyze",
                    files={"image": ("early_blight.jpg", _jpeg(), "image/jpeg")},
                    data={"extent": "several", "weather_mode": "manual", "temp": "25", "humidity": "90", "rain_mm": "2"})
    assert r.status_code == 200, r.text
    body = r.json()
    sim = body["simulation"]
    assert len(sim["no_action"]) == len(body["weather"]["days"]) + 1
    assert "0" in sim["treated"]
    assert body["recommendation"]["primary"] is not None


def test_disease_without_forecast_still_diagnoses():
    r = client.post("/api/analyze",
                    files={"image": ("AppleScab1.JPG", _jpeg(), "image/jpeg")},
                    data={"weather_mode": "manual", "temp": "20", "humidity": "85", "rain_mm": "0"})
    assert r.status_code == 200, r.text
    body = r.json()
    if body["diagnosis"]["mode"] == "demo":
        assert body["disease"]["id"] == "apple_apple_scab"
        assert body["simulation"] is None


def test_catalogue_and_samples():
    assert len(client.get("/api/diseases").json()) >= 38
    assert isinstance(client.get("/api/samples").json(), list)


def test_simulate_endpoint_and_bad_inputs():
    days = [{"date": f"2026-10-{d:02d}", "temp": 19, "humidity": 92, "rain_mm": 5} for d in range(1, 15)]
    r = client.post("/api/simulate", json={"disease_id": "tomato_late_blight", "extent": "few", "weather_days": days})
    assert r.status_code == 200
    assert client.post("/api/simulate", json={"disease_id": "nope", "weather_days": days}).status_code == 404
    bad = client.post("/api/analyze", files={"image": ("x.jpg", b"not an image", "image/jpeg")},
                      data={"weather_mode": "manual"})
    assert bad.status_code == 400
