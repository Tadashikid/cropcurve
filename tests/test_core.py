"""Core tests: simulation, recommendation, weather parsing, image handling.
Run from the project root:  pytest -q
"""
import io

import numpy as np
from PIL import Image

from backend import config, data_store
from backend.classifier import Classifier, preprocess, quality_warnings
from backend.recommend import recommend
from backend.simulation import DiseaseParams, Treatment, growth_rate, run_scenarios, simulate
from backend.weather import aggregate_daily, extend_days, manual_days


def params(did):
    return DiseaseParams.from_row(data_store.get_disease(did))


def test_csvs_consistent():
    assert data_store.check_consistency(None) == []


def test_all_38_dataset_classes_present():
    d = data_store.diseases()
    assert len(d) >= 38
    assert all(i == data_store.slugify(i) for i in d)
    for folder in ["Corn_(maize)___Common_rust_", "Pepper,_bell___Bacterial_spot",
                   "Tomato___Spider_mites Two-spotted_spider_mite", "Grape___Esca_(Black_Measles)"]:
        assert data_store.slugify(folder) in d


def test_healthy_has_no_forecast_or_spray():
    h = data_store.get_disease("tomato_healthy")
    assert h["is_healthy"] and not h["has_forecast"]
    assert recommend("tomato_healthy", "few", "low", manual_days(25, 80, 0, 14))["primary"] is None


def test_forecast_without_products_gives_spread_warning():
    p = params("potato_late_blight")
    out = run_scenarios(p, manual_days(18, 92, 8, 14), 0.08, None)
    assert out["action"]["status"] == "spread_warning"
    assert out["treated"] == {}
    assert out["action"]["untreated_crosses_threshold_on_day"] is not None


def test_virus_gets_no_spray_message():
    rec = recommend("tomato_tomato_mosaic_virus", "few", "unknown", manual_days(25, 80, 0, 14))
    assert rec["primary"] is None and "virus" in rec["notes"][0]


def test_severity_monotonic_and_bounded():
    curve = simulate(params("tomato_late_blight"), manual_days(18, 95, 10, 14), 0.05)
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert all(0 <= s < 1 for s in curve)


def test_treatment_beats_no_action_and_earlier_beats_later():
    p = params("tomato_early_blight")
    days = manual_days(25, 90, 3, 14)
    t = Treatment(0.7, 7, False)
    none = simulate(p, days, 0.05)
    today = simulate(p, days, 0.05, t, spray_day=0)
    later = simulate(p, days, 0.05, t, spray_day=5)
    assert today[-1] < later[-1] < none[-1]


def test_late_blight_prefers_cool_wet():
    p = params("tomato_late_blight")
    assert growth_rate(p, 18, 92, 5) > 5 * growth_rate(p, 34, 45, 0)


def test_extreme_weather_is_clamped():
    p = params("tomato_early_blight")
    for t, h, r in [(-50, -10, -5), (80, 150, 9999), (float("1e9"), 50, 0)]:
        g = growth_rate(p, t, h, r)
        assert np.isfinite(g) and g >= 0


def test_rain_washes_off_protectant_but_not_rainfast():
    p = params("tomato_early_blight")
    days = manual_days(25, 90, 15, 14)  # heavy rain every day
    washes = simulate(p, days, 0.05, Treatment(0.8, 7, False), spray_day=0)
    stays = simulate(p, days, 0.05, Treatment(0.8, 7, True), spray_day=0)
    assert stays[-1] < washes[-1]


def test_action_window_statuses():
    p = params("tomato_late_blight")
    t = Treatment(0.85, 10, True)
    cool_wet = run_scenarios(p, manual_days(18, 92, 8, 14), 0.02, t)
    assert cool_wet["action"]["status"] == "act_by"
    assert cool_wet["action"]["latest_spray_day"] is not None
    hot_dry = run_scenarios(p, manual_days(34, 45, 0, 14), 0.02, t)
    assert hot_dry["action"]["status"] == "low_risk"
    heavy = run_scenarios(p, manual_days(18, 92, 8, 14), 0.20, t)
    assert heavy["action"]["status"] == "act_now"


def test_recommend_prefers_systemic_when_widespread():
    days = manual_days(25, 85, 0, 14)
    light = recommend("tomato_early_blight", "few", "moderate", days)
    heavy = recommend("tomato_early_blight", "widespread", "high", days)
    assert light["primary"]["mode"] == "protectant"
    assert heavy["primary"]["mode"] == "systemic"
    assert recommend("tomato_healthy", "few", "low", days)["primary"] is None


def test_weather_aggregation_and_extension():
    base = 1_760_000_000
    raw = {"city": {"timezone": 19800, "name": "Testpur", "country": "IN"},
           "list": [{"dt": base + i * 10800, "main": {"temp": 20 + i % 8, "humidity": 70 + i % 20, "temp_max": 22 + i % 8},
                     **({"rain": {"3h": 1.5}} if i % 5 == 0 else {})} for i in range(40)]}
    days = aggregate_daily(raw, 14)
    assert len(days) == 14
    live = [d for d in days if not d["extrapolated"]]
    assert 5 <= len(live) <= 6
    assert all(d["extrapolated"] for d in days[len(live):])
    assert len(extend_days(days[:3], 7)) == 7


def _jpeg(color=(60, 140, 60), size=(400, 300), noise=True):
    a = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if noise:
        a = np.clip(a + np.random.default_rng(0).integers(-60, 60, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG")
    return buf.getvalue()


def test_preprocess_shape():
    img = Image.open(io.BytesIO(_jpeg(size=(640, 480))))
    x = preprocess(img, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32


def test_quality_warnings_flag_blur_and_dark():
    blurry = Image.open(io.BytesIO(_jpeg(noise=False)))
    assert any("blurry" in w for w in quality_warnings(blurry))
    dark = Image.open(io.BytesIO(_jpeg(color=(10, 10, 10), noise=False)))
    assert any("dark" in w for w in quality_warnings(dark))


def test_demo_guess_matches_dataset_file_names():
    from backend.classifier import demo_guess
    classes = sorted(data_store.diseases())
    assert demo_guess("PotatoEarlyBlight1.JPG", classes) == "potato_early_blight"
    assert demo_guess("AppleCedarRust3.JPG", classes) == "apple_cedar_apple_rust"
    assert demo_guess("TomatoYellowCurlVirus2.JPG", classes) == "tomato_tomato_yellow_leaf_curl_virus"
    assert demo_guess("IMG_2041.jpg", classes) is None


def test_demo_mode_uses_filename_and_is_deterministic(tmp_path):
    clf = Classifier(tmp_path / "missing.onnx", tmp_path / "missing.json")
    assert not clf.loaded
    data = _jpeg()
    r1 = clf.predict(data, "tomato_late_blight_01.jpg")
    assert r1["disease_id"] == "tomato_late_blight" and r1["mode"] == "demo"
    assert abs(sum(t["confidence"] for t in r1["top"]) - 1) < 0.2
    assert clf.predict(data, "IMG_1234.jpg") == clf.predict(data, "IMG_1234.jpg")
    for cid in clf.classes:
        data_store.get_disease(cid)  # every demo class has parameter data
