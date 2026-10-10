"""Central settings. Everything can be overridden with environment variables or a .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "cropcurve_model.onnx"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", ROOT / "models" / "labels.json"))
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", ROOT / "models" / "samples"))

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
WEATHER_CACHE_MINUTES = int(os.getenv("WEATHER_CACHE_MINUTES", "30"))

# Below this top-1 probability the UI shows the top-2 possibilities instead of one answer.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "14"))
MAX_SPRAY_DELAY = int(os.getenv("MAX_SPRAY_DELAY", "10"))

# Starting severity (fraction of leaf area infected) for the farmer's answer to
# "how much of the crop is affected?"
EXTENT_SEVERITY = {
    "few": 0.02,       # a few spots on some leaves
    "several": 0.08,   # several leaves affected
    "widespread": 0.20,  # many plants / most leaves affected
}

# Defaults used when the farmer picks manual weather but leaves a field empty.
MANUAL_DEFAULTS = {"temp": 26.0, "humidity": 75.0, "rain_mm": 0.0}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
