"""Loads data/diseases.csv and data/pesticides.csv once and serves lookups.

Each disease works at one of three levels, depending on what the CSVs contain:
  1. diagnosis only:           a row in diseases.csv (always)
  2. + spread forecast:        weather-response parameters filled in (r_base ... economic_threshold)
  3. + spray recommendation:   one or more rows in pesticides.csv for the disease id
Healthy classes never get a forecast or a spray.

Disease ids are the dataset folder names "slugified", e.g.
"Corn_(maize)___Common_rust_" -> "corn_maize_common_rust". ml/train.py writes the same ids
into models/labels.json, so the three sources line up.
"""
import csv
import re
from functools import lru_cache

from . import config

PARAM_FIELDS = ["r_base", "t_opt", "t_width", "h_thresh", "h_scale", "rain_boost", "economic_threshold"]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@lru_cache(maxsize=1)
def diseases() -> dict:
    out = {}
    with open(config.DATA_DIR / "diseases.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["is_healthy"] = row.get("pathogen_type", "") == "healthy" or row["id"].endswith("healthy")
            vals = [(row.get(k) or "").strip() for k in PARAM_FIELDS]
            if all(vals) and not row["is_healthy"]:
                for k, v in zip(PARAM_FIELDS, vals):
                    row[k] = float(v)
                row["has_forecast"] = True
            else:
                for k in PARAM_FIELDS:
                    row[k] = None
                row["has_forecast"] = False
            out[row["id"]] = row
    return out


@lru_cache(maxsize=1)
def pesticides() -> dict:
    out: dict[str, list] = {}
    with open(config.DATA_DIR / "pesticides.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["efficacy"] = float(row["efficacy"])
            row["protection_days"] = int(row["protection_days"])
            row["rainfast"] = row["rainfast"].strip().lower() in ("yes", "true", "1")
            out.setdefault(row["disease_id"], []).append(row)
    return out


def get_disease(disease_id: str) -> dict:
    d = diseases().get(disease_id)
    if d is None:
        raise KeyError(f"Unknown disease id '{disease_id}'. Add a row for it to data/diseases.csv.")
    return d


def products_for(disease_id: str) -> list:
    return pesticides().get(disease_id, [])


def summary(d: dict) -> dict:
    return {
        "id": d["id"], "crop": d["crop"], "crop_scientific": d.get("crop_scientific", ""),
        "name": d["display_name"], "is_healthy": d["is_healthy"],
        "pathogen_type": d.get("pathogen_type", ""),
        "has_forecast": d["has_forecast"], "has_products": bool(products_for(d["id"])),
    }


def check_consistency(model_classes: list | None) -> list[str]:
    """Human-readable warnings about mismatched ids across the CSVs and the model."""
    warnings = []
    d_ids = set(diseases())
    for pid in pesticides():
        if pid not in d_ids:
            warnings.append(f"pesticides.csv has '{pid}', which is not in diseases.csv")
    if model_classes:
        for c in model_classes:
            if c not in d_ids:
                warnings.append(f"Model class '{c}' has no row in diseases.csv")
    return warnings
