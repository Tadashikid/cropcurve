"""Picks a pesticide for the diagnosed disease, given how far it has spread and the weather.

Scoring (simple, explainable, easy to tune):
  base            product efficacy (0..1)
  +0.15           systemic product when infection is several/widespread or spread risk is high
  +0.10           protectant when infection is light and risk isn't high (cheaper, kinder to resistance)
  +0.05 / -0.05   low / high cost
  +0.08           rainfast product when the forecast has a heavy-rain day
"""
from . import data_store
from .simulation import WASHOFF_MM

COST_ADJ = {"low": 0.05, "medium": 0.0, "high": -0.05}


def _score(p: dict, extent: str, risk: str, heavy_rain: bool) -> tuple[float, list[str]]:
    score = p["efficacy"]
    reasons = []
    heavy_infection = extent in ("several", "widespread")
    if p["mode"] == "systemic" and (heavy_infection or risk == "high"):
        score += 0.15
        reasons.append("it works inside the leaf, which matters once infection has spread"
                       if heavy_infection else "it works inside the leaf, which matters when weather favours fast spread")
    if p["mode"] == "protectant" and extent == "few" and risk != "high":
        score += 0.10
        reasons.append("infection is still light, so a low-cost protective spray is enough")
    score += COST_ADJ.get(p["cost_level"], 0.0)
    if heavy_rain and p["rainfast"]:
        score += 0.08
        reasons.append("heavy rain is forecast and this product won't wash off")
    return score, reasons


def recommend(disease_id: str, extent: str, risk: str, weather_days: list[dict]) -> dict:
    disease = data_store.get_disease(disease_id)
    if disease["is_healthy"]:
        return {"primary": None, "alternatives": [], "cultural_practices": disease["cultural_practices"],
                "notes": ["No spray needed. Spraying a healthy crop wastes money and speeds up resistance."]}

    products = data_store.products_for(disease_id)
    if not products:
        kind = disease.get("pathogen_type", "")
        if kind == "viral":
            note = "No spray cures a virus. The steps below limit its spread."
        elif kind == "pest":
            note = "This is a pest, not a disease. Spray advice for it isn't set up yet; ask your local agriculture officer."
        else:
            note = "Spray advice for this disease isn't set up yet. Ask your local agriculture officer which product to use."
        return {"primary": None, "alternatives": [], "cultural_practices": disease["cultural_practices"],
                "notes": [note]}

    heavy_rain = any(w.get("rain_mm", 0) >= WASHOFF_MM for w in weather_days[:7])
    scored = []
    for p in products:
        s, reasons = _score(p, extent, risk, heavy_rain)
        scored.append((s, reasons, p))
    scored.sort(key=lambda x: x[0], reverse=True)

    def card(item):
        s, reasons, p = item
        return {
            "product": p["product"], "active_ingredient": p["active_ingredient"], "mode": p["mode"],
            "frac_group": p["frac_group"], "typical_rate": p["typical_rate"],
            "efficacy": p["efficacy"], "protection_days": p["protection_days"], "rainfast": p["rainfast"],
            "cost_level": p["cost_level"], "notes": p["notes"], "why": reasons, "score": round(s, 3),
        }

    primary = card(scored[0])
    alternatives = [card(x) for x in scored[1:3]]
    notes = [
        f"Repeat every {primary['protection_days']} days while the weather favours the disease.",
        "Alternate between products from different groups (shown as group numbers) so the disease doesn't become resistant.",
        "Always follow the product label for dose, safety gear and the waiting period before harvest. Check with your local agriculture officer.",
    ]
    if heavy_rain and not primary["rainfast"]:
        notes.insert(0, "Heavy rain is forecast. Spray on a dry day and plan to repeat after the rain.")
    return {"primary": primary, "alternatives": alternatives,
            "cultural_practices": disease["cultural_practices"], "notes": notes}
