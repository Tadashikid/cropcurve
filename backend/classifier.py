"""Leaf-photo classifier.

Uses the ONNX model exported by ml/train.py (onnxruntime is light and fast on CPU, so the
backend doesn't need PyTorch). If no model file exists yet, it runs in DEMO MODE so the
frontend and simulation can be built and tested in parallel: the diagnosis comes from
the filename (e.g. "late_blight_3.jpg") or, failing that, from a hash of the image.
Every response says which mode produced it, and the UI shows a banner in demo mode.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]



def _name_tokens(text: str) -> set[str]:
    """'PotatoEarlyBlight1.JPG' / 'late_blight-2.jpg' -> {'potato','early','blight'} etc."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    return {t for t in re.split(r"[^a-z]+", text.lower()) if len(t) > 1 and t not in {"jpg", "jpeg", "png", "img", "the", "leaf"}}


def demo_guess(filename: str, classes: list[str]) -> str | None:
    """Best class for a file name in demo mode: most shared words, at least two."""
    toks = _name_tokens(filename.rsplit(".", 1)[0])
    best, best_score = None, 0.0
    for c in classes:
        ct = set(c.split("_"))
        overlap = len(toks & ct)
        score = overlap - 0.01 * len(ct)
        if overlap >= 2 and score > best_score:
            best, best_score = c, score
    return best


class ImageError(ValueError):
    """Bad upload; message is safe to show to the user."""


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phone photos are often rotated via EXIF
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ImageError("That file isn't a readable image. Upload a JPG or PNG photo of a leaf.") from e


def preprocess(img: Image.Image, size: int, mean, std) -> np.ndarray:
    """Resize shorter side to size*256/224, centre-crop to size: matches train.py eval transform."""
    resize_to = int(round(size * 256 / 224))
    w, h = img.size
    scale = resize_to / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None, ...].astype(np.float32)


def quality_warnings(img: Image.Image) -> list[str]:
    warnings = []
    w, h = img.size
    if min(w, h) < 160:
        warnings.append("The photo is very small. Move closer so one leaf fills the frame.")
    g = img.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype=np.float32)
    if a.size and a.shape[0] > 2 and a.shape[1] > 2:
        lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
        if lap.var() < 40:
            warnings.append("The photo looks blurry. Hold the phone steady and tap the leaf to focus.")
    mean = float(a.mean()) if a.size else 128
    if mean < 45:
        warnings.append("The photo is very dark. Take it in daylight.")
    elif mean > 225:
        warnings.append("The photo is overexposed. Avoid direct glare on the leaf.")
    return warnings


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Classifier:
    def __init__(self, model_path: Path = config.MODEL_PATH, labels_path: Path = config.LABELS_PATH):
        self.session = None
        from . import data_store

        self.classes = sorted(data_store.diseases())
        self.size, self.mean, self.std = 224, DEFAULT_MEAN, DEFAULT_STD
        self.load_error = None
        if Path(model_path).exists() and Path(labels_path).exists():
            try:
                import onnxruntime as ort

                meta = json.loads(Path(labels_path).read_text())
                self.classes = meta["classes"]
                self.size = int(meta.get("input_size", 224))
                self.mean = meta.get("mean", DEFAULT_MEAN)
                self.std = meta.get("std", DEFAULT_STD)
                self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:  # keep the server up; report in /api/health
                self.session = None
                self.load_error = f"{type(e).__name__}: {e}"
        elif Path(model_path).exists() or Path(labels_path).exists():
            self.load_error = "Found only one of the model file and labels.json; both are needed."

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def _probs(self, img: Image.Image, data: bytes, filename: str) -> np.ndarray:
        if self.loaded:
            x = preprocess(img, self.size, self.mean, self.std)
            logits = self.session.run(None, {self.input_name: x})[0][0]
            return _softmax(np.asarray(logits, dtype=np.float64))

        # demo mode: deterministic fake probabilities
        h = int(hashlib.sha256(data).hexdigest(), 16)
        top = demo_guess(filename, self.classes)
        conf = 0.88 if top else 0.45 + (h % 50) / 100
        if top is None:
            top = self.classes[h % len(self.classes)]
        probs = np.full(len(self.classes), (1 - conf) / (len(self.classes) - 1))
        probs[self.classes.index(top)] = conf
        # make the runner-up clearly second so the low-confidence path is testable
        second = (self.classes.index(top) + 1 + h % (len(self.classes) - 1)) % len(self.classes)
        bump = min(probs.sum() - conf, 0.6 * (1 - conf))
        probs[second] += bump
        probs /= probs.sum()
        return probs

    def predict(self, data: bytes, filename: str = "") -> dict:
        img = load_image(data)
        probs = self._probs(img, data, filename)
        order = np.argsort(probs)[::-1]
        top = [{"disease_id": self.classes[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]]
        return {
            "disease_id": top[0]["disease_id"],
            "confidence": top[0]["confidence"],
            "low_confidence": top[0]["confidence"] < config.CONFIDENCE_THRESHOLD,
            "top": top,
            "quality_warnings": quality_warnings(img),
            "mode": "model" if self.loaded else "demo",
        }
