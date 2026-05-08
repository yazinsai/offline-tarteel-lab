"""Standalone smoke experiment used to validate lab orchestration.

Predictions resolve from ``expected_by_basename.json`` so additional corpus rows
stay JSON-only: add a manifest entry and matching basename key here.
"""

from __future__ import annotations

import json
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent
_LOOKUP: dict[str, dict] = json.loads(
    (_EXP_DIR / "expected_by_basename.json").read_text(encoding="utf-8"),
)
_DEFAULT = {"surah": 1, "ayah": 1}


def predict(audio_path: str) -> dict:
    name = Path(audio_path).name
    spec = _LOOKUP.get(name, _DEFAULT)
    return {
        "surah": int(spec["surah"]),
        "ayah": int(spec["ayah"]),
        "ayah_end": None,
        "score": 1.0 if name in _LOOKUP else 0.99,
        "transcript": "smoke",
    }


def model_size() -> int:
    return len(_LOOKUP) + 1
