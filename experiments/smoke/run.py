"""Standalone smoke experiment used to validate lab orchestration."""

from __future__ import annotations


def predict(audio_path: str) -> dict:
    return {
        "surah": 1,
        "ayah": 1,
        "ayah_end": None,
        "score": 1.0,
        "transcript": "smoke",
    }


def model_size() -> int:
    return 1
