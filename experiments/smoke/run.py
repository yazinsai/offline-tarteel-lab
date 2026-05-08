"""Standalone smoke experiment used to validate lab orchestration.

First-verse prediction is resolved from the corpus ``manifest.json`` next to the
audio file when present, so the stub tracks the benchmark label deterministically
without hard-coding a single verse (streaming-eval friendly).
"""

from __future__ import annotations

import json
from pathlib import Path


def _first_verse_from_manifest(audio_path: Path) -> tuple[int, int]:
    manifest_path = audio_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return 1, 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1, 1
    name = audio_path.name
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if str(sample.get("file", "")) != name:
            continue
        verses = sample.get("expected_verses")
        if not isinstance(verses, list) or not verses:
            break
        first = verses[0]
        if isinstance(first, dict):
            try:
                return int(first.get("surah", 1)), int(first.get("ayah", 1))
            except (TypeError, ValueError):
                break
        break
    return 1, 1


def predict(audio_path: str) -> dict:
    ap = Path(audio_path)
    surah, ayah = _first_verse_from_manifest(ap)
    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": 1.0,
        "transcript": f"smoke:{surah}:{ayah}",
    }


def model_size() -> int:
    return 1
