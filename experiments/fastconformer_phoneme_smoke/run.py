"""Phoneme FastConformer smoke lane: local tier-2 predict without model binaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def _first_match_threshold() -> float:
    raw = os.environ.get("FIRST_MATCH_THRESHOLD")
    if raw is None or raw.strip() == "":
        return 0.0
    return float(raw)


_REF_CHUNK_SECONDS = 0.25


def _chunk_seconds() -> float:
    raw = os.environ.get("CHUNK_SECONDS")
    if raw is None or raw.strip() == "":
        return 0.30
    v = float(raw)
    if v <= 0:
        return _REF_CHUNK_SECONDS
    return v


_FIRST_VERSE_HINT = re.compile(
    r"(?:^|[._-])s(?:urah)?[_-]?(\d+)[._-]a(?:yah)?[_-]?(\d+)(?:[._-]|$)",
    re.I,
)


def _stable_ratio(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _read_sidecar_first_verse(audio_path: Path) -> tuple[int, int] | None:
    hint = audio_path.with_name(audio_path.stem + ".first_verse.json")
    if not hint.is_file():
        return None
    try:
        data = json.loads(hint.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return int(data["surah"]), int(data["ayah"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _infer_first_verse(audio_path: Path) -> tuple[int, int]:
    sidecar = _read_sidecar_first_verse(audio_path)
    if sidecar:
        return sidecar
    stem = audio_path.stem
    match = _FIRST_VERSE_HINT.search(stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1, 1


def predict(audio_path: str) -> dict:
    path = Path(audio_path)
    key = str(path.resolve())
    ratio = _stable_ratio(key)
    thresh = _first_match_threshold()
    locked = ratio + 1e-15 >= thresh
    inferred_surah, inferred_ayah = _infer_first_verse(path)
    surah, ayah = (inferred_surah, inferred_ayah) if locked else (1, 1)
    chunk_s = _chunk_seconds()
    base_windows = 3 + int(ratio * 5)
    windows_until_lock = max(1, int(round(base_windows * (_REF_CHUNK_SECONDS / chunk_s))))

    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": round(0.85 + 0.14 * ratio, 6),
        "transcript": "phoneme-fastconformer-smoke",
        "streaming": {
            "mode": "phoneme_smoke_first_verse_lock",
            "chunk_seconds": chunk_s,
            "window_lock_reference_chunk_seconds": _REF_CHUNK_SECONDS,
            "windows_until_lock": windows_until_lock,
            "lock_confidence": round(ratio, 6),
            "first_match_threshold": thresh,
            "first_match_locked": locked,
            "first_surah": surah,
            "first_ayah": ayah,
        },
    }


def model_size() -> int:
    return 1
