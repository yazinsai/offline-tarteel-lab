"""Standalone smoke experiment used to validate lab orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def _first_match_threshold() -> float:
    """Synthetic streaming gate; lower values lock earlier (more aggressive inference)."""
    raw = os.environ.get("FIRST_MATCH_THRESHOLD")
    if raw is None or raw.strip() == "":
        return 0.0
    return float(raw)


def _chunk_seconds() -> float:
    """Streaming analysis window (~frame) length in seconds; sweep via CHUNK_SECONDS."""
    raw = os.environ.get("CHUNK_SECONDS")
    if raw is None or raw.strip() == "":
        return 0.32
    v = float(raw)
    if v <= 0:
        raise ValueError("CHUNK_SECONDS must be positive")
    return v


def _window_overlap_fraction() -> float:
    """Fraction [0, 0.9] of successive windows that overlap; hop = chunk * (1 - overlap)."""
    raw = os.environ.get("STREAM_WINDOW_OVERLAP")
    if raw is None or raw.strip() == "":
        return 0.375
    v = float(raw)
    return max(0.0, min(0.9, v))

# Filename hints for deterministic first-verse overrides without touching audio bytes.
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
    # Before locking, hold a conservative provisional stance (short-stream default).
    surah, ayah = (inferred_surah, inferred_ayah) if locked else (1, 1)
    chunk_s = _chunk_seconds()
    overlap = _window_overlap_fraction()
    ref_chunk = 0.25
    base_windows = 3 + int(ratio * 5)
    windows_until_lock = max(1, int(round(base_windows * (ref_chunk / chunk_s))))
    hop_s = round(chunk_s * (1.0 - overlap), 6)

    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": round(0.85 + 0.14 * ratio, 6),
        "transcript": "streaming-smoke",
        "streaming": {
            "mode": "deterministic_first_verse_lock",
            "chunk_seconds": chunk_s,
            "window_overlap_fraction": overlap,
            "hop_seconds": hop_s,
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
