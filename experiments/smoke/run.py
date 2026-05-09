"""Standalone smoke experiment used to validate lab orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


# Variant runtime.adaptive.FIRST_MATCH_THRESHOLD.03: mild hold-off until lock_confidence
# reaches this level when env is unset; sweeps still use FIRST_MATCH_THRESHOLD.
_DEFAULT_FIRST_MATCH_THRESHOLD = 0.12


def _first_match_threshold() -> float:
    """Synthetic streaming gate; lower values lock earlier (more aggressive inference)."""
    raw = os.environ.get("FIRST_MATCH_THRESHOLD")
    if raw is None or raw.strip() == "":
        return _DEFAULT_FIRST_MATCH_THRESHOLD
    return float(raw)


_REF_CHUNK_SECONDS = 0.25  # calibrates windows_until_lock vs chunk duration
# Variant runtime.adaptive.chunk_seconds.01: slightly shorter default frame than 0.30s
# for tier-2 smoke metadata sweeps without requiring CHUNK_SECONDS in the environment.
_DEFAULT_STREAM_CHUNK_SECONDS = 0.285
# Variant runtime.adaptive.overlap_seconds.02: hop stride = chunk - overlap shrinks as overlap grows,
# so windows_until_lock scales up deterministically without changing first-verse labels.
_DEFAULT_STREAM_OVERLAP_SECONDS = 0.062


def _chunk_seconds() -> float:
    """Decoder / streaming frame size for metadata; sweep via CHUNK_SECONDS env."""
    raw = os.environ.get("CHUNK_SECONDS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_STREAM_CHUNK_SECONDS
    v = float(raw)
    if v <= 0:
        return _REF_CHUNK_SECONDS
    return v


def _overlap_seconds(chunk_s: float) -> float:
    """Chunk overlap for streaming metadata; sweep via OVERLAP_SECONDS env."""
    raw = os.environ.get("OVERLAP_SECONDS")
    if raw is None or raw.strip() == "":
        v = _DEFAULT_STREAM_OVERLAP_SECONDS
    else:
        v = float(raw)
    if v < 0:
        return 0.0
    # Keep stride positive so lock delay stays finite.
    max_ov = max(chunk_s - 1e-9, 0.0)
    return min(v, max_ov)


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
    overlap_s = _overlap_seconds(chunk_s)
    stride_s = max(chunk_s - overlap_s, 1e-9)
    base_windows = 3 + int(ratio * 5)
    windows_until_lock = max(
        1,
        int(round(base_windows * (_REF_CHUNK_SECONDS / chunk_s) * (chunk_s / stride_s))),
    )

    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": round(0.85 + 0.14 * ratio, 6),
        "transcript": "streaming-smoke",
        "streaming": {
            "mode": "deterministic_first_verse_lock",
            "chunk_seconds": chunk_s,
            "overlap_seconds": overlap_s,
            "window_lock_stride_seconds": round(stride_s, 9),
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
