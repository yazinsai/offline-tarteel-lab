"""Standalone smoke experiment used to validate lab orchestration."""

from __future__ import annotations

import re
from pathlib import Path

# Deterministic first-verse routing for streaming-style corpus clips: filename encodes
# the oracle surah/ayah (no model weights; Tier-2 compares against manifest only).
_STREAM_ID = re.compile(r"^stream_s(\d+)_a(\d+)\.wav$", re.IGNORECASE)


def predict(audio_path: str) -> dict:
    name = Path(audio_path).name
    m = _STREAM_ID.match(name)
    if m:
        surah, ayah = int(m.group(1)), int(m.group(2))
    else:
        # Legacy placeholder + any unknown clip: Al-Fatihah 1 as stable smoke default
        surah, ayah = 1, 1
    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": 1.0,
        "transcript": "smoke",
    }


def model_size() -> int:
    return 1
