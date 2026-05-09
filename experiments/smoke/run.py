"""Standalone smoke experiment used to validate lab orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


# Variant runtime.adaptive.FIRST_MATCH_THRESHOLD.11: slightly lower default than variant 03
# (0.018) so the fused ratio crosses first_match_lock_bar marginally sooner when env unset.
_DEFAULT_FIRST_MATCH_THRESHOLD = 0.0165
# Variant runtime.adaptive.VERSE_MATCH_THRESHOLD.12: slightly relaxed vs variant 04 (0.412) so
# verse_match_locked flips true marginally sooner on the same fused lock_confidence; tier-2
# surah/ayah still follow first-match lock only.
_DEFAULT_VERSE_MATCH_THRESHOLD = 0.408


def _first_match_threshold() -> float:
    """Synthetic streaming gate; lower values lock earlier (more aggressive inference)."""
    raw = os.environ.get("FIRST_MATCH_THRESHOLD")
    if raw is None or raw.strip() == "":
        return _DEFAULT_FIRST_MATCH_THRESHOLD
    return float(raw)


def _verse_match_threshold() -> float:
    """Higher-confidence verse alignment marker; sweep via VERSE_MATCH_THRESHOLD env."""
    raw = os.environ.get("VERSE_MATCH_THRESHOLD")
    if raw is None or raw.strip() == "":
        return _DEFAULT_VERSE_MATCH_THRESHOLD
    return float(raw)


_REF_CHUNK_SECONDS = 0.25  # calibrates windows_until_lock vs chunk duration
# Variant runtime.adaptive.chunk_seconds.09: shorter default frame than variant 01 (0.285s)
# for tier-2 smoke metadata sweeps without requiring CHUNK_SECONDS in the environment.
_DEFAULT_STREAM_CHUNK_SECONDS = 0.268
# Variant runtime.adaptive.overlap_seconds.10: hop stride = chunk - overlap shrinks as overlap grows,
# so windows_until_lock scales up deterministically without changing first-verse labels.
_DEFAULT_STREAM_OVERLAP_SECONDS = 0.071
# Variant runtime.adaptive.smoothing_window.13: default integration depth when env unset is 4 frames
# (variant 05 used 3); metadata-only multiplier on windows_until_lock (tier-2 surah/ayah unchanged).
_DEFAULT_STREAM_SMOOTHING_WINDOW = 4
# Variant runtime.adaptive.correction_hysteresis.06: extra margin on the first-match
# ratio gate so provisional (1:1) persists until lock_confidence clears thresh+hysteresis.
_DEFAULT_CORRECTION_HYSTERESIS = 0.012
# Variant runtime.adaptive.partial_match_margin.07: extra bar beyond thresh+hysteresis before
# treating lock_confidence as a full first-match lock (metadata/sweeps via PARTIAL_MATCH_MARGIN).
_DEFAULT_PARTIAL_MATCH_MARGIN = 0.008
# Variant runtime.adaptive.debounce_ms.08: virtual hold-off on streaming lock stabilization;
# scales windows_until_lock only (tier-2 surah/ayah still follow first_match_lock_bar).
_DEFAULT_DEBOUNCE_MS = 52


def _debounce_ms() -> int:
    """Milliseconds of synthetic correction debounce; sweep via DEBOUNCE_MS (non-negative int)."""
    raw = os.environ.get("DEBOUNCE_MS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_DEBOUNCE_MS
    try:
        v = int(float(raw))
    except ValueError:
        return _DEFAULT_DEBOUNCE_MS
    if v < 0:
        return 0
    return min(v, 10_000)


def _correction_hysteresis() -> float:
    """Additive slack on first-match lock; sweep via CORRECTION_HYSTERESIS (non-negative)."""
    raw = os.environ.get("CORRECTION_HYSTERESIS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_CORRECTION_HYSTERESIS
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_CORRECTION_HYSTERESIS
    return max(0.0, v)


def _partial_match_margin() -> float:
    """Extra ratio required past first_match_effective_threshold to lock; PARTIAL_MATCH_MARGIN."""
    raw = os.environ.get("PARTIAL_MATCH_MARGIN")
    if raw is None or raw.strip() == "":
        return _DEFAULT_PARTIAL_MATCH_MARGIN
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_PARTIAL_MATCH_MARGIN
    return max(0.0, v)


def _smoothing_window_frames() -> int:
    """Virtual moving-average depth for streaming lock delay; sweep via SMOOTHING_WINDOW (integers)."""
    raw = os.environ.get("SMOOTHING_WINDOW")
    if raw is None or raw.strip() == "":
        return _DEFAULT_STREAM_SMOOTHING_WINDOW
    try:
        v = int(float(raw))
    except ValueError:
        return _DEFAULT_STREAM_SMOOTHING_WINDOW
    if v < 0:
        return 0
    return min(v, 32)


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


def _aux_stable_ratio(text: str) -> float:
    """Secondary deterministic score (BLAKE2b) for dual-signal fusion (explore_diverse.v3)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _fused_lock_confidence(key: str) -> tuple[float, float, float]:
    """Return (primary, auxiliary, fused_ratio) in [0, 1).

    runtime.exploit_champion narrow mutation (v2): slightly stronger tilt toward the
    primary SHA-256 ratio than v1 (0.515/0.485) for a wider lock margin on corpus-v3
    without changing surah/ayah when still above first_match_lock_bar.
    """
    primary = _stable_ratio(key)
    aux = _aux_stable_ratio(key + "|corpus-v3-explore")
    w_primary, w_aux = 0.515, 0.485
    fused = (primary**w_primary) * (aux**w_aux)
    return primary, aux, fused


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
    ratio_primary, ratio_aux, ratio = _fused_lock_confidence(key)
    thresh = _first_match_threshold()
    verse_thresh = _verse_match_threshold()
    hyst = _correction_hysteresis()
    first_effective = thresh + hyst
    pm = _partial_match_margin()
    lock_bar = first_effective + pm
    locked = ratio + 1e-15 >= lock_bar
    verse_locked = ratio + 1e-15 >= verse_thresh
    inferred_surah, inferred_ayah = _infer_first_verse(path)
    # Before locking, hold a conservative provisional stance (short-stream default).
    surah, ayah = (inferred_surah, inferred_ayah) if locked else (1, 1)
    chunk_s = _chunk_seconds()
    overlap_s = _overlap_seconds(chunk_s)
    stride_s = max(chunk_s - overlap_s, 1e-9)
    smooth_n = _smoothing_window_frames()
    debounce_m = _debounce_ms()
    smoothing_multiplier = 1.0 + 0.035 * float(smooth_n)
    hysteresis_lock_delay_multiplier = 1.0 + 0.045 * hyst
    debounce_lock_delay_multiplier = 1.0 + debounce_m / 7200.0
    base_windows = 3 + int(ratio * 5)
    windows_until_lock = max(
        1,
        int(
            round(
                base_windows
                * (_REF_CHUNK_SECONDS / chunk_s)
                * (chunk_s / stride_s)
                * smoothing_multiplier
                * hysteresis_lock_delay_multiplier
                * debounce_lock_delay_multiplier
            ),
        ),
    )

    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": round(0.85 + 0.14 * ratio, 6),
        "transcript": "streaming-smoke",
        "streaming": {
            "mode": "deterministic_first_verse_lock_weighted_geom_exploit_v2",
            "lock_confidence_primary": round(ratio_primary, 6),
            "lock_confidence_aux": round(ratio_aux, 6),
            "lock_confidence_fusion": "weighted_geometric_0.515_0.485",
            "chunk_seconds": chunk_s,
            "overlap_seconds": overlap_s,
            "smoothing_window": smooth_n,
            "smoothing_lock_delay_multiplier": round(smoothing_multiplier, 9),
            "correction_hysteresis": round(hyst, 9),
            "partial_match_margin": round(pm, 9),
            "first_match_effective_threshold": round(first_effective, 9),
            "first_match_lock_bar": round(lock_bar, 9),
            "hysteresis_lock_delay_multiplier": round(hysteresis_lock_delay_multiplier, 9),
            "debounce_ms": debounce_m,
            "debounce_lock_delay_multiplier": round(debounce_lock_delay_multiplier, 9),
            "window_lock_stride_seconds": round(stride_s, 9),
            "window_lock_reference_chunk_seconds": _REF_CHUNK_SECONDS,
            "windows_until_lock": windows_until_lock,
            "lock_confidence": round(ratio, 6),
            "first_match_threshold": thresh,
            "first_match_locked": locked,
            "verse_match_threshold": verse_thresh,
            "verse_match_locked": verse_locked,
            "first_surah": surah,
            "first_ayah": ayah,
        },
    }


def model_size() -> int:
    return 1
