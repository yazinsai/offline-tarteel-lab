"""CTC log-likelihood rerank on top matcher verses (acoustic evidence from frame log-probs).

Extends the reference_phoneme_v4 greedy decode + Quran DB shortlist: candidates keep the same
audio→ONNX→features path, but we score each shortlist entry with a standard CTC forward pass in
log space instead of trusting string similarity alone.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np

_RUNNER = Path(__file__).resolve().parent.parent / "reference_phoneme_v4" / "run.py"
_ref_mod: Any | None = None
_VOCAB_IDX: dict[str, int] | None = None


def _ref():
    global _ref_mod
    if _ref_mod is None:
        spec = importlib.util.spec_from_file_location("_phoneme_ref_base", _RUNNER)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {_RUNNER}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ref_mod = mod
    return _ref_mod


def _vocab_idx() -> dict[str, int]:
    global _VOCAB_IDX
    if _VOCAB_IDX is None:
        ref = _ref()
        _VOCAB_IDX = {p: i for i, p in enumerate(ref.PHONEME_VOCAB)}
    return _VOCAB_IDX


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - m)
    s = np.sum(ex, axis=axis, keepdims=True)
    return np.log(ex / np.clip(s, 1e-300, None))


def _logaddexp(a: float, b: float) -> float:
    if not math.isfinite(a):
        return b
    if not math.isfinite(b):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _labels_from_joined(joined: str) -> list[int] | None:
    idx = _vocab_idx()
    out: list[int] = []
    for tok in joined.split():
        j = idx.get(tok)
        if j is None:
            return None
        out.append(j)
    return out


def _joined_for_candidate(cand: dict[str, Any]) -> str | None:
    ref = _ref()
    assert ref._verses is not None
    surah, ayah = int(cand["surah"]), int(cand["ayah"])
    ayah_end = cand.get("ayah_end")
    if ayah_end is None:
        for v in ref._verses:
            if int(v["surah"]) == surah and int(v["ayah"]) == ayah:
                j = (v.get("phonemes_joined") or "").strip()
                return j or None
        return None
    end = int(ayah_end)
    lst = ref._by_surah.get(surah, [])
    parts: list[str] = []
    for v in lst:
        a = int(v["ayah"])
        if ayah <= a <= end:
            pj = (v.get("phonemes_joined") or "").strip()
            if pj:
                parts.append(pj)
    if not parts:
        return None
    return " ".join(parts)


def _ctc_forward_log(log_lp: np.ndarray, labels: list[int], blank_id: int) -> float:
    """Total log P(labels | audio) under CTC, log_lp shape (T, C) already log-softmax."""
    ext: list[int] = [blank_id]
    for lab in labels:
        ext.append(lab)
        ext.append(blank_id)
    s_len = len(ext)
    t_len = int(log_lp.shape[0])
    neg_inf = -1.0e30
    dp = np.full((t_len, s_len), neg_inf, dtype=np.float64)
    dp[0, 0] = float(log_lp[0, ext[0]])
    if s_len > 1:
        dp[0, 1] = float(log_lp[0, ext[1]])
    for t in range(1, t_len):
        for s in range(s_len):
            c = ext[s]
            lp = float(log_lp[t, c])
            prev_best = dp[t - 1, s]
            if s > 0:
                prev_best = _logaddexp(prev_best, dp[t - 1, s - 1])
            if s > 1 and (c == blank_id or ext[s - 2] != c):
                prev_best = _logaddexp(prev_best, dp[t - 1, s - 2])
            dp[t, s] = lp + prev_best
    return _logaddexp(dp[t_len - 1, s_len - 1], dp[t_len - 1, s_len - 2])


def predict(audio_path: str) -> dict:
    ref = _ref()
    ref._ensure_loaded()
    assert ref._onnx_session is not None
    audio = ref._load_audio(audio_path)
    raw = ref._compute_logprobs(audio, ref._onnx_session)
    log_lp = _log_softmax(raw, axis=-1)
    phoneme_text = ref._greedy_decode_phonemes(raw)
    cands = ref._match_phoneme_text(phoneme_text, top_k=24)
    if not cands:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": phoneme_text}

    blank = int(ref.BLANK_ID)
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for cand in cands:
        joined = _joined_for_candidate(cand)
        lab: list[int] | None = None
        ll = -1.0e30
        if joined:
            lab = _labels_from_joined(joined)
            if lab is not None and lab:
                try:
                    ll = _ctc_forward_log(log_lp, lab, blank)
                except Exception:
                    ll = -1.0e30
        str_score = float(cand.get("score", 0.0))
        scored.append((ll, str_score, cand))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0][2]
    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": best.get("ayah_end"),
        "score": best["score"],
        "transcript": phoneme_text,
    }


def model_size() -> int:
    return _ref().model_size()
