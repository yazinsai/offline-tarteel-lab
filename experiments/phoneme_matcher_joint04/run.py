"""Joint03 phoneme matcher plus bounded same-surah span rewind."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from Levenshtein import ratio

_DIR = Path(__file__).resolve().parent
_JOINT02 = _DIR.parent / "phoneme_matcher_joint02" / "run.py"

_spec = importlib.util.spec_from_file_location("phoneme_matcher_joint02_base", _JOINT02)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load joint02 from {_JOINT02}")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

PREFIX_MAX_SPAN = 7
PREFIX_MIN_CHARS = 34
PREFIX_MIN_SCORE = 0.50
PREFIX_MARGIN = -0.02
PREFIX_HEAD_MAX_CHARS = 96
PREFIX_HEAD_BLEND = 0.50
GLOBAL_SPAN_MIN_CHARS = 80
GLOBAL_SPAN_MIN_SCORE = 0.54
GLOBAL_SPAN_MARGIN = -0.015
GLOBAL_SPAN_SHORTLIST = 320
REWIND_MAX_START_AYAH = 4
REWIND_MIN_CHARS = 30
REWIND_MIN_SCORE = 0.50
REWIND_MARGIN = -0.13

_prefix_spans: list[tuple[int, int, str, str]] | None = None
_global_spans: list[tuple[int, int, int, str, str, set[str], set[str]]] | None = None
_global_span_index: tuple[dict[str, list[int]], dict[str, list[int]]] | None = None


def _head_alignment_score(query_ns: str, ref_ns: str) -> float:
    if not query_ns or not ref_ns:
        return 0.0
    scores = []
    for n in (8, 12, 16):
        if len(query_ns) >= n and len(ref_ns) >= n:
            scores.append(ratio(query_ns[:n], ref_ns[:n]))
    return max(scores) if scores else 0.0


def _prefix_span_table() -> list[tuple[int, int, str, str]]:
    global _prefix_spans
    if _prefix_spans is not None:
        return _prefix_spans
    spans: list[tuple[int, int, str, str]] = []
    for surah_num, verses in _base._by_surah.items():
        if not verses or int(verses[0].get("ayah", 0)) != 1:
            continue
        max_span = min(PREFIX_MAX_SPAN, len(verses))
        for span in range(2, max_span + 1):
            chunk = verses[:span]
            first = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
            span_phonemes = first + " " + " ".join(
                v.get("phonemes_joined", "") for v in chunk[1:]
            )
            spans.append((surah_num, chunk[-1]["ayah"], span_phonemes, span_phonemes.replace(" ", "")))
    _prefix_spans = spans
    return spans


def _global_span_table() -> list[tuple[int, int, int, str, str, set[str], set[str]]]:
    global _global_spans
    if _global_spans is not None:
        return _global_spans
    spans: list[tuple[int, int, int, str, str, set[str], set[str]]] = []
    for surah_num, verses in _base._by_surah.items():
        for i in range(len(verses)):
            max_span = min(PREFIX_MAX_SPAN, len(verses) - i)
            for span in range(2, max_span + 1):
                chunk = verses[i : i + span]
                first = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get(
                    "phonemes_joined", ""
                )
                span_phonemes = first + " " + " ".join(
                    v.get("phonemes_joined", "") for v in chunk[1:]
                )
                ref_ns = span_phonemes.replace(" ", "")
                spans.append(
                    (
                        surah_num,
                        chunk[0]["ayah"],
                        chunk[-1]["ayah"],
                        span_phonemes,
                        ref_ns,
                        _base._query_bigrams(ref_ns),
                        _base._query_trigrams(ref_ns),
                    )
                )
    _global_spans = spans
    return spans


def _global_span_ngram_index() -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    global _global_span_index
    if _global_span_index is not None:
        return _global_span_index
    bigrams: dict[str, list[int]] = {}
    trigrams: dict[str, list[int]] = {}
    for idx, row in enumerate(_global_span_table()):
        for gram in row[5]:
            bigrams.setdefault(gram, []).append(idx)
        for gram in row[6]:
            trigrams.setdefault(gram, []).append(idx)
    _global_span_index = (bigrams, trigrams)
    return _global_span_index


def _surah_prefix_candidates(phoneme_text: str) -> list[dict]:
    """Score global ayah-1 spans that joint02's shortlist can miss."""
    if not phoneme_text.strip() or _base._verses is None:
        return []
    no_space_text = phoneme_text.replace(" ", "")
    if len(no_space_text) < PREFIX_MIN_CHARS:
        return []

    out: list[dict] = []
    for surah_num, ayah_end, span_phonemes, ref_ns in _prefix_span_table():
        raw = ratio(phoneme_text, span_phonemes)
        frag = _base.fragment_score(no_space_text, ref_ns)
        score = max(raw, raw + (frag - raw) * _base.FRAGMENT_BLEND)
        if len(no_space_text) <= PREFIX_HEAD_MAX_CHARS:
            head = _head_alignment_score(no_space_text, ref_ns)
            if head > score:
                score = max(score, score + (head - score) * PREFIX_HEAD_BLEND)
        if score < PREFIX_MIN_SCORE:
            continue
        out.append(
            {
                "surah": surah_num,
                "ayah": 1,
                "ayah_end": ayah_end,
                "score": round(score, 4),
                "_prefix_rescue": True,
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:12]


def _global_span_candidates(phoneme_text: str) -> list[dict]:
    if not phoneme_text.strip() or _base._verses is None:
        return []
    no_space_text = phoneme_text.replace(" ", "")
    if len(no_space_text) < GLOBAL_SPAN_MIN_CHARS:
        return []

    qb = _base._query_bigrams(no_space_text)
    qt = _base._query_trigrams(no_space_text)
    bigram_index, trigram_index = _global_span_ngram_index()
    rough_scores: dict[int, float] = {}
    for gram in qb:
        for idx in bigram_index.get(gram, []):
            rough_scores[idx] = rough_scores.get(idx, 0.0) + 1.0
    for gram in qt:
        for idx in trigram_index.get(gram, []):
            rough_scores[idx] = rough_scores.get(idx, 0.0) + 0.48
    rough = sorted(rough_scores.items(), key=lambda x: (-x[1], x[0]))

    out: list[dict] = []
    table = _global_span_table()
    for idx, _ in rough[:GLOBAL_SPAN_SHORTLIST]:
        surah_num, ayah, ayah_end, span_phonemes, ref_ns, _rb, _rt = table[idx]
        raw = ratio(phoneme_text, span_phonemes)
        frag = _base.fragment_score(no_space_text, ref_ns)
        score = max(raw, raw + (frag - raw) * _base.FRAGMENT_BLEND)
        if score < GLOBAL_SPAN_MIN_SCORE:
            continue
        out.append(
            {
                "surah": surah_num,
                "ayah": ayah,
                "ayah_end": ayah_end,
                "score": round(score, 4),
                "_global_span_rescue": True,
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:12]


def _same_surah_rewind_candidate(phoneme_text: str, best: dict) -> dict | None:
    if not phoneme_text.strip() or _base._verses is None:
        return None
    ayah_end = best.get("ayah_end")
    if ayah_end is None:
        return None
    surah_num = int(best.get("surah", 0))
    ayah = int(best.get("ayah", 0))
    ayah_end = int(ayah_end)
    if ayah <= 1 or ayah > REWIND_MAX_START_AYAH:
        return None
    no_space_text = phoneme_text.replace(" ", "")
    if len(no_space_text) < REWIND_MIN_CHARS:
        return None

    verses = _base._by_surah.get(surah_num, [])
    if ayah_end > len(verses):
        return None
    chunk = verses[:ayah_end]
    if not chunk:
        return None
    first = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
    span_phonemes = first + " " + " ".join(
        v.get("phonemes_joined", "") for v in chunk[1:]
    )
    ref_ns = span_phonemes.replace(" ", "")
    raw = ratio(phoneme_text, span_phonemes)
    frag = _base.fragment_score(no_space_text, ref_ns)
    score = max(raw, raw + (frag - raw) * _base.FRAGMENT_BLEND)
    best_score = float(best.get("score", 0.0))
    if score < REWIND_MIN_SCORE or score < best_score + REWIND_MARGIN:
        return None
    return {
        "surah": surah_num,
        "ayah": 1,
        "ayah_end": ayah_end,
        "score": round(score, 4),
        "_same_surah_rewind": True,
    }


def _best_match_for_hypothesis(phoneme_text: str) -> dict | None:
    top = _base._match_phoneme_text(phoneme_text, top_k=_base.TOP_K_LEVENSHTEIN)
    if not top:
        return None
    best = top[0]
    best_score = float(best.get("score", 0.0))
    best_is_late_span = best.get("ayah_end") is not None and int(best.get("ayah", 0)) > 1
    low_confidence = best_score < 0.62
    if not (best_is_late_span or low_confidence):
        return best

    prefix = _surah_prefix_candidates(phoneme_text)
    if not prefix:
        return best
    prefix_best = prefix[0]
    prefix_score = float(prefix_best.get("score", 0.0))
    if prefix_score >= best_score + PREFIX_MARGIN:
        return prefix_best

    global_span = _global_span_candidates(phoneme_text)
    if not global_span:
        return best
    global_best = global_span[0]
    global_score = float(global_best.get("score", 0.0))
    if global_score >= best_score + GLOBAL_SPAN_MARGIN:
        return global_best
    rewind = _same_surah_rewind_candidate(phoneme_text, best)
    if rewind is not None:
        return rewind
    return best


def predict(audio_path: str) -> dict:
    _base._ensure_loaded()
    assert _base._onnx_session is not None
    audio = _base._load_audio(audio_path)
    logprobs = _base._compute_logprobs(audio, _base._onnx_session)
    hyps = _base._hypotheses_from_logprobs(logprobs)
    best_hit: dict | None = None
    best_transcript = ""
    best_rank = -1.0
    for hyp in hyps:
        hit = _best_match_for_hypothesis(hyp)
        if hit is None:
            continue
        if float(hit["score"]) > best_rank:
            best_rank = float(hit["score"])
            best_hit = hit
            best_transcript = hyp
    if not best_hit:
        fallback = _base._labels_to_phoneme_string(_base._greedy_decode_ids(logprobs))
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": fallback}
    return {
        "surah": best_hit["surah"],
        "ayah": best_hit["ayah"],
        "ayah_end": best_hit.get("ayah_end"),
        "score": best_hit["score"],
        "transcript": best_transcript,
    }


def model_size() -> int:
    return _base.model_size()
