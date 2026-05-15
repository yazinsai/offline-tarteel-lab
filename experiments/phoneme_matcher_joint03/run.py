"""Joint02 phoneme matcher plus bounded global surah-prefix rescue."""

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
GLOBAL_SPAN_MIN_CHARS = 80
GLOBAL_SPAN_MIN_SCORE = 0.54
GLOBAL_SPAN_MARGIN = -0.015
GLOBAL_SPAN_SHORTLIST = 320
OPENING_COLLAPSE_MIN_CHARS = 34
OPENING_COLLAPSE_MAX_CHARS = 115
OPENING_COLLAPSE_MIN_SCORE = 0.50

_prefix_spans: list[tuple[int, int, str, str]] | None = None
_global_spans: list[tuple[int, int, int, str, str, set[str], set[str]]] | None = None


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
    rough: list[tuple[float, tuple[int, int, int, str, str, set[str], set[str]]]] = []
    for row in _global_span_table():
        rb = row[5]
        rt = row[6]
        ov = float(len(qb & rb)) + 0.48 * float(len(qt & rt))
        if ov > 0:
            rough.append((ov, row))
    rough.sort(key=lambda x: x[0], reverse=True)

    out: list[dict] = []
    for _, (surah_num, ayah, ayah_end, span_phonemes, ref_ns, _rb, _rt) in rough[
        :GLOBAL_SPAN_SHORTLIST
    ]:
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

    no_space_len = len(phoneme_text.replace(" ", ""))
    prefix = _surah_prefix_candidates(phoneme_text)
    global_span = _global_span_candidates(phoneme_text)

    candidates = [best]
    candidates.extend(
        p
        for p in prefix
        if float(p.get("score", 0.0)) >= best_score + PREFIX_MARGIN
    )
    candidates.extend(
        g
        for g in global_span
        if float(g.get("score", 0.0)) >= best_score + GLOBAL_SPAN_MARGIN
    )
    candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    chosen = candidates[0]

    if (
        OPENING_COLLAPSE_MIN_CHARS <= no_space_len <= OPENING_COLLAPSE_MAX_CHARS
        and best.get("ayah_end") is not None
        and int(best.get("ayah", 0)) > 1
    ):
        same_surah_prefix = [
            p
            for p in prefix
            if int(p.get("surah", 0)) == int(best.get("surah", 0))
            and float(p.get("score", 0.0)) >= OPENING_COLLAPSE_MIN_SCORE
            and (
                p.get("ayah_end") is None
                or best.get("ayah_end") is None
                or int(p.get("ayah_end", 0)) >= int(best.get("ayah_end", 0))
            )
        ]
        if same_surah_prefix:
            same_surah_prefix.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return same_surah_prefix[0]

    if chosen is not best:
        return chosen
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
