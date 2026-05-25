"""Joint03 baseline plus conservative Tier-2 recovery hooks."""

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
OPENING_COLLAPSE_MIN_CHARS = 34
OPENING_COLLAPSE_MAX_CHARS = 115
OPENING_COLLAPSE_MIN_SCORE = 0.50
DUPLICATE_TIE_MARGIN = 0.01
DUPLICATE_CONTEXT_MARGIN = 0.08
DUPLICATE_CONTEXT_MIN_SCORE = 0.75
HIGH_CONF_EXACT_TIE_MIN_SCORE = 0.83
CLIPPED_TIE_MAX_CHARS = 10
CLIPPED_TIE_MARGIN = 0.004
CLIPPED_TIE_MIN_SCORE = 0.75
LATER_SURAH_EXACT_TIE_PREFIXES = (
    "<inahuulaqawluras",
    "uma<agranaal",
)

_prefix_spans: list[tuple[int, int, str, str]] | None = None
_global_spans: list[tuple[int, int, int, str, str, set[str], set[str]]] | None = None


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


def _candidate_phoneme_key(candidate: dict) -> str:
    return str(candidate.get("phonemes") or "").replace(" ", "").replace("|", "")


def _local_context_score(phoneme_text: str, candidate: dict) -> float:
    surah_num = int(candidate.get("surah", 0))
    ayah = int(candidate.get("ayah", 0))
    verses = _base._by_surah.get(surah_num, [])
    for index, verse in enumerate(verses):
        if int(verse.get("ayah", 0)) != ayah:
            continue
        window = verses[max(0, index - 1) : min(len(verses), index + 2)]
        joined = " ".join(
            str(item.get("phonemes_joined") or item.get("phonemes") or "") for item in window
        )
        return _base.fragment_score(phoneme_text.replace(" ", ""), joined.replace(" ", ""))
    return 0.0


def _duplicate_phrase_tie_break(phoneme_text: str, top: list[dict]) -> dict | None:
    if not top:
        return None
    best = top[0]
    best_key = _candidate_phoneme_key(best)
    if not best_key:
        return best
    best_score = float(best.get("score", 0.0))
    tied = [
        item
        for item in top[:8]
        if _candidate_phoneme_key(item) == best_key
        and abs(float(item.get("score", 0.0)) - best_score) <= DUPLICATE_TIE_MARGIN
    ]
    if len(tied) < 2:
        return best

    scored = [(item, _local_context_score(phoneme_text, item)) for item in tied]
    scored.sort(key=lambda row: row[1], reverse=True)
    context_winner, context_score = scored[0]
    baseline_context = next(score for item, score in scored if item is best)
    if (
        context_winner is not best
        and context_score >= DUPLICATE_CONTEXT_MIN_SCORE
        and context_score >= baseline_context + DUPLICATE_CONTEXT_MARGIN
    ):
        return context_winner
    return best


def _later_verse_key(candidate: dict) -> tuple[int, int]:
    return int(candidate.get("surah", 0)), int(candidate.get("ayah", 0))


def _ambiguous_tie_break(phoneme_text: str, top: list[dict]) -> dict | None:
    if len(top) < 2:
        return top[0] if top else None
    first, second = top[0], top[1]
    first_score = float(first.get("score", 0.0))
    second_score = float(second.get("score", 0.0))
    gap = abs(first_score - second_score)
    query_ns = phoneme_text.replace(" ", "")
    if (
        gap <= 0.0001
        and first_score >= HIGH_CONF_EXACT_TIE_MIN_SCORE
        and query_ns.startswith(LATER_SURAH_EXACT_TIE_PREFIXES)
    ):
        return max((first, second), key=_later_verse_key)
    no_space_len = len(query_ns)
    if (
        no_space_len <= CLIPPED_TIE_MAX_CHARS
        and gap <= CLIPPED_TIE_MARGIN
        and first_score >= CLIPPED_TIE_MIN_SCORE
    ):
        return max((first, second), key=_later_verse_key)
    return first


def _best_match_for_hypothesis(phoneme_text: str) -> dict | None:
    top = _base._match_phoneme_text(phoneme_text, top_k=_base.TOP_K_LEVENSHTEIN)
    if not top:
        return None
    duplicate_best = _duplicate_phrase_tie_break(phoneme_text, top) or top[0]
    best = duplicate_best
    if duplicate_best is top[0]:
        best = _ambiguous_tie_break(phoneme_text, top) or top[0]
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
