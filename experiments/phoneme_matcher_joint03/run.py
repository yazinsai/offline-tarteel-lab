"""Joint02 stack plus bounded prefix rescue, margin-gated local rerank, and hyp agreement.

Offline recoverability analysis (gold in top-K vs missing from shortlist) motivates present-gold
proxying at inference: rerank only when matcher scores are tightly clustered, using only audio-
derived transcript text and public verse phoneme strings (no manifest/expected labels).

Global surah-span enumeration was removed — it dominated CPU on the canonical Tier-2 prefix slice.
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
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

OPENING_COLLAPSE_MIN_CHARS = 34
OPENING_COLLAPSE_MAX_CHARS = 115
OPENING_COLLAPSE_MIN_SCORE = 0.50

LOCAL_RERANK_MARGIN = float(__import__("os").getenv("JOINT03_LOCAL_RERANK_MARGIN", "0.028"))
LOCAL_RERANK_TOPN = 10
LOCAL_MATCHER_W = 0.72
LOCAL_FRAG_W = 0.28
HYPO_AGREE_BONUS = 0.018

_prefix_spans: list[tuple[int, int, str, str]] | None = None


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


def _surah_prefix_candidates(phoneme_text: str) -> list[dict]:
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


def _span_ref_ns(surah: int, ayah: int, ayah_end: int | None) -> str | None:
    if _base._by_surah is None:
        return None
    verses = _base._by_surah.get(surah)
    if not verses:
        return None
    idx = next((i for i, v in enumerate(verses) if int(v.get("ayah", 0)) == ayah), None)
    if idx is None:
        return None
    end_ayah = ayah_end if ayah_end is not None else ayah
    j = next((k for k, v in enumerate(verses) if int(v.get("ayah", 0)) == end_ayah), idx)
    chunk = verses[idx : j + 1]
    if not chunk:
        return None
    first = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
    joined = first + " " + " ".join(v.get("phonemes_joined", "") for v in chunk[1:])
    return joined.replace(" ", "")


def _local_rerank_candidates(phoneme_text: str, candidates: list[dict]) -> list[dict]:
    """Margin-gated rerank: only when the matcher is ambiguous on the top shortlist."""
    if len(candidates) < 2:
        return candidates
    s0 = float(candidates[0]["score"])
    s1 = float(candidates[1]["score"])
    if s0 - s1 > LOCAL_RERANK_MARGIN:
        return candidates

    no_space_text = phoneme_text.replace(" ", "")
    scored: list[tuple[float, dict]] = []
    for c in candidates[:LOCAL_RERANK_TOPN]:
        m = float(c["score"])
        ref_ns = _span_ref_ns(int(c["surah"]), int(c["ayah"]), c.get("ayah_end"))
        frag = 0.0
        if ref_ns:
            frag = _base.fragment_score(no_space_text, ref_ns)
            no_bsm = None
            verses = _base._by_surah.get(int(c["surah"])) if _base._by_surah else None
            if verses:
                row = next((v for v in verses if int(v.get("ayah", 0)) == int(c["ayah"])), None)
                if row:
                    no_bsm = row.get("_phonemes_joined_no_bsm_ns")
            if no_bsm:
                frag = max(frag, _base.fragment_score(no_space_text, no_bsm))
        combo = LOCAL_MATCHER_W * m + LOCAL_FRAG_W * frag
        nc = dict(c)
        nc["score"] = round(combo, 4)
        scored.append((combo, nc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _best_match_for_hypothesis(phoneme_text: str) -> dict | None:
    top = _base._match_phoneme_text(phoneme_text, top_k=_base.TOP_K_LEVENSHTEIN)
    if not top:
        return None
    top = _local_rerank_candidates(phoneme_text, top)
    best = top[0]
    best_score = float(best.get("score", 0.0))
    best_is_late_span = best.get("ayah_end") is not None and int(best.get("ayah", 0)) > 1
    low_confidence = best_score < 0.62
    if not (best_is_late_span or low_confidence):
        return best

    no_space_len = len(phoneme_text.replace(" ", ""))
    prefix = _surah_prefix_candidates(phoneme_text)
    candidates = [best]
    candidates.extend(
        p
        for p in prefix
        if float(p.get("score", 0.0)) >= best_score + PREFIX_MARGIN
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

    votes: dict[tuple[int, int, int | None], float] = defaultdict(float)
    counts: dict[tuple[int, int, int | None], int] = defaultdict(int)
    transcript_for: dict[tuple[int, int, int | None], str] = {}
    best_hyp_score_for_key: dict[tuple[int, int, int | None], float] = defaultdict(lambda: -1.0)

    for hyp in hyps:
        hit = _best_match_for_hypothesis(hyp)
        if hit is None:
            continue
        ae = hit.get("ayah_end")
        ae_int = int(ae) if ae is not None else None
        key = (int(hit["surah"]), int(hit["ayah"]), ae_int)
        sc = float(hit["score"])
        counts[key] += 1
        votes[key] += sc
        if sc > best_hyp_score_for_key[key]:
            best_hyp_score_for_key[key] = sc
            transcript_for[key] = hyp

    if not votes:
        fallback = _base._labels_to_phoneme_string(_base._greedy_decode_ids(logprobs))
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": fallback}

    def adjusted(k: tuple[int, int, int | None]) -> float:
        return votes[k] + HYPO_AGREE_BONUS * float(max(0, counts[k] - 1))

    best_key = max(votes, key=adjusted)
    surah, ayah, ae_out = best_key
    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": ae_out,
        "score": round(votes[best_key] / max(1, counts[best_key]), 4),
        "transcript": transcript_for.get(best_key, ""),
    }


def model_size() -> int:
    return _base.model_size()
