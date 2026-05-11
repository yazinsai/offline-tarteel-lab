"""Lightweight multi-pass acoustic front-end + same joint02 beam/matcher (model_only 46.03).

Runs the shipped phoneme ONNX on several Mel front-ends (pre-emphasis / f_max variants), merges
unique CTC hypotheses, then reuses phoneme_matcher_joint02 matching. This is decode/search
diversity from audio conditioning, not matcher reranks on a single decode.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from operator import attrgetter
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from Levenshtein import ratio

_DIR = Path(__file__).resolve().parent
_QURAN_JSON = _DIR / "quran_phonemes.json"

_ONNX_URL = (
    "https://github.com/yazinsai/offline-tarteel/releases/download/v0.1.0/fastconformer_phoneme_q8.onnx"
)
_ONNX_SHA256 = "78279e112f2762312b1412880d35f21f423bd117ee9222b82b8b9782c81bfaf9"

PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]
BLANK_ID = len(PHONEME_VOCAB)

_onnx_session: ort.InferenceSession | None = None
_verses: list[dict] | None = None
_by_surah: dict[int, list[dict]] = {}

_BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi"

TOP_K_LEVENSHTEIN = int(os.getenv("PHONEME_LM_TOP_K", "18"))
TOP_SURAHS = int(os.getenv("PHONEME_LM_TOP_SURAHS", "32"))
MAX_SPAN = int(os.getenv("PHONEME_LM_MAX_SPAN", "6"))
FRAGMENT_BLEND = float(os.getenv("PHONEME_FRAGMENT_BLEND", "0.82"))
# Narrow beam keeps full-corpus tier-2 wall time bounded on CPU.
BEAM_WIDTH = int(os.getenv("PHONEME_CTC_BEAM_WIDTH", "6"))
BEAM_TOP_SYMBOLS = int(os.getenv("PHONEME_CTC_TOP_SYMBOLS", "8"))
MAX_HYPOTHESES = int(os.getenv("PHONEME_MAX_HYPOTHESES", "4"))
# Collect hypotheses per acoustic pass, then cap merged list before matcher.
PER_PASS_HYPOTHESES = int(os.getenv("PHONEME_TWOPASS_PER_PASS", "5"))
MERGED_HYP_CAP = int(os.getenv("PHONEME_TWOPASS_MERGE_CAP", "12"))

_onnx_input_name = attrgetter("name")


def _cache_onnx_path() -> Path:
    raw = os.getenv("PHONEME_ONNX_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "offline-tarteel-lab" / "fastconformer_phoneme_q8.onnx"


def _ensure_onnx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1_000_000:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == _ONNX_SHA256:
            return
    tmp = path.with_suffix(path.suffix + ".download")
    req = urllib.request.Request(_ONNX_URL, headers={"User-Agent": "offline-tarteel-lab/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    h = hashlib.sha256()
    with tmp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != _ONNX_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded ONNX failed sha256 check")
    tmp.replace(path)


def _load_audio(path: str, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def semi_global_distance(query: str, ref: str) -> int:
    if not query:
        return 0
    if not ref:
        return len(query)
    m, n = len(query), len(ref)
    prev = list(range(m + 1))
    best = prev[m]
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        for i in range(1, m + 1):
            cost = 0 if query[i - 1] == ref[j - 1] else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        best = min(best, curr[m])
        prev = curr
    return best


def fragment_score(query: str, ref: str) -> float:
    if not query:
        return 1.0
    return max(0.0, 1.0 - semi_global_distance(query, ref) / len(query))


def _short_query_boost(no_space_text: str, verse: dict, use_no_bsm: bool = False) -> float:
    if use_no_bsm:
        candidate = verse.get("_phonemes_joined_no_bsm_ns", "") or verse.get("_phonemes_joined_ns", "")
    else:
        candidate = verse.get("_phonemes_joined_ns", "")
    if not candidate:
        return 0.0
    prefix_window = min(len(candidate), len(no_space_text) + 6)
    prefix = ratio(no_space_text, candidate[:prefix_window])
    if use_no_bsm:
        joined = verse.get("_phonemes_joined_no_bsm", "") or ""
    else:
        joined = verse.get("phonemes_joined", "")
    first_word = joined.split(" ")[0] if joined else ""
    first_word_score = ratio(no_space_text, first_word) if first_word else 0.0
    return max(prefix, first_word_score)


def _query_bigrams(s: str) -> set[str]:
    if len(s) < 2:
        return set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _query_trigrams(s: str) -> set[str]:
    if len(s) < 3:
        return set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _candidate_verses(no_space_text: str, *, max_candidates: int = 950) -> list[dict]:
    if _verses is None or len(no_space_text) < 4:
        return list(_verses or [])
    qb = _query_bigrams(no_space_text)
    qt = _query_trigrams(no_space_text)
    if not qb and not qt:
        return list(_verses)
    scored: list[tuple[float, int]] = []
    for i, verse in enumerate(_verses):
        ref_ns = verse.get("_phonemes_joined_ns", "")
        if len(ref_ns) < 2:
            continue
        rb = _query_bigrams(ref_ns)
        rt = _query_trigrams(ref_ns)
        ov = float(len(qb & rb)) + 0.48 * float(len(qt & rt))
        if ov > 0:
            scored.append((ov, i))
    if len(scored) < 80:
        return list(_verses)
    scored.sort(key=lambda x: x[0], reverse=True)
    pick = [idx for _, idx in scored[:max_candidates]]
    return [_verses[i] for i in pick]


def _match_phoneme_text(phoneme_text: str, top_k: int = 10) -> list[dict]:
    if not phoneme_text.strip() or _verses is None:
        return []
    no_space_text = phoneme_text.replace(" ", "")
    scored: list[list] = []
    for verse in _candidate_verses(no_space_text):
        ref = verse.get("phonemes_joined", "")
        if not ref:
            continue
        raw = ratio(phoneme_text, ref)
        if len(no_space_text) <= 10:
            raw = max(raw, _short_query_boost(no_space_text, verse))
        no_bsm = verse.get("_phonemes_joined_no_bsm")
        if no_bsm:
            raw = max(raw, ratio(phoneme_text, no_bsm))
            if len(no_space_text) <= 10:
                raw = max(raw, _short_query_boost(no_space_text, verse, use_no_bsm=True))
        scored.append([verse, raw, raw])
    scored.sort(key=lambda x: x[2], reverse=True)

    pass2_surahs: list[int] = []
    for entry in scored:
        s = entry[0]["surah"]
        if s not in pass2_surahs:
            pass2_surahs.append(s)
        if len(pass2_surahs) >= TOP_SURAHS:
            break

    if len(no_space_text) >= 8:
        resorted = False
        for i, (verse, raw, _) in enumerate(scored):
            ref_ns = verse.get("_phonemes_joined_ns", "")
            if not ref_ns or len(no_space_text) >= len(ref_ns) * 0.8:
                continue
            frag = fragment_score(no_space_text, ref_ns)
            no_bsm_ns = verse.get("_phonemes_joined_no_bsm_ns")
            if no_bsm_ns:
                frag = max(frag, fragment_score(no_space_text, no_bsm_ns))
            if frag > raw:
                boosted = raw + (frag - raw) * FRAGMENT_BLEND
                scored[i] = [verse, boosted, boosted]
                resorted = True
        if resorted:
            scored.sort(key=lambda x: x[2], reverse=True)

    span_results: list[dict] = []
    for surah_num in pass2_surahs:
        verses = _by_surah.get(surah_num, [])
        for i in range(len(verses)):
            for span in range(2, MAX_SPAN + 1):
                if i + span > len(verses):
                    break
                chunk = verses[i : i + span]
                first_phonemes = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get(
                    "phonemes_joined", ""
                )
                span_phonemes = first_phonemes + " " + " ".join(
                    v.get("phonemes_joined", "") for v in chunk[1:]
                )
                raw = ratio(phoneme_text, span_phonemes)
                span_results.append(
                    {
                        "surah": surah_num,
                        "ayah": chunk[0]["ayah"],
                        "ayah_end": chunk[-1]["ayah"],
                        "score": round(raw, 4),
                        "phonemes": " | ".join(v.get("phonemes", "") for v in chunk),
                    }
                )

    singles = []
    for verse, raw, boosted in scored[: max(top_k, 32)]:
        singles.append(
            {
                "surah": verse["surah"],
                "ayah": verse["ayah"],
                "ayah_end": None,
                "score": round(boosted, 4),
                "phonemes": verse.get("phonemes", ""),
            }
        )
    combined = singles + span_results
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


def _preemph_waveform(audio: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0.0:
        return audio
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)


def _compute_logprobs(
    audio: np.ndarray,
    session: ort.InferenceSession,
    *,
    preemph: float = 0.97,
    fmax: float = 8000.0,
) -> np.ndarray:
    x = _preemph_waveform(audio, preemph)
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmax=fmax,
        htk=True,
        norm="slaney",
    )
    mel = np.log(mel + 1e-5)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)
    features = mel.astype(np.float32)[np.newaxis]
    length = np.array([mel.shape[1]], dtype=np.int64)
    input_names = [_onnx_input_name(inp) for inp in session.get_inputs()]
    results = session.run(
        None,
        {
            input_names[0]: features,
            input_names[1]: length,
        },
    )
    return results[0][0]


def _labels_to_phoneme_string(labeling: tuple[int, ...]) -> str:
    tokens = [PHONEME_VOCAB[i] for i in labeling if 0 <= i < len(PHONEME_VOCAB)]
    words = []
    cur: list[str] = []
    for t in tokens:
        if t == "|":
            if cur:
                words.append("".join(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        words.append("".join(cur))
    return " ".join(words)


def _greedy_decode_ids(logprobs: np.ndarray) -> tuple[int, ...]:
    ids = logprobs.argmax(axis=1)
    prev = -1
    out: list[int] = []
    for idx in ids:
        i = int(idx)
        if i != prev and i != BLANK_ID:
            out.append(i)
        prev = i
    return tuple(out)


def _ctc_prefix_beam_decode(
    logprobs: np.ndarray,
    *,
    beam_width: int,
    top_symbols: int,
) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam; log-probs assumed to be natural log of softmax probabilities."""
    t_len, v_dim = int(logprobs.shape[0]), int(logprobs.shape[1])
    beam: dict[tuple[int, ...], float] = {(): 0.0}
    blank = BLANK_ID
    for t in range(t_len):
        pr = logprobs[t]
        idx = np.argpartition(-pr, min(top_symbols, v_dim - 1))[:top_symbols]
        next_beam: dict[tuple[int, ...], float] = {}
        items = sorted(beam.items(), key=lambda x: -x[1])[: max(beam_width * 3, beam_width)]
        for labeling, lp in items:
            for c in idx:
                c = int(c)
                logp = lp + float(pr[c])
                if c == blank:
                    nl = labeling
                elif len(labeling) > 0 and labeling[-1] == c:
                    nl = labeling
                else:
                    nl = labeling + (c,)
                prev_lp = next_beam.get(nl)
                if prev_lp is None:
                    next_beam[nl] = logp
                else:
                    next_beam[nl] = float(np.logaddexp(prev_lp, logp))
        beam = dict(sorted(next_beam.items(), key=lambda x: -x[1])[:beam_width])
    return sorted(beam.items(), key=lambda x: -x[1])[:beam_width]


def _hypotheses_from_logprobs(logprobs: np.ndarray, *, max_out: int | None = None) -> list[str]:
    cap = MAX_HYPOTHESES if max_out is None else max_out
    greedy = _greedy_decode_ids(logprobs)
    g_str = _labels_to_phoneme_string(greedy)
    out: list[str] = []
    if g_str.strip():
        out.append(g_str)
    if BEAM_WIDTH <= 1:
        return out[:cap]
    seen = set(out)
    for labeling, _ in _ctc_prefix_beam_decode(
        logprobs,
        beam_width=BEAM_WIDTH,
        top_symbols=BEAM_TOP_SYMBOLS,
    ):
        s = _labels_to_phoneme_string(labeling)
        if not s.strip() or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out[:cap]


def _merged_hypotheses(
    audio: np.ndarray,
    session: ort.InferenceSession,
    *,
    variants: list[tuple[float, float]] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return merged hypothesis strings and the last logprob tensor (fallback decode)."""
    if variants is None:
        variants = [
            (0.97, 8000.0),
            (0.88, 8000.0),
            (0.0, 8000.0),
            (0.97, 7600.0),
        ]
    merged: list[str] = []
    seen: set[str] = set()
    last_lp: np.ndarray | None = None
    for preemph, fmax in variants:
        lp = _compute_logprobs(audio, session, preemph=preemph, fmax=fmax)
        last_lp = lp
        for h in _hypotheses_from_logprobs(lp, max_out=PER_PASS_HYPOTHESES):
            if h not in seen:
                seen.add(h)
                merged.append(h)
        if len(merged) >= MERGED_HYP_CAP:
            break
    assert last_lp is not None
    return merged, last_lp


def _ensure_loaded() -> None:
    global _onnx_session, _verses, _by_surah
    if _onnx_session is not None and _verses is not None:
        return
    path = _cache_onnx_path()
    _ensure_onnx(path)
    _onnx_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _by_surah.clear()
    _verses = _load_verse_table()


def _phonemes_field_to_joined(raw: str) -> str:
    parts = [p.strip().replace(" ", "") for p in raw.split("|")]
    return " ".join(p for p in parts if p)


def _load_verse_table() -> list[dict]:
    verses: list[dict] = json.loads(_QURAN_JSON.read_text(encoding="utf-8"))
    for v in verses:
        joined = (v.get("phonemes_joined") or "").strip()
        if not joined and v.get("phonemes"):
            joined = _phonemes_field_to_joined(str(v["phonemes"]))
        v["phonemes_joined"] = joined
        v["_phonemes_joined_ns"] = joined.replace(" ", "")
        no_bsm = None
        if (
            v["ayah"] == 1
            and v["surah"] != 1
            and v["surah"] != 9
            and joined.startswith(_BSM_PHONEMES_JOINED)
        ):
            tail = joined[len(_BSM_PHONEMES_JOINED) :].strip()
            no_bsm = tail or None
        v["_phonemes_joined_no_bsm"] = no_bsm
        v["_phonemes_joined_no_bsm_ns"] = no_bsm.replace(" ", "") if no_bsm else None
        _by_surah.setdefault(v["surah"], []).append(v)
    for lst in _by_surah.values():
        lst.sort(key=lambda x: x["ayah"])
    return verses


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    assert _onnx_session is not None
    audio = _load_audio(audio_path)
    hyps, logprobs = _merged_hypotheses(audio, _onnx_session)
    best_hit: dict | None = None
    best_transcript = ""
    best_rank = -1.0
    for hyp in hyps:
        top = _match_phoneme_text(hyp, top_k=TOP_K_LEVENSHTEIN)
        if not top:
            continue
        if float(top[0]["score"]) > best_rank:
            best_rank = float(top[0]["score"])
            best_hit = top[0]
            best_transcript = hyp
    if not best_hit:
        fallback = _labels_to_phoneme_string(_greedy_decode_ids(logprobs))
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": fallback}
    return {
        "surah": best_hit["surah"],
        "ayah": best_hit["ayah"],
        "ayah_end": best_hit.get("ayah_end"),
        "score": best_hit["score"],
        "transcript": best_transcript,
    }


def model_size() -> int:
    p = _cache_onnx_path()
    if p.is_file():
        return int(p.stat().st_size)
    return 131_000_000
