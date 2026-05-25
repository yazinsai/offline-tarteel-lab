from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_joint05():
    root = Path(__file__).resolve().parents[1]
    run_py = root / "experiments" / "phoneme_matcher_joint05" / "run.py"
    spec = importlib.util.spec_from_file_location("joint05_test", run_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_opening_collapse_keeps_same_surah_ayah_one_over_late_span(monkeypatch):
    mod = _load_joint05()

    def fake_match(_text: str, top_k: int = 10):
        return [{"surah": 79, "ayah": 4, "ayah_end": 5, "score": 0.69}]

    def fake_prefix(_text: str):
        return [{"surah": 79, "ayah": 1, "ayah_end": 5, "score": 0.55}]

    monkeypatch.setattr(mod._base, "_match_phoneme_text", fake_match)
    monkeypatch.setattr(mod, "_surah_prefix_candidates", fake_prefix)
    monkeypatch.setattr(mod, "_global_span_candidates", lambda _text: [])

    hit = mod._best_match_for_hypothesis("a" * 70)

    assert hit == {"surah": 79, "ayah": 1, "ayah_end": 5, "score": 0.55}


def test_opening_collapse_keeps_fatiha_start_when_best_is_late(monkeypatch):
    mod = _load_joint05()

    def fake_match(_text: str, top_k: int = 10):
        return [{"surah": 1, "ayah": 4, "ayah_end": 7, "score": 0.76}]

    def fake_prefix(_text: str):
        return [{"surah": 1, "ayah": 1, "ayah_end": 7, "score": 0.51}]

    monkeypatch.setattr(mod._base, "_match_phoneme_text", fake_match)
    monkeypatch.setattr(mod, "_surah_prefix_candidates", fake_prefix)
    monkeypatch.setattr(mod, "_global_span_candidates", lambda _text: [])

    hit = mod._best_match_for_hypothesis("b" * 70)

    assert hit == {"surah": 1, "ayah": 1, "ayah_end": 7, "score": 0.51}


def test_head_alignment_score_detects_opening_prefix() -> None:
    mod = _load_joint05()

    score = mod._head_alignment_score(
        "Eabasawatawallaa",
        "Eabasawatawallaa<anj",
    )

    assert score >= 0.9


def test_prefix_candidates_use_head_alignment_for_short_openings(monkeypatch):
    mod = _load_joint05()
    monkeypatch.setattr(mod._base, "_verses", [{"surah": 54, "ayah": 1}])
    monkeypatch.setattr(
        mod,
        "_prefix_span_table",
        lambda: [(54, 5, "Eabasawatawallaa trailing verse phones", "EabasawatawallaaTrailing")],
    )
    monkeypatch.setattr(mod._base, "fragment_score", lambda _query, _ref: 0.48)
    monkeypatch.setattr(mod, "ratio", lambda _query, _ref: 0.49)
    monkeypatch.setattr(mod, "_head_alignment_score", lambda _query, _ref: 1.0)

    hits = mod._surah_prefix_candidates("Eabasawatawallaa noisy tail extra phones")

    assert hits == [
        {
            "surah": 54,
            "ayah": 1,
            "ayah_end": 5,
            "score": 0.745,
            "_prefix_rescue": True,
        }
    ]


def test_duplicate_tie_breaker_leaves_tie_without_context(monkeypatch):
    mod = _load_joint05()
    top = [
        {"surah": 55, "ayah": 13, "ayah_end": None, "score": 0.88, "phonemes": "repeat"},
        {"surah": 55, "ayah": 53, "ayah_end": None, "score": 0.879, "phonemes": "repeat"},
    ]
    monkeypatch.setattr(mod._base, "_by_surah", {55: []})

    assert mod._duplicate_phrase_tie_break("repeat", top) is top[0]


def test_duplicate_tie_breaker_prefers_candidate_with_stronger_context(monkeypatch):
    mod = _load_joint05()
    top = [
        {"surah": 55, "ayah": 13, "ayah_end": None, "score": 0.88, "phonemes": "repeat"},
        {"surah": 55, "ayah": 53, "ayah_end": None, "score": 0.879, "phonemes": "repeat"},
    ]
    monkeypatch.setattr(
        mod._base,
        "_by_surah",
        {
            55: [
                {"surah": 55, "ayah": 12, "phonemes_joined": "wrong before"},
                {"surah": 55, "ayah": 13, "phonemes_joined": "repeat"},
                {"surah": 55, "ayah": 14, "phonemes_joined": "wrong after"},
                {"surah": 55, "ayah": 52, "phonemes_joined": "right before"},
                {"surah": 55, "ayah": 53, "phonemes_joined": "repeat"},
                {"surah": 55, "ayah": 54, "phonemes_joined": "right after"},
            ]
        },
    )

    def fake_fragment(query: str, ref: str) -> float:
        if "rightbefore" in ref and "rightafter" in ref:
            return 0.95
        if "wrongbefore" in ref and "wrongafter" in ref:
            return 0.60
        return 0.0

    monkeypatch.setattr(mod._base, "fragment_score", fake_fragment)

    hit = mod._duplicate_phrase_tie_break("right before repeat right after", top)

    assert hit is top[1]


def test_ambiguous_tie_breaker_prefers_later_surah_for_exact_high_confidence_tie():
    mod = _load_joint05()
    top = [
        {"surah": 69, "ayah": 40, "ayah_end": None, "score": 0.8387},
        {"surah": 81, "ayah": 19, "ayah_end": None, "score": 0.8387},
    ]

    assert mod._ambiguous_tie_break("<inahuu laqawlurasUuulinkarIiim", top) is top[1]


def test_ambiguous_tie_breaker_prefers_later_surah_for_tiny_clipped_near_tie():
    mod = _load_joint05()
    top = [
        {"surah": 17, "ayah": 35, "ayah_end": None, "score": 0.796},
        {"surah": 88, "ayah": 14, "ayah_end": None, "score": 0.7931},
    ]

    assert mod._ambiguous_tie_break("wa<akwauua", top) is top[1]


def test_ambiguous_tie_breaker_does_not_reorder_unlisted_repeated_phrase():
    mod = _load_joint05()
    top = [
        {"surah": 10, "ayah": 48, "ayah_end": None, "score": 0.8776},
        {"surah": 21, "ayah": 38, "ayah_end": None, "score": 0.8776},
    ]

    assert mod._ambiguous_tie_break("wayaquuluunamataa haaalEu", top) is top[0]
