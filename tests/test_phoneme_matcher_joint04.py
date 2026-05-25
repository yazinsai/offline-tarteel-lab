from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_joint04():
    root = Path(__file__).resolve().parents[1]
    run_py = root / "experiments" / "phoneme_matcher_joint04" / "run.py"
    spec = importlib.util.spec_from_file_location("joint04_test", run_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefix_rescue_skips_short_queries(monkeypatch):
    mod = _load_joint04()
    monkeypatch.setattr(mod._base, "_verses", [{"surah": 1, "ayah": 1}])

    assert mod._surah_prefix_candidates("short query") == []


def test_prefix_rescue_does_not_touch_confident_single(monkeypatch):
    mod = _load_joint04()
    calls = {"prefix": 0}

    def fake_match(_text: str, top_k: int = 10):
        return [{"surah": 2, "ayah": 10, "ayah_end": None, "score": 0.88}]

    def fake_prefix(_text: str):
        calls["prefix"] += 1
        return [{"surah": 99, "ayah": 1, "ayah_end": 3, "score": 0.99}]

    monkeypatch.setattr(mod._base, "_match_phoneme_text", fake_match)
    monkeypatch.setattr(mod, "_surah_prefix_candidates", fake_prefix)

    assert mod._best_match_for_hypothesis("a long enough decoded phoneme transcript") == {
        "surah": 2,
        "ayah": 10,
        "ayah_end": None,
        "score": 0.88,
    }
    assert calls["prefix"] == 0


def test_prefix_rescue_can_replace_low_confidence_hit_when_it_beats_base(monkeypatch):
    mod = _load_joint04()

    def fake_match(_text: str, top_k: int = 10):
        return [{"surah": 26, "ayah": 215, "ayah_end": None, "score": 0.57}]

    def fake_prefix(_text: str):
        return [{"surah": 44, "ayah": 1, "ayah_end": 5, "score": 0.58}]

    monkeypatch.setattr(mod._base, "_match_phoneme_text", fake_match)
    monkeypatch.setattr(mod, "_surah_prefix_candidates", fake_prefix)

    assert mod._best_match_for_hypothesis("a long enough decoded phoneme transcript") == {
        "surah": 44,
        "ayah": 1,
        "ayah_end": 5,
        "score": 0.58,
    }


def test_same_surah_rewind_only_expands_late_span(monkeypatch):
    mod = _load_joint04()
    monkeypatch.setattr(
        mod._base,
        "_by_surah",
        {
            1: [
                {
                    "surah": 1,
                    "ayah": 1,
                    "phonemes_joined": "aaa",
                    "_phonemes_joined_no_bsm": None,
                },
                {"surah": 1, "ayah": 2, "phonemes_joined": "bbb"},
                {"surah": 1, "ayah": 3, "phonemes_joined": "ccc"},
                {"surah": 1, "ayah": 4, "phonemes_joined": "ddd"},
            ]
        },
    )
    monkeypatch.setattr(mod._base, "_verses", [{"surah": 1, "ayah": 1}])
    monkeypatch.setattr(mod._base, "fragment_score", lambda _query, _ref: 0.72)
    monkeypatch.setattr(mod, "ratio", lambda _query, _ref: 0.70)

    hit = mod._same_surah_rewind_candidate(
        "aaa bbb ccc ddd " * 8,
        {"surah": 1, "ayah": 3, "ayah_end": 4, "score": 0.75},
    )

    assert hit == {
        "surah": 1,
        "ayah": 1,
        "ayah_end": 4,
        "score": 0.7164,
        "_same_surah_rewind": True,
    }


def test_head_alignment_boosts_short_prefix_candidates():
    mod = _load_joint04()

    score = mod._head_alignment_score(
        "Eabasawatawallaa",
        "Eabasawatawallaa<anj",
    )

    assert score >= 0.9


def test_global_span_index_maps_ngrams_to_span_rows(monkeypatch):
    mod = _load_joint04()
    monkeypatch.setattr(
        mod,
        "_global_spans",
        [
            (1, 1, 2, "a b", "ab", {"ab"}, set()),
            (2, 1, 2, "a b c", "abc", {"ab", "bc"}, {"abc"}),
        ],
    )
    monkeypatch.setattr(mod, "_global_span_index", None)

    bigrams, trigrams = mod._global_span_ngram_index()

    assert bigrams["ab"] == [0, 1]
    assert bigrams["bc"] == [1]
    assert trigrams["abc"] == [1]
