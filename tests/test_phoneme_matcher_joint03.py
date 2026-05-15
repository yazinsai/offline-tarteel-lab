from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_joint03():
    root = Path(__file__).resolve().parents[1]
    run_py = root / "experiments" / "phoneme_matcher_joint03" / "run.py"
    spec = importlib.util.spec_from_file_location("joint03_test", run_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefix_rescue_skips_short_queries(monkeypatch):
    mod = _load_joint03()
    monkeypatch.setattr(mod._base, "_verses", [{"surah": 1, "ayah": 1}])

    assert mod._surah_prefix_candidates("short query") == []


def test_prefix_rescue_does_not_touch_confident_single(monkeypatch):
    mod = _load_joint03()
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
    mod = _load_joint03()

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
