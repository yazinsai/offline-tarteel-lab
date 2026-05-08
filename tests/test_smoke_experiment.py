import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_run():
    path = ROOT / "experiments" / "smoke" / "run.py"
    spec = importlib.util.spec_from_file_location("smoke_run_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_predict_resolves_manifest_for_benchmark_audio():
    mod = _load_smoke_run()
    audio = ROOT / "benchmark" / "test_corpus_v3" / "sample_placeholder.wav"
    out = mod.predict(str(audio))
    assert out["surah"] == 1
    assert out["ayah"] == 1
    assert "manifest:test_corpus_v3:" in out["transcript"]


def test_smoke_predict_fallback_without_benchmark_segment():
    mod = _load_smoke_run()
    out = mod.predict("/tmp/plain_named.wav")
    assert out["surah"] == 1 and out["ayah"] == 1
    assert out["transcript"] == "smoke_fallback"
