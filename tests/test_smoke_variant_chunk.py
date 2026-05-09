import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_METADATA = _ROOT / "benchmarks" / "adaptive_chunk_seconds_01" / "metadata.json"


def _load_smoke_run():
    run_py = _ROOT / "experiments" / "smoke" / "run.py"
    spec = importlib.util.spec_from_file_location("smoke_lab_run", run_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sample_wav_path():
    p = _ROOT / "benchmark" / "test_corpus_v3" / "sample_placeholder.wav"
    assert p.is_file()
    return str(p)


def test_adaptive_chunk_metadata_manifest():
    raw = json.loads(_METADATA.read_text(encoding="utf-8"))
    assert raw["schema"] == "offline-tarteel.adaptive_variant_manifest.v1"
    assert raw["autopilot_key"] == "runtime.adaptive.chunk_seconds.01"
    assert raw["param"] == "chunk_seconds"


def test_smoke_predict_uses_yaml_chunk_when_env_unset(sample_wav_path, monkeypatch):
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    mod = _load_smoke_run()
    pred = mod.predict(sample_wav_path)
    streaming = pred.get("streaming") or {}
    assert streaming.get("chunk_seconds") == pytest.approx(0.28)
