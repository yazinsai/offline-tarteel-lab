"""Placeholder tier-2 experiment for the phoneme FastConformer autopilot track.

Delegates prediction to ``experiments/smoke`` so local tier gates stay deterministic
without committing model binaries. Replace with ONNX-backed inference once a
checkpoint path is plumbed through lab metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_smoke_module = None


def _smoke():
    global _smoke_module
    if _smoke_module is None:
        smoke_py = Path(__file__).resolve().parent.parent / "smoke" / "run.py"
        spec = importlib.util.spec_from_file_location("_lab_smoke_delegated", smoke_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load smoke experiment from {smoke_py}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _smoke_module = mod
    return _smoke_module


def predict(audio_path: str) -> dict:
    out = dict(_smoke().predict(audio_path))
    stream = dict(out.get("streaming") or {})
    stream["experiment_profile"] = "fastconformer_phoneme_smoke"
    out["transcript"] = "streaming-fastconformer-phoneme-smoke-placeholder"
    out["streaming"] = stream
    return out


def model_size() -> int:
    return 1
