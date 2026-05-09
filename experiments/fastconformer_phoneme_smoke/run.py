"""Phoneme FastConformer smoke shim: tier-2 ABI compatible without committed weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SMOKE_RUN = Path(__file__).resolve().parent.parent / "smoke" / "run.py"


def _smoke_mod():
    spec = importlib.util.spec_from_file_location("lab_smoke_experiment", _SMOKE_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load smoke experiment from {_SMOKE_RUN}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SM = _smoke_mod()


def predict(audio_path: str) -> dict:
    out = dict(_SM.predict(audio_path))
    out["transcript"] = "fastconformer-phoneme-candidate"
    stream = dict(out.get("streaming") or {})
    stream["model_family"] = "fastconformer_phoneme_smoke"
    stream["phoneme_head"] = "ctc_placeholder"
    out["streaming"] = stream
    return out


def model_size() -> int:
    return _SM.model_size()
