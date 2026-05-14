"""Diagnostic sibling of phoneme_matcher_joint02: identical predict()/model_size()."""

from __future__ import annotations

import importlib.util

from lab_tools.paths import lab_root


def _load_joint():
    jp = lab_root() / "experiments" / "phoneme_matcher_joint02" / "run.py"
    spec = importlib.util.spec_from_file_location("phoneme_joint02_delegate", jp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing phoneme_joint02 at {jp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_j = None


def _mod():
    global _j
    if _j is None:
        _j = _load_joint()
    return _j


def predict(audio_path: str) -> dict:
    return _mod().predict(audio_path)


def model_size() -> int:
    return _mod().model_size()
