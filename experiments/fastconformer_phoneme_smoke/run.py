"""Phoneme FastConformer smoke stub: Tier-2 ABI + training-candidate metadata (no binaries)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from lab_tools.paths import reference_root

_SMOKE_RUN = Path(__file__).resolve().parent.parent / "smoke" / "run.py"
_MANIFEST = reference_root() / "benchmark" / "manifests" / "phoneme_candidate_stub.json"

_smoke_mod = None


def _smoke():
    global _smoke_mod
    if _smoke_mod is None:
        spec = importlib.util.spec_from_file_location("exp_smoke_delegate", _SMOKE_RUN)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load smoke delegate from {_SMOKE_RUN}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _smoke_mod = mod
    return _smoke_mod


def _candidate_manifest() -> dict:
    if not _MANIFEST.is_file():
        raise RuntimeError(f"missing phoneme candidate manifest {_MANIFEST}")
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def predict(audio_path: str) -> dict:
    base = dict(_smoke().predict(audio_path))
    manifest = _candidate_manifest()
    base["transcript"] = "phoneme-fastconformer-smoke"
    base["phoneme_training_candidate"] = {
        "manifest_schema": manifest.get("schema"),
        "modal_job_name_default": manifest.get("modal_job_name_default"),
        "training_entrypoint_relative": manifest.get("training_entrypoint_relative"),
        "tier2_stub_experiment": manifest.get("tier2_stub_experiment"),
    }
    return base


def model_size() -> int:
    return _smoke().model_size()
