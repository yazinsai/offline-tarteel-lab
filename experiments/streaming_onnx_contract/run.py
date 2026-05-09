"""Streaming consumer stub with ONNX export/placement contract metadata checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from lab_tools.paths import reference_root

_SMOKE_RUN = Path(__file__).resolve().parent.parent / "smoke" / "run.py"
_MANIFEST = reference_root() / "benchmark" / "manifests" / "onnx_placeholder_contract.json"

_smoke_mod = None


def _smoke():
    global _smoke_mod
    if _smoke_mod is None:
        spec = importlib.util.spec_from_file_location("exp_smoke_delegate_onnx", _SMOKE_RUN)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load smoke delegate from {_SMOKE_RUN}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _smoke_mod = mod
    return _smoke_mod


def _contract() -> dict:
    if not _MANIFEST.is_file():
        raise RuntimeError(f"missing ONNX contract manifest {_MANIFEST}")
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def predict(audio_path: str) -> dict:
    out = dict(_smoke().predict(audio_path))
    m = _contract()
    sz = float(m.get("size_mb_estimate", 0.0))
    cap = float(m.get("max_onnx_mb", 200.0))
    if sz > cap + 1e-6:
        raise RuntimeError("onnx_placeholder exceeds declared max_onnx_mb budget")
    out["transcript"] = "streaming-onnx-contract"
    out["onnx_contract"] = {
        "manifest_schema": m.get("schema"),
        "relative_placeholder": m.get("artifact_relative_path_placeholder"),
        "size_mb_estimate": sz,
        "max_onnx_mb": cap,
        "within_budget": sz <= cap + 1e-9,
        "opset_min": m.get("opset_min"),
    }
    return out


def model_size() -> int:
    return max(1, int(round(_contract().get("size_mb_estimate", 1.0))))
