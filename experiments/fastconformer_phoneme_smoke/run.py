"""Phoneme FastConformer smoke shim: exposes predict() via the runtime smoke logic (no weights)."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SMOKE_RUN = Path(__file__).resolve().parent.parent / "smoke" / "run.py"
_spec = spec_from_file_location("_offline_tarteel_smoke_bridge", _SMOKE_RUN)
if _spec is None or _spec.loader is None:
    raise RuntimeError("missing experiments/smoke/run.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
predict = _mod.predict
model_size = _mod.model_size
