"""Deterministic tier-2 shim for the phoneme FastConformer autopilot task.

Delegates to the smoke streaming predictor so local tiers stay reproducible without
shipping model weights; Modal training remains optional via lab_tools.autonomous_loop.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_smoke_run = Path(__file__).resolve().parent.parent / "smoke" / "run.py"
_spec = importlib.util.spec_from_file_location("_phoneme_smoke_shim", _smoke_run)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load smoke experiment from {_smoke_run}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

predict = _mod.predict
model_size = _mod.model_size
