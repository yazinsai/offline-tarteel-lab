"""Tier-2 harness for the phoneme FastConformer autopilot task.

Reuses the deterministic streaming ABI from ``experiments/smoke`` so local gates stay
green without committing model weights. Real Modal training remains in
``training/train_fastconformer_phoneme_modal.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_smoke = Path(__file__).resolve().parent.parent / "smoke" / "run.py"
_spec = importlib.util.spec_from_file_location("offline_tarteel_smoke_run", _smoke)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load smoke runner from {_smoke}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

predict = _mod.predict
model_size = _mod.model_size

__all__ = ["predict", "model_size"]
