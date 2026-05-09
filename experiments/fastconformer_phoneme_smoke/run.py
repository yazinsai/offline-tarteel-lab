"""Tier-2 shim for the phoneme FastConformer autopilot key (model.fastconformer_phoneme_smoke).

Reuses the deterministic smoke predict() until a real acoustic model is plumbed in.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_smoke_exp",
    Path(__file__).resolve().parent.parent / "smoke" / "run.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

predict = _mod.predict
model_size = _mod.model_size
