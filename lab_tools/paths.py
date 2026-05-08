from __future__ import annotations

import os
from pathlib import Path


def lab_root() -> Path:
    return Path(__file__).resolve().parent.parent


def reference_root() -> Path:
    """Root containing benchmark/experiments/web assets used by lab tools.

    Standalone default is this lab repository root.
    """
    env = os.environ.get("OFFLINE_TARTEEL_ROOT") or os.environ.get("OFFLINE_TARTEEL_REFERENCE_ROOT")
    if env:
        return Path(env).resolve()
    return lab_root()
