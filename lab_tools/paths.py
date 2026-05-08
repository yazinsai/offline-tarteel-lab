from __future__ import annotations

import os
from pathlib import Path


def reference_root() -> Path:
    """Root of the reference offline-tarteel tree (contains benchmark/, web/, etc.)."""
    env = os.environ.get("OFFLINE_TARTEEL_ROOT")
    if env:
        return Path(env).resolve()
    # Default: parent of offline-tarteel-lab directory
    return Path(__file__).resolve().parent.parent.parent


def lab_root() -> Path:
    return Path(__file__).resolve().parent.parent
