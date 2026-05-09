from __future__ import annotations

from pathlib import Path


def lab_root() -> Path:
    return Path(__file__).resolve().parent.parent


def reference_root() -> Path:
    """Root containing benchmark/experiments/web assets used by lab tools.

    Standalone default is this lab repository root.
    """
    return lab_root()
