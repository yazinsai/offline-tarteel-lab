"""Sanity checks for the phoneme training entrypoint (no Modal decorators required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_training_script_runs_with_job_name():
    root = Path(__file__).resolve().parents[1]
    script = root / "training" / "train_fastconformer_phoneme_modal.py"
    r = subprocess.run(
        [sys.executable, str(script), "--job-name", "pytest-phoneme-smoke"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "pytest-phoneme-smoke" in r.stdout
