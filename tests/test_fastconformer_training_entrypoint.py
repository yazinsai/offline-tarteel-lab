"""Local smoke: phoneme training entrypoint stays CLI-runnable without Modal."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lab_tools.paths import lab_root


def test_train_fastconformer_phoneme_modal_main_smoke():
    root = lab_root()
    script = root / "training" / "train_fastconformer_phoneme_modal.py"
    assert script.is_file()
    r = subprocess.run(
        [sys.executable, str(script), "--job-name", "lab-local-smoke"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    assert "lab-local-smoke" in (r.stdout + r.stderr)
