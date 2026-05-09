"""Tier-2 ABI: real audio -> shipped fastconformer-phoneme v4-tlog + RecitationTracker (reference repo)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

_REF = Path(os.environ.get("OFFLINE_TARTEEL_REFERENCE_ROOT", "/tmp/offline-tarteel-reference")).resolve()
_FE = _REF / "web" / "frontend"
_MODEL = _FE / "public" / "fastconformer_phoneme_q8.onnx"
_WORKER_SRC = Path(__file__).resolve().parent / "lab_predict_worker.ts"
_WORKER_DST = _FE / "test" / "lab_predict_worker.ts"

_worker: subprocess.Popen[str] | None = None


def _ensure_ref() -> None:
    if not _REF.is_dir():
        msg = (
            f"Reference repo missing at {_REF}. "
            "Clone: git clone https://github.com/yazinsai/offline-tarteel.git "
            f"{_REF} && cd {_REF} && git lfs pull"
        )
        raise RuntimeError(msg)
    if not _MODEL.is_file() or _MODEL.stat().st_size < 1_000_000:
        msg = f"ONNX missing or Git LFS stub at {_MODEL}. Run: cd {_REF} && git lfs pull"
        raise RuntimeError(msg)


def _ensure_npm() -> None:
    if (_FE / "node_modules").is_dir():
        return
    subprocess.run(["npm", "ci"], cwd=str(_FE), check=True, timeout=600)


def _sync_worker() -> None:
    _WORKER_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_WORKER_SRC, _WORKER_DST)


def _get_worker() -> subprocess.Popen[str]:
    global _worker
    if _worker is not None and _worker.poll() is None and _worker.stdin is not None and _worker.stdout is not None:
        return _worker
    if _worker is not None:
        try:
            _worker.terminate()
        except OSError:
            pass
        _worker = None
    _ensure_ref()
    _ensure_npm()
    _sync_worker()
    _worker = subprocess.Popen(
        ["npx", "tsx", "test/lab_predict_worker.ts"],
        cwd=str(_FE),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if _worker.stdin is None or _worker.stdout is None:
        raise RuntimeError("worker missing stdio pipes")
    return _worker


def predict(audio_path: str) -> dict[str, int]:
    """First streaming verse_match surah/ayah from reference tracker + ONNX (no path/filename labels)."""
    w = _get_worker()
    assert w.stdin is not None and w.stdout is not None
    w.stdin.write(str(audio_path) + "\n")
    w.stdin.flush()
    line = w.stdout.readline()
    if not line:
        global _worker
        err = (w.stderr.read() if w.stderr else "") if w.poll() is not None else ""
        raise RuntimeError(f"worker EOF (code={w.poll()}): {err}")
    data = json.loads(line.strip())
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return {"surah": int(data.get("surah", 0)), "ayah": int(data.get("ayah", 0))}
