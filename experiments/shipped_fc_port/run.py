"""Shipped offline-tarteel ONNX + RN tracker wired for tier-2 corpus evaluation."""

from __future__ import annotations

import atexit
import json
import subprocess
import threading
from pathlib import Path

from lab_tools.paths import lab_root

_proc: subprocess.Popen[str] | None = None


def _ref_frontend() -> Path:
    return lab_root() / "reference_repo" / "offline-tarteel" / "web" / "frontend"


def _ensure_frontend_deps() -> None:
    fe = _ref_frontend()
    marker = fe / "node_modules" / "onnxruntime-node"
    onnx = fe / "public" / "fastconformer_phoneme_q8.onnx"
    if not onnx.is_file():
        raise FileNotFoundError(
            "Missing reference ONNX weights. Clone offline-tarteel under reference_repo/offline-tarteel "
            "and run: cd reference_repo/offline-tarteel && git lfs pull",
        )
    # Git LFS pointer is tiny protobuf (missing weights).
    if onnx.stat().st_size < 1_000_000:
        raise RuntimeError(
            "ONNX weights not materialized — run git lfs pull inside reference_repo/offline-tarteel",
        )
    if not marker.is_dir():
        subprocess.run(["npm", "ci"], cwd=fe, check=True, timeout=600)


def _drain_stderr(p: subprocess.Popen[str]) -> None:
    if p.stderr is None:
        return
    for line in p.stderr:
        print(line, end="", flush=True)


def _stop_server() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


atexit.register(_stop_server)


def _ensure_server() -> subprocess.Popen[str]:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc

    _ensure_frontend_deps()
    root = lab_root()
    srv_ts = Path(__file__).resolve().parent / "inference_server.ts"
    tsx = _ref_frontend() / "node_modules" / ".bin" / "tsx"
    if not tsx.is_file():
        raise FileNotFoundError(f"missing tsx under reference frontend deps: {tsx}")
    _proc = subprocess.Popen(
        [str(tsx), str(srv_ts)],
        cwd=str(root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_drain_stderr, args=(_proc,), daemon=True).start()
    assert _proc.stdin is not None and _proc.stdout is not None
    return _proc


def predict(audio_path: str) -> dict:
    p = _ensure_server()
    req = json.dumps({"path": str(Path(audio_path).resolve())})
    p.stdin.write(req + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line.startswith("RESULT_JSON:"):
        raise RuntimeError(f"bad server line: {line!r}")

    payload: dict = json.loads(line[len("RESULT_JSON:") :].strip())

    err = payload.get("error")
    streaming = payload.get("streaming")
    if not isinstance(streaming, dict):
        streaming = {}

    streaming.setdefault("reference_frontend", str(_ref_frontend()))
    if isinstance(err, str):
        streaming["error"] = err[:500]

    return {
        "surah": int(payload.get("surah", 1)),
        "ayah": int(payload.get("ayah", 1)),
        "ayah_end": payload.get("ayah_end"),
        "score": float(payload.get("score", 0.0)),
        "transcript": str(payload.get("transcript", "")),
        "streaming": {**streaming, "mode": streaming.get("mode", "shipped_fc_port")},
    }
