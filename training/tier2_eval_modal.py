"""Sharded Modal runner for Tier-2 experiment evaluation."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "offline-tarteel-tier2-eval"
REPO_ROOT = Path("/repo")
MODEL_VOLUME_NAME = os.getenv("TARTEEL_MODAL_OUTPUT_VOLUME", "offline-tarteel-lab-models")
MODEL_CACHE_ROOT = Path("/model-cache")
ONNX_CACHE_PATH = MODEL_CACHE_ROOT / "onnx" / "fastconformer_phoneme_q8.onnx"


def _ignore_upload(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts
        & {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "artifacts",
        }
    )


app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "librosa>=0.10",
        "numpy>=1.26",
        "onnxruntime>=1.16",
        "python-Levenshtein>=0.21",
        "pyyaml>=6.0",
        "soundfile>=0.12",
    )
    .add_local_dir(Path(__file__).resolve().parents[1], str(REPO_ROOT), copy=True, ignore=_ignore_upload)
)


def _expected_first(sample: dict[str, Any]) -> tuple[int, int] | None:
    expected = sample.get("expected_verses")
    if isinstance(expected, list) and expected and isinstance(expected[0], dict):
        return int(expected[0].get("surah", 0)), int(expected[0].get("ayah", 0))
    return None


def _load_experiment(experiment: str):
    import importlib.util

    run_py = REPO_ROOT / "experiments" / experiment / "run.py"
    spec = importlib.util.spec_from_file_location(f"modal_{experiment}", run_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment from {run_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={MODEL_CACHE_ROOT: model_volume},
)
def eval_shard(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    os.environ["PHONEME_ONNX_CACHE"] = str(ONNX_CACHE_PATH)
    model_volume.reload()

    experiment = str(payload["experiment"])
    corpus = str(payload["corpus"])
    sample_indices = [int(i) for i in payload["sample_indices"]]

    bench = REPO_ROOT / "benchmark" / corpus
    manifest = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    mod = _load_experiment(experiment)

    rows: list[dict[str, Any]] = []
    correct = 0
    failures = 0
    for idx in sample_indices:
        sample = samples[idx]
        expected = _expected_first(sample)
        try:
            pred = mod.predict(str(bench / str(sample.get("file", ""))))
            ok = bool(
                expected
                and int(pred.get("surah", 0)) == expected[0]
                and int(pred.get("ayah", 0)) == expected[1]
            )
            correct += int(ok)
            rows.append(
                {
                    "index": idx,
                    "id": sample.get("id"),
                    "expected": {"surah": expected[0], "ayah": expected[1]} if expected else None,
                    "predicted": {
                        "surah": pred.get("surah"),
                        "ayah": pred.get("ayah"),
                        "ayah_end": pred.get("ayah_end"),
                        "score": pred.get("score"),
                        "transcript": pred.get("transcript"),
                    },
                    "correct": ok,
                }
            )
        except Exception as exc:
            failures += 1
            rows.append({"index": idx, "id": sample.get("id"), "error": repr(exc), "correct": False})

    return {
        "shard": payload["shard"],
        "sample_count": len(sample_indices),
        "correct": correct,
        "failures": failures,
        "rows": rows,
    }


@app.local_entrypoint()
def main(
    experiment: str = "phoneme_matcher_joint04",
    corpus: str = "test_corpus_v3",
    limit: int = 32,
    indices: str = "",
    shards: int = 8,
    output: str = "",
) -> None:
    manifest_path = Path("benchmark") / corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_manifest_samples = len(manifest.get("samples", []))
    if indices.strip():
        selected = [int(part) for part in indices.replace(",", " ").split()]
    else:
        selected = list(range(total_manifest_samples if limit <= 0 else min(limit, total_manifest_samples)))
    shard_count = max(1, min(shards, len(selected) or 1))
    shard_payloads = [
        {
            "experiment": experiment,
            "corpus": corpus,
            "shard": i,
            "sample_indices": selected[i::shard_count],
        }
        for i in range(shard_count)
    ]

    shard_results = list(eval_shard.map(shard_payloads, order_outputs=True))
    rows = [row for shard in shard_results for row in shard["rows"]]
    rows.sort(key=lambda row: int(row["index"]))
    correct = sum(int(row.get("correct", False)) for row in rows)
    failures = sum(int(shard["failures"]) for shard in shard_results)
    report = {
        "tier": 2,
        "schema": "offline-tarteel.modal_tier2_report.v1",
        "experiment": experiment,
        "corpus": corpus,
        "manifest_samples": total_manifest_samples,
        "sample_limit": limit,
        "selected_samples": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected) if selected else 0.0,
        "failures": failures,
        "shards": shard_results,
        "rows": rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, indent=2))
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
