"""Autopilot task generator for unattended lab progress."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from lab_tools.task_queue import add_task_once, count_active, load_state


@dataclass(frozen=True)
class Candidate:
    key: str
    kind: str
    title: str
    payload: dict[str, Any]


def candidates() -> list[Candidate]:
    return [
        Candidate(
            key="runtime.smoke_streaming_baseline",
            kind="runtime_only",
            title="Improve streaming first-verse baseline",
            payload={
                "experiment": "smoke",
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Inspect the current experiments and benchmark manifests. Create or update a "
                    "small runnable experiment under experiments/ that improves first-verse "
                    "streaming accuracy without adding large artifacts. Keep it deterministic and "
                    "compatible with lab_tools.tier2_local predict(audio_path)."
                ),
            },
        ),
        Candidate(
            key="runtime.threshold_sweep.first_match",
            kind="runtime_only",
            title="Tune first-match streaming threshold",
            payload={
                "param": "FIRST_MATCH_THRESHOLD",
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Search for runtime threshold logic or create an experiment that models a "
                    "first-match threshold sweep. Evaluate multiple candidates locally and commit "
                    "the best deterministic experiment only if it improves Tier-2 accuracy."
                ),
            },
        ),
        Candidate(
            key="runtime.chunk_window_sweep",
            kind="runtime_only",
            title="Tune streaming chunk/window size",
            payload={
                "param": "chunk_seconds",
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Explore chunk/window parameters for streaming recitation tracking. Prefer "
                    "small code changes and record the selected parameter vector in the task "
                    "payload or experiment metadata."
                ),
            },
        ),
        Candidate(
            key="model.fastconformer_phoneme_smoke",
            kind="model_only",
            title="Evaluate phoneme FastConformer training candidate",
            payload={
                "modal_training": True,
                "job_name": "fastconformer-phoneme-autopilot",
                "min_accuracy": 0.85,
                "max_onnx_mb": 200,
                "agent_instructions": (
                    "Only launch Modal when LAB_AUTONOMY_ALLOW_MODAL is enabled. Otherwise inspect "
                    "the training entrypoint and prepare a small, reviewable training/eval "
                    "improvement that keeps local tests passing. Never commit model binaries."
                ),
            },
        ),
        Candidate(
            key="joint.model_runtime_export_contract",
            kind="joint_model_runtime",
            title="Tighten ONNX export and streaming wrapper contract",
            payload={
                "min_accuracy": 0.85,
                "max_onnx_mb": 200,
                "agent_instructions": (
                    "Improve the path from a candidate model artifact to an offline streaming "
                    "consumer wrapper. Add tests or metadata checks for ONNX size/hash and package "
                    "compatibility. Do not commit generated model artifacts."
                ),
            },
        ),
    ]


def plan(target_backlog: int) -> dict[str, Any]:
    added: list[str] = []
    active = count_active()
    for candidate in candidates():
        if active >= target_backlog:
            break
        task = add_task_once(
            candidate.kind,
            candidate.title,
            candidate.payload,
            key=candidate.key,
        )
        if task is not None:
            active += 1
            added.append(task.id)
    state = load_state()
    return {
        "active": active,
        "added": added,
        "total_tasks": len(state.tasks),
        "queued": sum(1 for t in state.tasks if t.status == "queued"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Seed autonomous lab tasks when backlog is low")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan")
    sp.add_argument("--target-backlog", type=int, default=3)
    args = p.parse_args()

    if args.cmd == "plan":
        print(json.dumps(plan(args.target_backlog), indent=2))


if __name__ == "__main__":
    main()
