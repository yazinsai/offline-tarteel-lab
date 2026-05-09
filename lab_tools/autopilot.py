"""Autopilot task generator for unattended lab progress."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from lab_tools.experiment_ledger import champion, failed_families, read_entries, worst_slice
from lab_tools.task_queue import (
    add_task_once,
    count_active,
    load_state,
    queue_lock,
    save_state,
    state_path,
)


@dataclass(frozen=True)
class Candidate:
    key: str
    kind: str
    title: str
    payload: dict[str, Any]


def _key_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))


def candidates() -> list[Candidate]:
    static = [
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
    entries = read_entries()
    return ledger_guided_candidates(entries) + static + adaptive_runtime_candidates()


def ledger_guided_candidates(entries: list[dict[str, Any]] | None = None) -> list[Candidate]:
    entries = entries if entries is not None else read_entries()
    champ = champion(entries)
    weak = worst_slice(entries)
    out: list[Candidate] = []

    if weak and champ:
        slice_name, slice_data = weak
        run_token = _key_token(champ.get("run_id"))
        out.append(
            Candidate(
                key=f"runtime.repair_slice.{_key_token(slice_name)}.{run_token}",
                kind="runtime_only",
                title=f"Repair weak corpus slice: {slice_name}",
                payload={
                    "experiment": (champ.get("parameters") or {}).get("experiment", "smoke"),
                    "target_slice": slice_name,
                    "target_slice_score": slice_data.get("score"),
                    "champion_run_id": champ.get("run_id"),
                    "baseline_objective": champ.get("objective"),
                    "min_accuracy": 0.8,
                    "agent_instructions": (
                        f"Optimize corpus v3 slice '{slice_name}' without regressing the champion. "
                        "Inspect manifest tags and per-sample failures for this slice, make a small "
                        "runtime-only matcher/tracker change, then emit scorer-compatible slice metrics."
                    ),
                },
            ),
        )

    if champ:
        params = dict(champ.get("parameters") or {})
        family = _key_token(champ.get("experiment_family") or "champion")
        run_token = _key_token(champ.get("run_id"))
        out.append(
            Candidate(
                key=f"runtime.exploit_champion.{family}.{run_token}",
                kind="runtime_only",
                title="Exploit current champion with local runtime mutation",
                payload={
                    "experiment": params.get("experiment", "smoke"),
                    "champion_run_id": champ.get("run_id"),
                    "champion_parameters": params,
                    "baseline_objective": champ.get("objective"),
                    "min_accuracy": max(0.8, float(champ.get("objective") or 0.0)),
                    "agent_instructions": (
                        "Start from the champion parameter vector and try a narrow deterministic "
                        "runtime mutation. Only keep the change if the scorer objective improves beyond "
                        "variance and no critical slice regresses."
                    ),
                },
            ),
        )

    out.append(
        Candidate(
            key="runtime.explore_diverse.v3",
            kind="runtime_only",
            title="Explore a diverse corpus-v3 runtime strategy",
            payload={
                "experiment": "smoke",
                "exploration": True,
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Try one runtime-only strategy that is structurally different from the current "
                    "champion and recent rejected families. Keep changes small, deterministic, and "
                    "scored with the canonical objective."
                ),
            },
        ),
    )
    return out


def adaptive_runtime_candidates() -> list[Candidate]:
    focuses = [
        ("chunk_seconds", "Tune streaming chunk duration"),
        ("overlap_seconds", "Tune streaming chunk overlap"),
        ("FIRST_MATCH_THRESHOLD", "Tune first-match threshold"),
        ("VERSE_MATCH_THRESHOLD", "Tune verse-match threshold"),
        ("smoothing_window", "Tune streaming smoothing window"),
        ("correction_hysteresis", "Tune correction hysteresis"),
        ("partial_match_margin", "Tune partial-match margin"),
        ("debounce_ms", "Tune correction debounce"),
    ]
    out: list[Candidate] = []
    for i in range(1, 25):
        param, title = focuses[(i - 1) % len(focuses)]
        out.append(
            Candidate(
                key=f"runtime.adaptive.{param}.{i:02d}",
                kind="runtime_only",
                title=f"{title} variant {i:02d}",
                payload={
                    "experiment": "smoke",
                    "param": param,
                    "min_accuracy": 0.8,
                    "agent_instructions": (
                        "Create a small deterministic runtime experiment variant under "
                        "experiments/ with any supporting tests or benchmark JSON metadata. "
                        "Use artifacts/autonomy_failures as negative memory and avoid repeating "
                        "blocked paths. Do not touch training/, lab_tools/, orchestration/, "
                        ".github/, pyproject.toml, or generated model/audio artifacts."
                    ),
                },
            ),
        )
    return out


def _failure_records() -> list[dict[str, Any]]:
    root = state_path().parent.parent / "autonomy_failures"
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _task_ids_from_failure(record: dict[str, Any]) -> set[str]:
    task_ids: set[str] = set()
    for filename in record.get("changed_files", []):
        task_ids.update(re.findall(r"(task-[0-9a-f]{12})", str(filename)))
    return task_ids


def _retire_repeatedly_blocked_tasks(*, threshold: int = 2) -> list[str]:
    failures = _failure_records()
    if not failures:
        return []

    by_task: Counter[str] = Counter()
    blocked_non_runtime = 0
    for record in failures:
        changed = [str(p) for p in record.get("changed_files", [])]
        if any(
            p == "pyproject.toml"
            or p.startswith((".github/", "lab_tools/", "orchestration/", "training/"))
            for p in changed
        ):
            blocked_non_runtime += 1
        for task_id in _task_ids_from_failure(record):
            by_task[task_id] += 1

    retired: list[str] = []
    with queue_lock():
        state = load_state()
        changed = False
        for task in state.tasks:
            if task.status not in {"queued", "running", "needs_eval"}:
                continue
            repeated_task_failure = by_task[task.id] >= threshold
            unsafe_family = task.kind in {"model_only", "joint_model_runtime"} and blocked_non_runtime >= threshold
            if not (repeated_task_failure or unsafe_family):
                continue
            task.status = "rejected"
            task.judge_reasons = ["autopilot_failure_memory_retired"]
            task.notes = (
                "Retired by autopilot after repeated autonomous merge-gate failures. "
                "Future work should be split into runtime-only auto-merge tasks or normal "
                "human-review PRs for training/plumbing changes."
            )
            task.touch()
            retired.append(task.id)
            changed = True
        if changed:
            save_state(state)
    return retired


def plan(target_backlog: int) -> dict[str, Any]:
    retired = _retire_repeatedly_blocked_tasks()
    added: list[str] = []
    entries = read_entries()
    blocked_families = failed_families(entries)
    champ = champion(entries)
    weak = worst_slice(entries)
    active = count_active()
    for candidate in candidates():
        if candidate.key in blocked_families:
            continue
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
        "retired": retired,
        "total_tasks": len(state.tasks),
        "queued": sum(1 for t in state.tasks if t.status == "queued"),
        "champion": {
            "run_id": champ.get("run_id"),
            "objective": champ.get("objective"),
            "family": champ.get("experiment_family"),
        }
        if champ
        else None,
        "worst_slice": {"name": weak[0], "score": weak[1].get("score")} if weak else None,
        "blocked_families": sorted(blocked_families),
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
