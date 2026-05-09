"""Autonomous lab controller.

The controller keeps the queue state durable while making local experiment evaluation
repeatable: claim a queued task, run tier gates, write a run record, judge it, then
promote or reject it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab_tools.paths import lab_root
from lab_tools.judge_policy import JudgeInput, judge
from lab_tools.task_queue import Task, load_state, next_queued, set_status


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int

    @property
    def display(self) -> str:
        return " ".join(self.cmd)


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=lab_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _latest_report(tier: int) -> Path | None:
    art = lab_root() / "artifacts" / f"tier{tier}"
    if not art.is_dir():
        return None
    files = sorted(art.glob(f"tier{tier}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _run(cmd: list[str]) -> CommandResult:
    print("+", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=lab_root())
    return CommandResult(cmd=cmd, returncode=r.returncode)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_tier2_result(tier2: dict[str, Any], experiment: str | None) -> dict[str, Any]:
    rows = tier2.get("results", []) if isinstance(tier2, dict) else []
    if experiment:
        rows = [r for r in rows if r.get("experiment") == experiment]
    if not rows:
        return {}
    return max(rows, key=lambda r: float(r.get("accuracy", 0.0)))


def _metrics_from_reports(
    task: Task,
    tier1_path: Path | None,
    tier2_path: Path | None,
    tier3_path: Path | None,
) -> dict[str, Any]:
    payload = task.payload or {}
    tier1 = _read_json(tier1_path)
    tier2 = _read_json(tier2_path)
    tier3 = _read_json(tier3_path)
    experiment = payload.get("experiment")
    best = _best_tier2_result(tier2, str(experiment) if experiment else None)

    accuracy = best.get("accuracy")
    metrics: dict[str, Any] = {
        "tier1_passed": tier1.get("passed"),
        "tier1_total": tier1.get("total"),
        "tier2_experiment": best.get("experiment"),
        "tier2_accuracy": accuracy,
        "tier2_correct": best.get("correct"),
        "tier2_samples": best.get("samples"),
        "tier2_failures": best.get("failures"),
        "tier3_completed": bool(tier3.get("completed")),
        "baseline_accuracy": payload.get("baseline_accuracy", payload.get("baseline_recall")),
        "min_accuracy": payload.get("min_accuracy"),
        "onnx_mb": payload.get("onnx_mb"),
        "max_onnx_mb": payload.get("max_onnx_mb", 200),
    }

    # judge_policy uses recall-oriented names; for the current local benchmark,
    # first-verse accuracy is the available proxy until richer streaming metrics land.
    if accuracy is not None:
        metrics["target_recall"] = accuracy
    if metrics["baseline_accuracy"] is not None:
        metrics["baseline_recall"] = metrics["baseline_accuracy"]
    return metrics


def _judge_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = judge(
        JudgeInput(
            target_recall=metrics.get("target_recall"),
            baseline_recall=metrics.get("baseline_recall"),
            target_precision=metrics.get("target_precision"),
            baseline_precision=metrics.get("baseline_precision"),
            blind_recall=metrics.get("blind_recall"),
            blind_baseline_recall=metrics.get("blind_baseline_recall"),
            max_onnx_mb=float(metrics.get("max_onnx_mb", 200)),
            onnx_mb=metrics.get("onnx_mb"),
            tier3_completed=bool(metrics.get("tier3_completed")),
        ),
    )
    min_accuracy = metrics.get("min_accuracy")
    target = metrics.get("tier2_accuracy")
    if min_accuracy is not None and (target is None or float(target) + 1e-6 < float(min_accuracy)):
        out["accept"] = False
        out.setdefault("reasons", []).append("min_accuracy_not_met")
    if target is None:
        out["accept"] = False
        out.setdefault("reasons", []).append("missing_tier2_accuracy")
    return out


def _write_run_record(
    task: Task,
    metrics: dict[str, Any],
    completed_tiers: list[int],
    commands: list[CommandResult],
) -> Path:
    run_id = f"{task.id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = lab_root() / "artifacts" / "runs" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "offline-tarteel.run_record.v1",
        "run_id": run_id,
        "task_id": task.id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "dataset_revision": str((task.payload or {}).get("corpus", "test_corpus_v3")),
        "experiment_kind": task.kind,
        "parameter_vector": task.payload or {},
        "metrics": metrics,
        "tier_completed": completed_tiers,
        "artifact_hashes": {},
        "commands": [c.display for c in commands],
    }
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


def _promote_run(run_record: Path, tier1: Path | None, tier3: Path | None) -> tuple[int, Path | None]:
    record = _read_json(run_record)
    out_dir = lab_root() / "artifacts" / "promotions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "lab_tools.promote",
        "--run-id",
        str(record.get("run_id", run_record.stem)),
        "--git-sha",
        str(record.get("git_sha", "")),
        "--output",
        str(out_dir),
    ]
    if tier1:
        cmd += ["--tier1-report", str(tier1)]
    if tier3:
        cmd += ["--tier3-report", str(tier3)]
    result = _run(cmd)
    if result.returncode != 0:
        return result.returncode, None
    promotions = sorted(
        out_dir.glob(f"promotion-{record.get('run_id', run_record.stem)}-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return result.returncode, promotions[0] if promotions else None


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _maybe_launch_modal(task: Task, allow_modal: bool) -> CommandResult | None:
    payload = task.payload or {}
    if task.kind not in {"model_only", "joint_model_runtime"} or not payload.get("modal_training"):
        return None
    job_name = str(payload.get("job_name", task.id))
    cmd = ["modal", "run", "--detach", "training/train_fastconformer_phoneme_modal.py", "--job-name", job_name]
    if not allow_modal or not _env_truthy("LAB_AUTONOMY_ALLOW_MODAL"):
        print(
            f"modal training skipped for {task.id} (need --allow-modal and LAB_AUTONOMY_ALLOW_MODAL=1): "
            f"{' '.join(cmd)}",
            file=sys.stderr,
        )
        return CommandResult(cmd=cmd, returncode=77)
    return _run(cmd)


def tick(dry_run: bool = False) -> int:
    t = next_queued()
    if not t:
        print("no queued tasks")
        return 0
    if dry_run:
        print(json.dumps({"would_claim": t.id, "title": t.title, "kind": t.kind}, indent=2))
        return 0
    set_status(t.id, "running")
    orch = lab_root() / "orchestration"
    prompt = (
        f"Lab task {t.id} ({t.kind}): {t.title}\n"
        f"Payload: {json.dumps(t.payload)}\n"
        "1) Implement or tune per payload. 2) Run: python -m lab_tools.eval_tier --tier 1\n"
        "3) Then tier 2/3 as needed. 4) Write run record JSON under lab artifacts/. "
        "5) lab-task set-status --id ... --status needs_eval --run-record path"
    )
    print("DISPATCH_HINT:\n", prompt)
    if (orch / "node_modules").is_dir():
        print("\nOptional: cd orchestration && npx tsx src/dispatch.ts <prompt>")
    return 0


def run_once(
    *,
    dry_run: bool = False,
    allow_modal: bool = False,
    corpus: str = "test_corpus_v3",
    limit: int = 12,
    promote: bool = True,
) -> int:
    task = next_queued()
    if not task:
        print("no queued tasks")
        return 0

    payload = dict(task.payload or {})
    payload.setdefault("corpus", corpus)
    experiment = payload.get("experiment")

    if dry_run:
        print(
            json.dumps(
                {
                    "would_run": task.id,
                    "kind": task.kind,
                    "title": task.title,
                    "payload": payload,
                    "tiers": [1, 2, 3],
                    "allow_modal": allow_modal,
                },
                indent=2,
            ),
        )
        return 0

    set_status(task.id, "running")
    commands: list[CommandResult] = []
    completed: list[int] = []

    modal_result = _maybe_launch_modal(task, allow_modal)
    if modal_result is not None:
        commands.append(modal_result)
        if modal_result.returncode not in {0, 77}:
            set_status(task.id, "rejected", judge_reasons=["modal_launch_failed"])
            return modal_result.returncode

    tier1 = _run(
        [
            sys.executable,
            "-m",
            "lab_tools.eval_tier",
            "--tier",
            "1",
            "--corpus",
            str(payload["corpus"]),
            "--limit",
            str(limit),
        ],
    )
    commands.append(tier1)
    tier1_path = _latest_report(1)
    if tier1.returncode == 0:
        completed.append(1)
    else:
        metrics = _metrics_from_reports(task, tier1_path, None, None)
        run_record = _write_run_record(task, metrics, completed, commands)
        set_status(
            task.id,
            "rejected",
            run_record_path=str(run_record),
            judge_reasons=["tier1_failed"],
        )
        return tier1.returncode

    tier2_cmd = [
        sys.executable,
        "-m",
        "lab_tools.eval_tier",
        "--tier",
        "2",
        "--corpus",
        str(payload["corpus"]),
        "--limit",
        str(limit),
    ]
    if experiment:
        tier2_cmd += ["--experiment", str(experiment)]
    tier2 = _run(tier2_cmd)
    commands.append(tier2)
    tier2_path = _latest_report(2)
    if tier2.returncode == 0:
        completed.append(2)
    else:
        metrics = _metrics_from_reports(task, tier1_path, tier2_path, None)
        run_record = _write_run_record(task, metrics, completed, commands)
        set_status(
            task.id,
            "rejected",
            run_record_path=str(run_record),
            judge_reasons=["tier2_failed"],
        )
        return tier2.returncode

    tier3 = _run([sys.executable, "-m", "lab_tools.eval_tier", "--tier", "3"])
    commands.append(tier3)
    tier3_path = _latest_report(3)
    if tier3.returncode == 0:
        completed.append(3)

    metrics = _metrics_from_reports(task, tier1_path, tier2_path, tier3_path)
    decision = _judge_from_metrics(metrics)
    run_record = _write_run_record(task, metrics, completed, commands)

    if decision["accept"]:
        if promote:
            rc, promotion_record = _promote_run(run_record, tier1_path, tier3_path)
            if rc != 0:
                set_status(
                    task.id,
                    "judged",
                    run_record_path=str(run_record),
                    judge_reasons=["promotion_record_failed"],
                )
                return rc
            if promotion_record:
                print(json.dumps({"promotion_record": str(promotion_record)}, indent=2))
        set_status(task.id, "promoted", run_record_path=str(run_record), judge_reasons=[])
        print(json.dumps({"task": task.id, "accepted": True, "run_record": str(run_record)}, indent=2))
        return 0

    reasons = list(decision.get("reasons", []))
    set_status(task.id, "rejected", run_record_path=str(run_record), judge_reasons=reasons)
    print(
        json.dumps(
            {"task": task.id, "accepted": False, "reasons": reasons, "run_record": str(run_record)},
            indent=2,
        ),
    )
    return 1


def run_loop(
    *,
    max_cycles: int,
    sleep_seconds: float,
    allow_modal: bool,
    corpus: str,
    limit: int,
    promote: bool,
) -> int:
    cycles = 0
    failures = 0
    while max_cycles <= 0 or cycles < max_cycles:
        if next_queued() is None:
            print("no queued tasks")
            return 0 if failures == 0 else 1
        rc = run_once(
            allow_modal=allow_modal,
            corpus=corpus,
            limit=limit,
            promote=promote,
        )
        cycles += 1
        if rc != 0:
            failures += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return 0 if failures == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous loop (single tick)")
    sub = p.add_subparsers(dest="command", required=True)

    sp_tick = sub.add_parser("tick", help="Claim one queued task and print dispatch prompt")
    sp_tick.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Print queue counts by status")

    sp_run = sub.add_parser("run-once", help="Claim and evaluate one queued task locally")
    sp_run.add_argument("--dry-run", action="store_true")
    sp_run.add_argument("--allow-modal", action="store_true")
    sp_run.add_argument("--corpus", default="test_corpus_v3")
    sp_run.add_argument("--limit", type=int, default=12)
    sp_run.add_argument("--no-promote", action="store_true")

    sp_loop = sub.add_parser("run", help="Continuously evaluate queued tasks")
    sp_loop.add_argument("--max-cycles", type=int, default=0, help="0 means until queue is empty")
    sp_loop.add_argument("--sleep-seconds", type=float, default=0)
    sp_loop.add_argument("--allow-modal", action="store_true")
    sp_loop.add_argument("--corpus", default="test_corpus_v3")
    sp_loop.add_argument("--limit", type=int, default=12)
    sp_loop.add_argument("--no-promote", action="store_true")

    args = p.parse_args()
    if args.command == "status":
        state = load_state()
        by = {}
        for t in state.tasks:
            by[t.status] = by.get(t.status, 0) + 1
        print(json.dumps(by, indent=2))
        return
    if args.command == "run-once":
        sys.exit(
            run_once(
                dry_run=args.dry_run,
                allow_modal=args.allow_modal,
                corpus=args.corpus,
                limit=args.limit,
                promote=not args.no_promote,
            ),
        )
    if args.command == "run":
        sys.exit(
            run_loop(
                max_cycles=args.max_cycles,
                sleep_seconds=args.sleep_seconds,
                allow_modal=args.allow_modal,
                corpus=args.corpus,
                limit=args.limit,
                promote=not args.no_promote,
            ),
        )
    sys.exit(tick(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
