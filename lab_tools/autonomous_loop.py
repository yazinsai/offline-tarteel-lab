"""Autonomous lab controller.

The controller keeps the queue state durable while making local experiment evaluation
repeatable: claim a queued task, run tier gates, write a run record, judge it, then
promote or reject it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lab_tools.experiment_ledger import append_run_record, read_entries
from lab_tools.paths import lab_root
from lab_tools.judge_policy import JudgeInput, judge
from lab_tools.scorer import DEFAULT_WEIGHTS, latency_budget_score, score_metrics
from lab_tools.task_queue import Task, load_state, next_queued, set_status

MIN_OBJECTIVE_DELTA = 0.0001
MIN_PROMOTION_CORPUS_SAMPLES = 12
OBJECTIVE_COMPONENTS = tuple(DEFAULT_WEIGHTS.keys())


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


def _dataset_revision(corpus: str) -> str:
    registry_path = lab_root() / "datasets" / "registry.yaml"
    if not registry_path.is_file():
        return corpus
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return corpus
    for entry in registry.get("corpora") or []:
        if isinstance(entry, dict) and entry.get("id") == corpus:
            revision = entry.get("revision")
            return f"{corpus}@{revision}" if revision else corpus
    return corpus


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
    tier2_samples = best.get("samples")
    tier2_failures = best.get("failures")
    tier2_evaluated = best.get("evaluated_samples")
    if tier2_evaluated is None and tier2_samples is not None:
        try:
            tier2_evaluated = int(tier2_samples) + int(tier2_failures or 0)
        except (TypeError, ValueError):
            tier2_evaluated = tier2_samples
    manifest_samples = best.get("manifest_samples", tier2.get("manifest_samples"))
    metrics: dict[str, Any] = {
        "tier1_passed": tier1.get("passed"),
        "tier1_total": tier1.get("total"),
        "tier2_experiment": best.get("experiment"),
        "tier2_accuracy": accuracy,
        "tier2_correct": best.get("correct"),
        "tier2_samples": tier2_samples,
        "tier2_evaluated_samples": tier2_evaluated,
        "tier2_manifest_samples": manifest_samples,
        "tier2_selected_samples": tier2.get("selected_samples"),
        "tier2_sample_limit": tier2.get("sample_limit"),
        "tier2_failures": tier2_failures,
        "tier3_completed": bool(tier3.get("completed")),
        "baseline_accuracy": payload.get("baseline_accuracy", payload.get("baseline_recall")),
        "min_accuracy": payload.get("min_accuracy"),
        "onnx_mb": payload.get("onnx_mb"),
        "max_onnx_mb": payload.get("max_onnx_mb", 200),
    }
    if isinstance(best.get("metrics"), dict):
        metrics.update(best["metrics"])
    if isinstance(best.get("slices"), dict):
        metrics["slices"] = best["slices"]

    # judge_policy uses recall-oriented names; for the current local benchmark,
    # first-verse accuracy is the available proxy until richer streaming metrics land.
    if accuracy is not None:
        metrics["target_recall"] = accuracy
        metrics.setdefault("streaming_alignment_accuracy", accuracy)
        metrics.setdefault("correction_precision", accuracy)
        metrics.setdefault("verse_boundary_f1", accuracy)
    if metrics["baseline_accuracy"] is not None:
        metrics["baseline_recall"] = metrics["baseline_accuracy"]
    metrics["latency_budget_score"] = latency_budget_score(metrics)
    return metrics


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _full_corpus_coverage(metrics: dict[str, Any]) -> bool:
    evaluated = _number(metrics.get("tier2_evaluated_samples"))
    manifest = _number(metrics.get("tier2_manifest_samples"))
    return (
        evaluated is not None
        and manifest is not None
        and manifest >= MIN_PROMOTION_CORPUS_SAMPLES
        and evaluated >= manifest
    )


def _valid_full_corpus_champion(corpus: str) -> dict[str, Any] | None:
    corpus_revision = _dataset_revision(corpus)
    entries = read_entries(path=lab_root() / "artifacts" / "experiment_ledger.jsonl")
    invalidated = {
        str(entry.get("invalidates_run_id"))
        for entry in entries
        if entry.get("invalidates_run_id")
        and entry.get("status") in {"invalidated", "superseded", "reverted"}
    }
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") not in {"promoted", "accepted", "merged"}:
            continue
        if str(entry.get("run_id")) in invalidated:
            continue
        if entry.get("corpus_revision") != corpus_revision:
            continue
        params = entry.get("parameters") or {}
        if not params.get("full_corpus_gate"):
            continue
        objective = _number(entry.get("objective"))
        if objective is None:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda e: float(e.get("objective") or 0.0))


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
    if metrics.get("requires_full_corpus_gate") and not _full_corpus_coverage(metrics):
        out["accept"] = False
        out.setdefault("reasons", []).append("full_corpus_coverage_required")

    if metrics.get("requires_champion_improvement"):
        missing_components = [name for name in OBJECTIVE_COMPONENTS if metrics.get(name) is None]
        if missing_components:
            out["accept"] = False
            out.setdefault("reasons", []).extend(
                f"missing_objective_component:{name}" for name in missing_components
            )
        candidate_objective = _number(metrics.get("candidate_objective"))
        champion_objective = _number(metrics.get("champion_objective"))
        if candidate_objective is None:
            out["accept"] = False
            out.setdefault("reasons", []).append("missing_candidate_objective")
        elif champion_objective is not None and candidate_objective <= champion_objective + MIN_OBJECTIVE_DELTA:
            out["accept"] = False
            out.setdefault("reasons", []).append("champion_objective_not_improved")
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
        "dataset_revision": _dataset_revision(str((task.payload or {}).get("corpus", "test_corpus_v3"))),
        "experiment_kind": task.kind,
        "parameter_vector": task.payload or {},
        "metrics": metrics,
        "tier_completed": completed_tiers,
        "artifact_hashes": {},
        "commands": [c.display for c in commands],
    }
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


def _append_ledger(run_record: Path, *, status: str, decision: dict[str, Any] | None = None) -> None:
    try:
        append_run_record(
            run_record,
            status=status,
            decision=decision,
            path=lab_root() / "artifacts" / "experiment_ledger.jsonl",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"warning: failed to append experiment ledger: {exc}", file=sys.stderr)


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
        "--run-record",
        str(run_record),
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


def _maybe_launch_modal(task: Task, allow_modal: bool) -> CommandResult | None:
    payload = task.payload or {}
    if task.kind not in {"model_only", "joint_model_runtime"} or not payload.get("modal_training"):
        return None
    job_name = str(payload.get("job_name", task.id))
    cmd = ["modal", "run", "--detach", "training/train_fastconformer_phoneme_modal.py", "--job-name", job_name]
    if not allow_modal:
        print(
            f"modal training requested for {task.id}; rerun with --allow-modal to launch: {' '.join(cmd)}",
            file=sys.stderr,
        )
        return CommandResult(cmd=cmd, returncode=77)
    return _run(cmd)


def tick(dry_run: bool = False, *, shard_index: int = 0, shard_total: int = 1) -> int:
    t = next_queued(shard_index=shard_index, shard_total=shard_total)
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
    shard_index: int = 0,
    shard_total: int = 1,
) -> int:
    task = next_queued(shard_index=shard_index, shard_total=shard_total)
    if not task:
        print("no queued tasks")
        return 0

    payload = dict(task.payload or {})
    payload.setdefault("corpus", corpus)
    if promote:
        payload["full_corpus_gate"] = True
    task.payload = payload
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
        _append_ledger(run_record, status="rejected", decision={"accept": False, "reasons": ["tier1_failed"]})
        set_status(
            task.id,
            "rejected",
            run_record_path=str(run_record),
            judge_reasons=["tier1_failed"],
        )
        return tier1.returncode

    # Promotion decisions must be based on the complete requested corpus. The
    # caller's limit is still useful for tier-1 smoke, but tier-2 is the scoring gate.
    tier2_limit = 0 if promote else limit
    tier2_cmd = [
        sys.executable,
        "-m",
        "lab_tools.eval_tier",
        "--tier",
        "2",
        "--corpus",
        str(payload["corpus"]),
        "--limit",
        str(tier2_limit),
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
        _append_ledger(run_record, status="rejected", decision={"accept": False, "reasons": ["tier2_failed"]})
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
    metrics["requires_full_corpus_gate"] = bool(promote)
    metrics["requires_champion_improvement"] = bool(promote)
    metrics["candidate_objective"] = score_metrics(metrics)["objective"]
    champion = _valid_full_corpus_champion(str(payload["corpus"]))
    metrics["champion_run_id"] = champion.get("run_id") if champion else None
    metrics["champion_objective"] = champion.get("objective") if champion else None
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
        _append_ledger(run_record, status="promoted", decision=decision)
        set_status(task.id, "promoted", run_record_path=str(run_record), judge_reasons=[])
        print(json.dumps({"task": task.id, "accepted": True, "run_record": str(run_record)}, indent=2))
        return 0

    reasons = list(decision.get("reasons", []))
    _append_ledger(run_record, status="rejected", decision=decision)
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
    shard_index: int = 0,
    shard_total: int = 1,
) -> int:
    cycles = 0
    failures = 0
    while max_cycles <= 0 or cycles < max_cycles:
        if next_queued(shard_index=shard_index, shard_total=shard_total) is None:
            print("no queued tasks")
            return 0 if failures == 0 else 1
        rc = run_once(
            allow_modal=allow_modal,
            corpus=corpus,
            limit=limit,
            promote=promote,
            shard_index=shard_index,
            shard_total=shard_total,
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
    sp_tick.add_argument("--shard-index", type=int, default=0)
    sp_tick.add_argument("--shard-total", type=int, default=1)

    sub.add_parser("status", help="Print queue counts by status")

    sp_run = sub.add_parser("run-once", help="Claim and evaluate one queued task locally")
    sp_run.add_argument("--dry-run", action="store_true")
    sp_run.add_argument("--allow-modal", action="store_true")
    sp_run.add_argument("--corpus", default="test_corpus_v3")
    sp_run.add_argument("--limit", type=int, default=12)
    sp_run.add_argument("--no-promote", action="store_true")
    sp_run.add_argument("--shard-index", type=int, default=0)
    sp_run.add_argument("--shard-total", type=int, default=1)

    sp_loop = sub.add_parser("run", help="Continuously evaluate queued tasks")
    sp_loop.add_argument("--max-cycles", type=int, default=0, help="0 means until queue is empty")
    sp_loop.add_argument("--sleep-seconds", type=float, default=0)
    sp_loop.add_argument("--allow-modal", action="store_true")
    sp_loop.add_argument("--corpus", default="test_corpus_v3")
    sp_loop.add_argument("--limit", type=int, default=12)
    sp_loop.add_argument("--no-promote", action="store_true")
    sp_loop.add_argument("--shard-index", type=int, default=0)
    sp_loop.add_argument("--shard-total", type=int, default=1)

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
                shard_index=args.shard_index,
                shard_total=args.shard_total,
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
                shard_index=args.shard_index,
                shard_total=args.shard_total,
            ),
        )
    sys.exit(tick(dry_run=args.dry_run, shard_index=args.shard_index, shard_total=args.shard_total))


if __name__ == "__main__":
    main()
