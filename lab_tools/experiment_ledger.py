"""Append-only experiment memory for autopilot planning."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab_tools.paths import lab_root
from lab_tools.scorer import score_run_record

LEDGER_SCHEMA = "offline-tarteel.experiment_ledger.v1"


def ledger_path() -> Path:
    path = lab_root() / "artifacts" / "experiment_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_entries(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or ledger_path()
    if not p.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("schema") == LEDGER_SCHEMA:
            entries.append(entry)
    return entries


def append_entry(entry: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    p = path or ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(entry)
    body.setdefault("schema", LEDGER_SCHEMA)
    body.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body, sort_keys=True) + "\n")
    return body


def _artifact_ref(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(lab_root().resolve()))
    except ValueError:
        return str(path)


def _status_from_decision(status: str, decision: dict[str, Any] | None) -> str:
    if status != "judged":
        return status
    return "promoted" if (decision or {}).get("accept") else "rejected"


def entry_from_run_record(
    record: dict[str, Any],
    *,
    run_record_path: Path | None = None,
    status: str = "judged",
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = score_run_record(record)
    params = record.get("parameter_vector") or {}
    artifacts = dict(record.get("artifact_hashes") or {})
    if run_record_path is not None:
        artifacts["run_record"] = _artifact_ref(run_record_path)

    return {
        "schema": LEDGER_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": record.get("run_id"),
        "task_id": record.get("task_id"),
        "status": _status_from_decision(status, decision),
        "decision": decision or {},
        "experiment_kind": record.get("experiment_kind"),
        "experiment_family": params.get("autopilot_key") or params.get("param") or record.get("experiment_kind"),
        "parameters": params,
        "git_sha": record.get("git_sha"),
        "corpus_revision": record.get("dataset_revision"),
        "score": score,
        "objective": score["objective"],
        "components": score["components"],
        "slices": score["slices"],
        "artifacts": artifacts,
        "failure_modes": list((decision or {}).get("reasons") or []),
    }


def append_run_record(
    run_record_path: Path,
    *,
    status: str = "judged",
    decision: dict[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    record = json.loads(run_record_path.read_text(encoding="utf-8"))
    return append_entry(
        entry_from_run_record(
            record,
            run_record_path=run_record_path,
            status=status,
            decision=decision,
        ),
        path=path,
    )


def champion(entries: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    rows = [
        e
        for e in (entries if entries is not None else read_entries())
        if e.get("status") in {"promoted", "accepted", "merged"} and e.get("objective") is not None
    ]
    if not rows:
        return None
    return max(rows, key=lambda e: float(e.get("objective") or 0.0))


def failed_families(entries: list[dict[str, Any]] | None = None, *, threshold: int = 2) -> set[str]:
    counts: dict[str, int] = {}
    for entry in entries if entries is not None else read_entries():
        family = entry.get("experiment_family")
        if not family or entry.get("status") not in {"rejected", "failed", "superseded"}:
            continue
        counts[str(family)] = counts.get(str(family), 0) + 1
    return {family for family, count in counts.items() if count >= threshold}


def worst_slice(entries: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, Any]] | None:
    champ = champion(entries)
    if not champ:
        return None
    slices = champ.get("slices") or {}
    scored = [(name, data) for name, data in slices.items() if isinstance(data, dict)]
    if not scored:
        return None
    return min(scored, key=lambda item: float(item[1].get("score") or 0.0))


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment ledger utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_append = sub.add_parser("append-run-record")
    sp_append.add_argument("run_record", type=Path)
    sp_append.add_argument("--status", default="judged")
    sp_append.add_argument("--decision", default="{}", help="JSON decision object")

    sub.add_parser("champion")
    sub.add_parser("list")

    args = p.parse_args()
    if args.cmd == "append-run-record":
        entry = append_run_record(
            args.run_record,
            status=args.status,
            decision=json.loads(args.decision),
        )
        print(json.dumps(entry, indent=2))
        return
    if args.cmd == "champion":
        print(json.dumps(champion() or {}, indent=2))
        return
    if args.cmd == "list":
        for entry in read_entries():
            print(
                "\t".join(
                    [
                        str(entry.get("run_id") or ""),
                        str(entry.get("status") or ""),
                        str(entry.get("objective") or ""),
                        str(entry.get("experiment_family") or ""),
                    ],
                ),
            )
        return


if __name__ == "__main__":
    main()
