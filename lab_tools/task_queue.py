"""Experiment task queue + state machine (queued → running → needs_eval → judged → promoted|rejected)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from lab_tools.paths import lab_root

TaskStatus = Literal[
    "queued",
    "running",
    "needs_eval",
    "judged",
    "promoted",
    "rejected",
]


@dataclass
class Task:
    id: str
    status: TaskStatus
    kind: str  # model_only | runtime_only | joint_model_runtime
    title: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    cursor_agent_id: str | None = None
    cursor_run_id: str | None = None
    run_record_path: str | None = None
    judge_reasons: list[str] | None = None
    notes: str = ""

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class QueueState:
    schema: str = "offline-tarteel.task_queue.v1"
    tasks: list[Task] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"schema": self.schema, "tasks": [asdict(t) for t in self.tasks]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> QueueState:
        tasks = [Task(**t) for t in data.get("tasks", [])]
        return cls(schema=data.get("schema", cls.schema), tasks=tasks)


def state_path() -> Path:
    p = lab_root() / "artifacts" / "queue" / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextlib.contextmanager
def queue_lock():
    path = state_path().with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_state() -> QueueState:
    path = state_path()
    if not path.is_file():
        return QueueState()
    return QueueState.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_state(state: QueueState) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state.to_json(), indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_task(kind: str, title: str, payload: dict[str, Any] | None = None) -> Task:
    now = datetime.now(timezone.utc).isoformat()
    return Task(
        id=f"task-{uuid.uuid4().hex[:12]}",
        status="queued",
        kind=kind,
        title=title,
        payload=payload or {},
        created_at=now,
        updated_at=now,
    )


def add_task(
    kind: str,
    title: str,
    payload: dict[str, Any] | None = None,
) -> Task:
    with queue_lock():
        state = load_state()
        t = _build_task(kind, title, payload)
        state.tasks.append(t)
        save_state(state)
        return t


def add_task_once(
    kind: str,
    title: str,
    payload: dict[str, Any] | None = None,
    *,
    key: str,
) -> Task | None:
    """Add a task unless a task with the same autopilot key already exists."""
    with queue_lock():
        state = load_state()
        for existing in state.tasks:
            if (existing.payload or {}).get("autopilot_key") == key:
                return None
        body = dict(payload or {})
        body["autopilot_key"] = key
        task = _build_task(kind, title, body)
        state.tasks.append(task)
        save_state(state)
        return task


def count_active() -> int:
    state = load_state()
    return sum(1 for t in state.tasks if t.status in {"queued", "running", "needs_eval"})


def next_queued() -> Task | None:
    state = load_state()
    for t in state.tasks:
        if t.status == "queued":
            return t
    return None


def set_status(task_id: str, status: TaskStatus, **extra: Any) -> bool:
    with queue_lock():
        state = load_state()
        for t in state.tasks:
            if t.id == task_id:
                t.status = status
                for k, v in extra.items():
                    if hasattr(t, k):
                        setattr(t, k, v)
                t.touch()
                save_state(state)
                return True
        return False


def seed_runtime_sweeps() -> int:
    """Enqueue a small set of runtime-only sweep placeholders (hypothesis params via env)."""
    ideas = [
        ("runtime_only", "Sweep chunk 0.25-0.35s", {"param": "chunk_seconds"}, "runtime.chunk_seconds"),
        (
            "runtime_only",
            "Sweep FIRST_MATCH_THRESHOLD +/-0.05",
            {"param": "FIRST_MATCH_THRESHOLD"},
            "runtime.first_match_threshold",
        ),
        (
            "runtime_only",
            "Sweep VERSE_MATCH_THRESHOLD +/-0.05",
            {"param": "VERSE_MATCH_THRESHOLD"},
            "runtime.verse_match_threshold",
        ),
    ]
    n = 0
    for kind, title, payload, key in ideas:
        if add_task_once(kind, title, payload, key=key) is not None:
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Lab task queue")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Ensure empty state file exists")

    sp = sub.add_parser("add", help="Add task")
    sp.add_argument("--kind", required=True)
    sp.add_argument("--title", required=True)
    sp.add_argument("--payload", default="{}", help="JSON object")

    sub.add_parser("list", help="List all tasks")
    sub.add_parser("next", help="Print next queued task id")

    sp = sub.add_parser("set-status")
    sp.add_argument("--id", required=True)
    sp.add_argument(
        "--status",
        required=True,
        choices=["queued", "running", "needs_eval", "judged", "promoted", "rejected"],
    )
    sp.add_argument("--cursor-run-id")
    sp.add_argument("--run-record")

    sub.add_parser("seed-sweeps", help="Enqueue default runtime sweep tasks")

    args = p.parse_args()

    if args.cmd == "init":
        save_state(load_state())
        print(state_path())
        return

    if args.cmd == "add":
        payload = json.loads(args.payload)
        t = add_task(args.kind, args.title, payload)
        print(t.id)
        return

    if args.cmd == "list":
        state = load_state()
        for t in state.tasks:
            print(f"{t.id}\t{t.status}\t{t.kind}\t{t.title}")
        return

    if args.cmd == "next":
        t = next_queued()
        print(t.id if t else "")
        return

    if args.cmd == "set-status":
        extra = {}
        if args.cursor_run_id:
            extra["cursor_run_id"] = args.cursor_run_id
        if args.run_record:
            extra["run_record_path"] = args.run_record
        if not set_status(args.id, args.status, **extra):
            print("task not found", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "seed-sweeps":
        n = seed_runtime_sweeps()
        print(f"seeded {n} tasks")
        return


if __name__ == "__main__":
    main()
