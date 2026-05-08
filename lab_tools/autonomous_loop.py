"""
Minimal autonomous loop tick: claim queued task → mark running → print dispatch hint.

Wire `orchestration/src/dispatch.ts` or Modal from your runner; this module keeps durable state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab_tools.paths import lab_root
from lab_tools.task_queue import load_state, next_queued, set_status


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


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous loop (single tick)")
    p.add_argument("command", choices=["tick", "status"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.command == "status":
        state = load_state()
        by = {}
        for t in state.tasks:
            by[t.status] = by.get(t.status, 0) + 1
        print(json.dumps(by, indent=2))
        return
    sys.exit(tick(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
