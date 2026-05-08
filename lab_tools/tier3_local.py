"""Local Tier-3 gate based on Tier-2 outputs (standalone fallback)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from lab_tools.paths import lab_root, reference_root


def _latest_tier2() -> Path | None:
    art = lab_root() / "artifacts" / "tier2"
    if not art.is_dir():
        return None
    files = sorted(art.glob("tier2-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-3 local gate")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    root = (args.root or reference_root()).resolve()
    tier2 = _latest_tier2()
    if tier2 is None:
        print("No Tier-2 report found under artifacts/tier2", file=sys.stderr)
        sys.exit(2)

    t2 = json.loads(tier2.read_text(encoding="utf-8"))
    rows = t2.get("results", []) if isinstance(t2, dict) else []
    completed = bool(rows)
    summary = {
        "tier": 3,
        "schema": "offline-tarteel.tier3_report.v1",
        "referenceRoot": str(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_tier2_report": str(tier2),
        "completed": completed,
        "checks": {
            "tier2_present": tier2 is not None,
            "at_least_one_result": completed,
        },
    }
    print(json.dumps(summary, indent=2))

    out = args.json
    if out is None:
        art = lab_root() / "artifacts" / "tier3"
        art.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = art / f"tier3-{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)

    sys.exit(0 if completed else 1)


if __name__ == "__main__":
    main()
