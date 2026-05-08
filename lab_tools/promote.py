"""Write a machine-readable promotion record for offline-tarteel-sdk."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Emit promotion manifest for SDK repo")
    p.add_argument("--run-id", required=True)
    p.add_argument("--git-sha", default="")
    p.add_argument("--tier3-report", type=Path, default=None, help="Path to stability JSON")
    p.add_argument("--onnx-sha256", default="")
    p.add_argument("--output", type=Path, required=True, help="Dir or file under sdk releases/")
    args = p.parse_args()

    out = args.output
    if out.is_dir():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = out / f"promotion-{args.run_id}-{ts}.json"

    record = {
        "schema": "offline-tarteel.promotion.v1",
        "run_id": args.run_id,
        "git_sha": args.git_sha,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "tier3_report_path": str(args.tier3_report) if args.tier3_report else None,
        "artifacts": {
            "onnx_sha256": args.onnx_sha256 or None,
        },
        "gates": {
            "corpus_qa": True,
            "tier3_browser_required": True,
            "blind_corpus_non_regression": None,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
