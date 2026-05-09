"""Write a machine-readable promotion record for a release target."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MIN_PROMOTION_CORPUS_SAMPLES = 12


def main() -> None:
    p = argparse.ArgumentParser(description="Emit promotion manifest for release pipeline")
    p.add_argument("--run-id", required=True)
    p.add_argument("--git-sha", default="")
    p.add_argument("--core-version", default="", help="npm version of @offline-tarteel/core")
    p.add_argument("--sdk-version", default="", help="npm version of @offline-tarteel/sdk")
    p.add_argument("--tier3-report", type=Path, default=None, help="Path to stability JSON")
    p.add_argument("--tier1-report", type=Path, default=None, help="Path to tier1 JSON from lab-eval-tier")
    p.add_argument("--run-record", type=Path, default=None, help="Path to the accepted run record")
    p.add_argument("--onnx-sha256", default="")
    p.add_argument(
        "--artifacts-json",
        type=Path,
        default=None,
        help="JSON object merged into record['artifacts'] (file contents must be a JSON object)",
    )
    p.add_argument(
        "--reference-min-sha",
        default="",
        help="Minimum tested reference offline-tarteel git SHA for this promotion",
    )
    p.add_argument("--output", type=Path, required=True, help="Dir or file under sdk releases/")
    args = p.parse_args()

    out = args.output
    if out.is_dir() or out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = out / f"promotion-{args.run_id}-{ts}.json"

    extra_artifacts: dict = {}
    if args.artifacts_json and args.artifacts_json.is_file():
        blob = json.loads(args.artifacts_json.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            print("--artifacts-json must contain a JSON object", file=sys.stderr)
            sys.exit(2)
        extra_artifacts = blob

    run_record: dict = {}
    metrics: dict = {}
    if args.run_record and args.run_record.is_file():
        blob = json.loads(args.run_record.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            print("--run-record must contain a JSON object", file=sys.stderr)
            sys.exit(2)
        run_record = blob
        metrics = run_record.get("metrics") if isinstance(run_record.get("metrics"), dict) else {}

    full_corpus_gate = bool(metrics.get("requires_full_corpus_gate"))
    evaluated = metrics.get("tier2_evaluated_samples")
    manifest = metrics.get("tier2_manifest_samples")
    try:
        full_corpus_gate = (
            full_corpus_gate
            and int(manifest) >= MIN_PROMOTION_CORPUS_SAMPLES
            and int(evaluated) >= int(manifest)
        )
    except (TypeError, ValueError):
        full_corpus_gate = False

    champion_objective = metrics.get("champion_objective")
    candidate_objective = metrics.get("candidate_objective")
    try:
        champion_gate = champion_objective is None or float(candidate_objective) > float(champion_objective)
    except (TypeError, ValueError):
        champion_gate = False

    record = {
        "schema": "offline-tarteel.promotion.v2",
        "run_id": args.run_id,
        "git_sha": args.git_sha,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {
            "core_version": args.core_version or None,
            "sdk_version": args.sdk_version or None,
        },
        "reports": {
            "tier1_report_path": str(args.tier1_report) if args.tier1_report else None,
            "tier3_report_path": str(args.tier3_report) if args.tier3_report else None,
            "run_record_path": str(args.run_record) if args.run_record else None,
        },
        "artifacts": {
            "onnx_sha256": args.onnx_sha256 or None,
            **extra_artifacts,
        },
        "compatibility": {
            "reference_repo_min_sha": args.reference_min_sha or None,
            "node_major_min": 20,
            "requires_ffmpeg_tier1": True,
        },
        "gates": {
            "corpus_qa": full_corpus_gate and champion_gate,
            "full_corpus_v3": full_corpus_gate,
            "champion_objective_improved_or_bootstrap": champion_gate,
            "tier1_onnx_recommended": True,
            "tier3_browser_required": True,
            "blind_corpus_non_regression": None,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
