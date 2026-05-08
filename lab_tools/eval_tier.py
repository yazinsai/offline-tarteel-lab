"""Tiered evaluation driver (standalone lab implementation)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lab_tools.paths import reference_root


def run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("+", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Tiered streaming evaluation driver")
    p.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--experiment", default=None, help="Tier 2: filter by experiment name")
    p.add_argument("--corpus", default="test_corpus_v3", help="Corpus dir name under benchmark/")
    p.add_argument("--limit", type=int, default=12, help="Max samples (0 = all)")
    p.add_argument("--repeats", type=int, default=1, help="Tier 1 repeats per sample")
    p.add_argument("--tier-json", type=Path, default=None, help="Optional output JSON path")
    p.add_argument(
        "--decode-audio",
        action="store_true",
        help="Tier 1: verify ffmpeg can decode each sampled file",
    )
    args = p.parse_args()

    root = reference_root()

    if args.tier == 1:
        cmd = [
            sys.executable,
            "-m",
            "lab_tools.tier1_local",
            "--root",
            str(root),
            "--corpus",
            args.corpus,
            "--limit",
            str(args.limit),
            "--repeats",
            str(args.repeats),
        ]
        if args.tier_json:
            cmd += ["--json", str(args.tier_json)]
        if args.decode_audio:
            cmd += ["--decode-audio"]
        sys.exit(run(cmd))

    if args.tier == 2:
        cmd = [
            sys.executable,
            "-m",
            "lab_tools.tier2_local",
            "--root",
            str(root),
            "--corpus",
            args.corpus,
            "--limit",
            str(args.limit),
        ]
        if args.experiment:
            cmd += ["--experiment", args.experiment]
        if args.tier_json:
            cmd += ["--json", str(args.tier_json)]
        sys.exit(run(cmd))

    cmd = [
        sys.executable,
        "-m",
        "lab_tools.tier3_local",
        "--root",
        str(root),
    ]
    if args.tier_json:
        cmd += ["--json", str(args.tier_json)]
    sys.exit(run(cmd))


if __name__ == "__main__":
    main()
