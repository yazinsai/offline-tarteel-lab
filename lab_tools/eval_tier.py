"""Invoke tiered evaluation against the reference tree (see plan: Tier1/2/3)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from lab_tools.paths import lab_root, reference_root


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> int:
    print("+", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    return r.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Tiered streaming evaluation driver")
    p.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    p.add_argument(
        "--experiment",
        default=None,
        help="For tier 2: benchmark.runner --experiment name",
    )
    p.add_argument(
        "--stability-args",
        default="--repeats=1 --limit=5",
        help='For tier 3: extra args to stability-report.ts (quoted)',
    )
    args = p.parse_args()
    root = reference_root()
    lab = lab_root()

    if args.tier == 1:
        # Fast gate: corpus QA + optional env hint for local ONNX smoke
        code = run([sys.executable, "-m", "lab_tools.validate_corpus", "--root", str(root)])
        if code != 0:
            sys.exit(code)
        print(
            "Tier 1 OK: corpus validated. "
            "Add Node/ORT smoke in CI by calling offline-tarteel-sdk tier1 script when ready.",
        )
        return

    if args.tier == 2:
        py = root / ".venv" / "bin" / "python"
        if not py.is_file():
            py = Path(sys.executable)
        cmd = [str(py), "-m", "benchmark.runner"]
        if args.experiment:
            cmd += ["--experiment", args.experiment]
        code = run(cmd, cwd=root)
        sys.exit(code)

    if args.tier == 3:
        frontend = root / "web" / "frontend"
        if not frontend.is_dir():
            print(f"Missing {frontend}", file=sys.stderr)
            sys.exit(2)
        npm = os.environ.get("NPM", "npx")
        extra = args.stability_args.split()
        cmd = [npm, "tsx", "test/stability-report.ts", *extra]
        env = os.environ.copy()
        code = run(cmd, cwd=frontend, env=env)
        sys.exit(code)

    sys.exit(2)


if __name__ == "__main__":
    main()
