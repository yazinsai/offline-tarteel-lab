"""Local Tier-2 benchmark runner (experiment run.py over local corpus)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab_tools.paths import lab_root, reference_root


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"exp_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _expected_first(sample: dict[str, Any]) -> tuple[int, int] | None:
    ev = sample.get("expected_verses")
    if isinstance(ev, list) and ev:
        first = ev[0]
        if isinstance(first, dict):
            try:
                return int(first.get("surah", 0)), int(first.get("ayah", 0))
            except Exception:
                return None
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-2 local benchmark")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--corpus", default="test_corpus_v3")
    p.add_argument("--experiment", default=None)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    root = (args.root or reference_root()).resolve()
    bench = root / "benchmark" / args.corpus
    manifest_path = bench / "manifest.json"
    if not manifest_path.is_file():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        sys.exit(2)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    total_manifest_samples = len(samples)
    if args.limit > 0:
        samples = samples[: args.limit]

    exp_root = root / "experiments"
    if not exp_root.is_dir():
        print(f"Missing experiments dir: {exp_root}", file=sys.stderr)
        sys.exit(2)

    exps: list[Path] = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "templates":
            continue
        rp = d / "run.py"
        if not rp.is_file():
            continue
        if args.experiment and d.name != args.experiment:
            continue
        exps.append(rp)

    if not exps:
        print("No runnable experiments found", file=sys.stderr)
        sys.exit(2)

    results: list[dict[str, Any]] = []
    for run_py in exps:
        name = run_py.parent.name
        mod = _load_module(run_py)
        if not hasattr(mod, "predict"):
            results.append({"experiment": name, "error": "missing predict()"})
            continue

        correct = 0
        total = 0
        failures = 0
        for s in samples:
            sid = str(s.get("id", "?"))
            ap = bench / str(s.get("file", ""))
            exp = _expected_first(s)
            try:
                pred = mod.predict(str(ap))
                total += 1
                if exp and int(pred.get("surah", 0)) == exp[0] and int(pred.get("ayah", 0)) == exp[1]:
                    correct += 1
            except Exception as e:
                failures += 1
                print(f"[{name}] {sid}: {e}", file=sys.stderr)

        acc = (correct / total) if total else 0.0
        results.append(
            {
                "experiment": name,
                "samples": total,
                "evaluated_samples": total + failures,
                "manifest_samples": total_manifest_samples,
                "correct": correct,
                "accuracy": acc,
                "failures": failures,
            },
        )

    summary = {
        "tier": 2,
        "schema": "offline-tarteel.tier2_report.v1",
        "referenceRoot": str(root),
        "corpus": args.corpus,
        "manifest_samples": total_manifest_samples,
        "sample_limit": args.limit,
        "selected_samples": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    print(json.dumps(summary, indent=2))

    out = args.json
    if out is None:
        art = lab_root() / "artifacts" / "tier2"
        art.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = art / f"tier2-{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
