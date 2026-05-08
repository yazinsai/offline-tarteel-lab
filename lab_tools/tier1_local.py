"""Local Tier-1 gate: corpus QA + optional ffmpeg decode smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab_tools.paths import lab_root, reference_root
from lab_tools.validate_corpus import validate_corpus_entry


def _load_registry() -> dict[str, Any]:
    reg = lab_root() / "datasets" / "registry.yaml"
    return json.loads(json.dumps(__import__("yaml").safe_load(reg.read_text(encoding="utf-8"))))


def _decode_ok(path: Path) -> tuple[bool, str | None]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return True, None
    return False, (r.stderr or r.stdout or "decode failed").strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-1 local gate")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--corpus", default="test_corpus_v3")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--decode-audio", action="store_true")
    args = p.parse_args()

    root = (args.root or reference_root()).resolve()
    reg = _load_registry()
    corpora = {c.get("id"): c for c in reg.get("corpora", []) if isinstance(c, dict)}
    entry = corpora.get(args.corpus)
    if not entry:
        print(f"Unknown corpus id: {args.corpus}", file=sys.stderr)
        sys.exit(2)

    errs = validate_corpus_entry(root, entry, compute_checksums=False)
    if errs:
        print("CORPUS VALIDATION FAILED", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads((root / entry["manifest"]).read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if args.limit > 0:
        samples = samples[: args.limit]

    out_samples: list[dict[str, Any]] = []
    for s in samples:
        sid = str(s.get("id", "?"))
        ap = root / entry["audio_root"] / str(s.get("file", ""))
        ok = True
        reason = None
        if args.decode_audio:
            ok, reason = _decode_ok(ap)
        out_samples.append({"id": sid, "pass": ok, "reason": reason})
        print(f"  {'PASS' if ok else 'FAIL'} {sid}")

    summary = {
        "tier": 1,
        "schema": "offline-tarteel.tier1_report.v1",
        "referenceRoot": str(root),
        "corpus": args.corpus,
        "limit": args.limit,
        "repeats": args.repeats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": sum(1 for s in out_samples if s["pass"]),
        "total": len(out_samples),
        "samples": out_samples,
    }
    print(json.dumps(summary, indent=2))

    out = args.json
    if out is None:
        art = lab_root() / "artifacts" / "tier1"
        art.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = art / f"tier1-{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)

    sys.exit(0 if all(s["pass"] for s in out_samples) else 1)


if __name__ == "__main__":
    main()
