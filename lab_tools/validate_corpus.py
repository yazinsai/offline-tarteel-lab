"""Fail-fast corpus QA: schema, file existence, minimal consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from lab_tools.paths import lab_root, reference_root


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_registry() -> dict[str, Any]:
    reg_path = lab_root() / "datasets" / "registry.yaml"
    if not reg_path.is_file():
        raise FileNotFoundError(f"Missing registry: {reg_path}")
    with reg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_corpus_entry(
    root: Path,
    entry: dict[str, Any],
    *,
    compute_checksums: bool = False,
) -> list[str]:
    errors: list[str] = []
    cid = entry.get("id", "?")
    manifest_rel = entry.get("manifest")
    audio_root_rel = entry.get("audio_root")
    if not manifest_rel or not audio_root_rel:
        errors.append(f"{cid}: missing manifest or audio_root")
        return errors

    manifest_path = root / manifest_rel
    audio_root = root / audio_root_rel
    if not manifest_path.is_file():
        errors.append(f"{cid}: manifest not found: {manifest_path}")
        return errors
    if not audio_root.is_dir():
        errors.append(f"{cid}: audio_root not a directory: {audio_root}")
        return errors

    try:
        with manifest_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"{cid}: invalid JSON manifest: {e}")
        return errors

    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append(f"{cid}: manifest must contain non-empty samples[]")
        return errors

    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            errors.append(f"{cid}: samples[{i}] not an object")
            continue
        sid = s.get("id", f"#{i}")
        fname = s.get("file")
        if not fname:
            errors.append(f"{cid}: sample {sid} missing file")
            continue
        ap = audio_root / fname
        if not ap.is_file():
            errors.append(f"{cid}: sample {sid} audio missing: {ap}")
            continue
        if compute_checksums and "_sha256" in s:
            got = _sha256_file(ap)
            if got != s["_sha256"]:
                errors.append(f"{cid}: sample {sid} sha256 mismatch (manifest stale?)")

        ev = s.get("expected_verses")
        if ev is not None and not isinstance(ev, list):
            errors.append(f"{cid}: sample {sid} expected_verses must be list if present")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark corpora against registry")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root containing benchmark/ (default: offline-tarteel-lab root)",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="Verify per-sample _sha256 in manifest if present",
    )
    args = parser.parse_args()
    root = (args.root or reference_root()).resolve()
    reg = _load_registry()
    corpora = reg.get("corpora") or []
    all_err: list[str] = []
    for entry in corpora:
        all_err.extend(
            validate_corpus_entry(root, entry, compute_checksums=args.checksums),
        )
    if all_err:
        print("CORPUS VALIDATION FAILED", file=sys.stderr)
        for e in all_err:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {len(corpora)} corpora validated under {root}")


if __name__ == "__main__":
    main()
