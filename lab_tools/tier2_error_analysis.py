"""Explain Tier-2 misses using matcher candidate pools."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def incorrect_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("rows", []):
        pred = row.get("predicted") if isinstance(row, dict) else None
        transcript = pred.get("transcript") if isinstance(pred, dict) else None
        if not row.get("correct", False) and transcript:
            rows.append(row)
    return rows


def _same_verse(candidate: dict[str, Any], expected: dict[str, Any] | None) -> bool:
    if not expected:
        return False
    return (
        int(candidate.get("surah", 0)) == int(expected.get("surah", 0))
        and int(candidate.get("ayah", 0)) == int(expected.get("ayah", 0))
    )


def _rank(candidates: list[dict[str, Any]], expected: dict[str, Any] | None) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if _same_verse(candidate, expected):
            return index
    return None


def _candidate_summary(
    candidates: list[dict[str, Any]], expected: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "expected_rank": _rank(candidates, expected),
        "candidates": [
            {
                "surah": item.get("surah"),
                "ayah": item.get("ayah"),
                "ayah_end": item.get("ayah_end"),
                "score": item.get("score"),
            }
            for item in candidates[:12]
        ],
    }


def analyze_miss(row: dict[str, Any], sample: dict[str, Any], module: Any) -> dict[str, Any]:
    expected = row.get("expected")
    predicted = row.get("predicted") or {}
    transcript = str(predicted.get("transcript", ""))
    base_candidates = module._base._match_phoneme_text(transcript, top_k=18)
    prefix_candidates = (
        module._surah_prefix_candidates(transcript)
        if hasattr(module, "_surah_prefix_candidates")
        else []
    )
    global_span_candidates = (
        module._global_span_candidates(transcript)
        if hasattr(module, "_global_span_candidates")
        else []
    )
    rewind = None
    if hasattr(module, "_same_surah_rewind_candidate") and base_candidates:
        rewind = module._same_surah_rewind_candidate(transcript, base_candidates[0])

    return {
        "index": row.get("index"),
        "id": row.get("id"),
        "file": sample.get("file"),
        "category": sample.get("category"),
        "expected": expected,
        "predicted": predicted,
        "base": _candidate_summary(base_candidates, expected),
        "prefix": _candidate_summary(prefix_candidates, expected),
        "global_span": _candidate_summary(global_span_candidates, expected),
        "same_surah_rewind": rewind,
    }


def analyze_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    module: Any,
) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    by_index = {int(row.get("index", -1)): row for row in samples if isinstance(row, dict)}
    return [
        analyze_miss(row, by_index.get(int(row["index"]), {}), module)
        for row in incorrect_rows(report)
    ]


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"tier2_analysis_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Tier-2 miss candidate pools")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    module = _load_module(args.experiment)
    if hasattr(module, "_base") and hasattr(module._base, "_ensure_loaded"):
        module._base._ensure_loaded()
    analysis = analyze_report(report, manifest, module)
    payload = {"schema": "offline-tarteel.tier2_error_analysis.v1", "misses": analysis}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
