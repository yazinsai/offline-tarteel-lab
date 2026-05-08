"""Codified promotion judge (streaming metrics + blind non-regression + budgets)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JudgeInput:
    """Minimal contract; extend as metrics evolve."""

    target_recall: float | None = None
    target_precision: float | None = None
    target_seq_exact: float | None = None
    baseline_recall: float | None = None
    baseline_precision: float | None = None
    blind_recall: float | None = None
    blind_baseline_recall: float | None = None
    max_onnx_mb: float = 200.0
    onnx_mb: float | None = None
    tier3_completed: bool = False


def judge(inp: JudgeInput) -> dict[str, Any]:
    reasons: list[str] = []
    ok = True

    if not inp.tier3_completed:
        ok = False
        reasons.append("tier3_required")

    if inp.onnx_mb is not None and inp.onnx_mb > inp.max_onnx_mb:
        ok = False
        reasons.append("onnx_size_budget")

    if inp.target_recall is not None and inp.baseline_recall is not None:
        if inp.target_recall + 1e-6 < inp.baseline_recall:
            ok = False
            reasons.append("target_recall_regression")

    if inp.blind_recall is not None and inp.blind_baseline_recall is not None:
        if inp.blind_recall + 1e-6 < inp.blind_baseline_recall:
            ok = False
            reasons.append("blind_corpus_regression")

    if inp.target_precision is not None and inp.baseline_precision is not None:
        if inp.target_precision + 1e-6 < inp.baseline_precision:
            ok = False
            reasons.append("precision_regression")

    return {"accept": ok, "reasons": reasons}


def main() -> None:
    p = argparse.ArgumentParser(description="Judge promotion from metric JSON")
    p.add_argument("--input", type=Path, required=False, default=None)
    args = p.parse_args()

    if args.input is None:
        raw = sys.stdin.read()
    else:
        raw = args.input.read_text(encoding="utf-8")
    data = json.loads(raw)
    inp = JudgeInput(
        target_recall=data.get("target_recall"),
        target_precision=data.get("target_precision"),
        target_seq_exact=data.get("target_seq_exact"),
        baseline_recall=data.get("baseline_recall"),
        baseline_precision=data.get("baseline_precision"),
        blind_recall=data.get("blind_recall"),
        blind_baseline_recall=data.get("blind_baseline_recall"),
        max_onnx_mb=float(data.get("max_onnx_mb", 200)),
        onnx_mb=data.get("onnx_mb"),
        tier3_completed=bool(data.get("tier3_completed")),
    )
    out = judge(inp)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
