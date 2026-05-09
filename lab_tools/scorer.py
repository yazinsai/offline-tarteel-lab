"""Canonical objective scoring for corpus optimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = {
    "streaming_alignment_accuracy": 0.55,
    "correction_precision": 0.20,
    "verse_boundary_f1": 0.15,
    "latency_budget_score": 0.10,
}


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float | None, default: float = 0.0) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


def latency_budget_score(metrics: dict[str, Any]) -> float:
    """Return 1.0 inside budget and degrade linearly after it."""
    explicit = _float(metrics.get("latency_budget_score"))
    if explicit is not None:
        return _clamp01(explicit)

    latency = _float(metrics.get("latency_ms") or metrics.get("p95_latency_ms"))
    budget = _float(metrics.get("latency_budget_ms") or metrics.get("max_latency_ms"))
    if latency is None or budget is None or budget <= 0:
        return 1.0
    return _clamp01(budget / latency)


def resource_penalty(metrics: dict[str, Any]) -> float:
    penalty = 0.0

    onnx_mb = _float(metrics.get("onnx_mb"))
    max_onnx_mb = _float(metrics.get("max_onnx_mb"))
    if onnx_mb is not None and max_onnx_mb is not None and max_onnx_mb > 0:
        penalty += max(0.0, (onnx_mb - max_onnx_mb) / max_onnx_mb) * 0.05

    cpu = _float(metrics.get("cpu_budget_score"))
    ram = _float(metrics.get("ram_budget_score"))
    for value in (cpu, ram):
        if value is not None:
            penalty += (1.0 - _clamp01(value)) * 0.025

    return penalty


def score_components(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize old and new metric names into the objective components."""
    alignment = _float(
        metrics.get("streaming_alignment_accuracy")
        or metrics.get("sequence_accuracy")
        or metrics.get("seq_accuracy")
        or metrics.get("tier2_accuracy")
        or metrics.get("accuracy")
        or metrics.get("target_recall"),
    )
    correction = _float(
        metrics.get("correction_precision")
        or metrics.get("target_precision")
        or metrics.get("precision")
        or alignment,
    )
    boundary = _float(
        metrics.get("verse_boundary_f1")
        or metrics.get("boundary_f1")
        or metrics.get("f1")
        or alignment,
    )

    return {
        "streaming_alignment_accuracy": _clamp01(alignment),
        "correction_precision": _clamp01(correction),
        "verse_boundary_f1": _clamp01(boundary),
        "latency_budget_score": latency_budget_score(metrics),
    }


def _score_from_components(components: dict[str, float], metrics: dict[str, Any]) -> float:
    raw = sum(DEFAULT_WEIGHTS[k] * components[k] for k in DEFAULT_WEIGHTS)
    return round(_clamp01(raw - resource_penalty(metrics)), 6)


def _slice_score(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    metrics = dict(value.get("metrics") or value)
    components = score_components(metrics)
    out = {
        "score": _score_from_components(components, metrics),
        "components": components,
    }
    if "n" in value:
        out["n"] = value["n"]
    elif "samples" in value:
        out["n"] = value["samples"]
    return out


def score_metrics(metrics: dict[str, Any], *, slices: dict[str, Any] | None = None) -> dict[str, Any]:
    components = score_components(metrics)
    scored_slices = {}
    for name, value in (slices or metrics.get("slices") or {}).items():
        scored = _slice_score(value)
        if scored is not None:
            scored_slices[str(name)] = scored

    return {
        "schema": "offline-tarteel.objective_score.v1",
        "objective": _score_from_components(components, metrics),
        "components": components,
        "slices": scored_slices,
        "penalties": {"resource": round(resource_penalty(metrics), 6)},
    }


def score_tier2_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "tier2_accuracy": result.get("accuracy"),
        "target_recall": result.get("accuracy"),
        "tier2_samples": result.get("samples"),
        "tier2_correct": result.get("correct"),
        "tier2_failures": result.get("failures"),
    }
    if isinstance(result.get("metrics"), dict):
        metrics.update(result["metrics"])
    return score_metrics(metrics, slices=result.get("slices"))


def score_run_record(record: dict[str, Any]) -> dict[str, Any]:
    return score_metrics(record.get("metrics") or {})


def score_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") == "offline-tarteel.run_record.v1":
        return score_run_record(data)
    if data.get("schema") == "offline-tarteel.tier2_report.v1":
        rows = data.get("results") or []
        if not rows:
            return score_metrics({})
        best = max(rows, key=lambda r: float(r.get("accuracy") or 0.0))
        return score_tier2_result(best)
    return score_metrics(data.get("metrics") if isinstance(data.get("metrics"), dict) else data)
