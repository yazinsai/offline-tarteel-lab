"""Guardrails against benchmark-label leakage in experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_experiments_do_not_parse_labels_from_paths_or_manifests():
    forbidden_tokens = {
        ".stem",
        ".name",
        "with_name(",
        ".first_verse",
        "first_verse.json",
        "manifest.json",
        "everyayah",
        "alafasy",
        "husary",
        "tlog_",
    }
    offenders: list[str] = []
    for run_py in (PROJECT_ROOT / "experiments").glob("*/run.py"):
        source = run_py.read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            if token in source:
                offenders.append(f"{run_py.relative_to(PROJECT_ROOT)} contains {token!r}")

    assert offenders == []
