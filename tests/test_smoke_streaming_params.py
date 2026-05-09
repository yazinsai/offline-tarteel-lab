"""Smoke experiment streaming metadata defaults (runtime-only benchmarks)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _load_smoke_run():
    root = Path(__file__).resolve().parent.parent
    path = root / "experiments" / "smoke" / "run.py"
    spec = importlib.util.spec_from_file_location("smoke_run_for_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_default_chunk_seconds_variant_01(monkeypatch, tmp_path):
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "s001-a001.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["chunk_seconds"] == mod._DEFAULT_STREAM_CHUNK_SECONDS


def test_smoke_chunk_seconds_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUNK_SECONDS", "0.22")
    mod = _load_smoke_run()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["chunk_seconds"] == 0.22


def test_smoke_default_overlap_seconds_variant_02(monkeypatch, tmp_path):
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "s003-a003.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["overlap_seconds"] == mod._DEFAULT_OVERLAP_SECONDS


def test_smoke_overlap_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OVERLAP_SECONDS", "0.06")
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "clip_overlap.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["overlap_seconds"] == 0.06
    assert out["streaming"]["stream_stride_seconds"] == pytest.approx(0.285 - 0.06)


def test_windows_until_lock_scales_with_chunk(monkeypatch, tmp_path):
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    mod_short = _load_smoke_run()
    monkeypatch.setenv("CHUNK_SECONDS", "0.20")
    mod_lower = _load_smoke_run()
    audio = tmp_path / "s002-a002.wav"
    audio.write_bytes(b"")
    default_w = mod_short.predict(str(audio))["streaming"]["windows_until_lock"]
    lower_w = mod_lower.predict(str(audio))["streaming"]["windows_until_lock"]
    assert lower_w >= default_w
