"""Smoke experiment streaming metadata defaults (runtime-only benchmarks)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_smoke_run():
    root = Path(__file__).resolve().parent.parent
    path = root / "experiments" / "smoke" / "run.py"
    spec = importlib.util.spec_from_file_location("smoke_run_for_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_default_first_match_threshold_variant_03(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRST_MATCH_THRESHOLD", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "fm-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["first_match_threshold"] == mod._DEFAULT_FIRST_MATCH_THRESHOLD


def test_smoke_first_match_threshold_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRST_MATCH_THRESHOLD", "0")
    mod = _load_smoke_run()
    audio = tmp_path / "fm-env.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["first_match_threshold"] == 0.0


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


def test_windows_until_lock_scales_with_chunk(monkeypatch, tmp_path):
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    mod_short = _load_smoke_run()
    monkeypatch.setenv("CHUNK_SECONDS", "0.20")
    mod_lower = _load_smoke_run()
    audio = tmp_path / "s002-a002.wav"
    audio.write_bytes(b"")
    default_w = mod_short.predict(str(audio))["streaming"]["windows_until_lock"]
    lower_w = mod_lower.predict(str(audio))["streaming"]["windows_until_lock"]
    assert lower_w >= default_w


def test_smoke_default_overlap_seconds_variant_02(monkeypatch, tmp_path):
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "overlap-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["overlap_seconds"] == mod._DEFAULT_STREAM_OVERLAP_SECONDS
    assert out["streaming"]["window_lock_stride_seconds"] == round(
        mod._DEFAULT_STREAM_CHUNK_SECONDS - mod._DEFAULT_STREAM_OVERLAP_SECONDS,
        9,
    )


def test_smoke_overlap_seconds_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OVERLAP_SECONDS", "0.10")
    monkeypatch.setenv("CHUNK_SECONDS", "0.30")
    mod = _load_smoke_run()
    audio = tmp_path / "overlap-env.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["overlap_seconds"] == 0.10
    assert out["streaming"]["window_lock_stride_seconds"] == 0.20


def test_windows_until_lock_increases_with_overlap(monkeypatch, tmp_path):
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    monkeypatch.setenv("CHUNK_SECONDS", "0.30")
    mod_low_overlap = _load_smoke_run()
    monkeypatch.setenv("OVERLAP_SECONDS", "0.12")
    mod_high_overlap = _load_smoke_run()
    audio = tmp_path / "s003-a003.wav"
    audio.write_bytes(b"")
    w_low = mod_low_overlap.predict(str(audio))["streaming"]["windows_until_lock"]
    w_high = mod_high_overlap.predict(str(audio))["streaming"]["windows_until_lock"]
    assert w_high >= w_low
