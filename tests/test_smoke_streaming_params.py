"""Smoke experiment streaming metadata defaults (runtime-only benchmarks)."""

from __future__ import annotations

import importlib.util
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
    audio = tmp_path / "first-match-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["first_match_threshold"] == mod._DEFAULT_FIRST_MATCH_THRESHOLD


def test_smoke_first_match_threshold_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRST_MATCH_THRESHOLD", "0.0")
    mod = _load_smoke_run()
    audio = tmp_path / "first-match-env.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["first_match_threshold"] == 0.0


def test_smoke_default_verse_match_threshold_variant_04(monkeypatch, tmp_path):
    monkeypatch.delenv("VERSE_MATCH_THRESHOLD", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "verse-match-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["verse_match_threshold"] == mod._DEFAULT_VERSE_MATCH_THRESHOLD


def test_smoke_verse_match_threshold_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VERSE_MATCH_THRESHOLD", "0.99")
    mod = _load_smoke_run()
    audio = tmp_path / "verse-match-env.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["verse_match_threshold"] == 0.99


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
    monkeypatch.delenv("SMOOTHING_WINDOW", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
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
    monkeypatch.delenv("SMOOTHING_WINDOW", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
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
    monkeypatch.delenv("SMOOTHING_WINDOW", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    monkeypatch.setenv("CHUNK_SECONDS", "0.30")
    mod_low_overlap = _load_smoke_run()
    monkeypatch.setenv("OVERLAP_SECONDS", "0.12")
    mod_high_overlap = _load_smoke_run()
    audio = tmp_path / "s003-a003.wav"
    audio.write_bytes(b"")
    w_low = mod_low_overlap.predict(str(audio))["streaming"]["windows_until_lock"]
    w_high = mod_high_overlap.predict(str(audio))["streaming"]["windows_until_lock"]
    assert w_high >= w_low


def test_smoke_default_smoothing_window_variant_05(monkeypatch, tmp_path):
    monkeypatch.delenv("SMOOTHING_WINDOW", raising=False)
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "smoothing-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["smoothing_window"] == mod._DEFAULT_STREAM_SMOOTHING_WINDOW
    assert out["streaming"]["smoothing_lock_delay_multiplier"] == 1.105


def test_smoke_smoothing_window_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SMOOTHING_WINDOW", "8")
    monkeypatch.setenv("CHUNK_SECONDS", "0.30")
    monkeypatch.setenv("OVERLAP_SECONDS", "0.06")
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "smoothing-env.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["smoothing_window"] == 8
    assert out["streaming"]["smoothing_lock_delay_multiplier"] == 1.28


def test_windows_until_lock_increases_with_smoothing_window(monkeypatch, tmp_path):
    monkeypatch.delenv("SMOOTHING_WINDOW", raising=False)
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    mod_smooth0 = _load_smoke_run()
    monkeypatch.setenv("SMOOTHING_WINDOW", "12")
    mod_smooth12 = _load_smoke_run()
    audio = tmp_path / "s004-a004.wav"
    audio.write_bytes(b"")
    w0 = mod_smooth0.predict(str(audio))["streaming"]["windows_until_lock"]
    w12 = mod_smooth12.predict(str(audio))["streaming"]["windows_until_lock"]
    assert w12 >= w0


def test_smoke_default_correction_hysteresis_variant_06(monkeypatch, tmp_path):
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    monkeypatch.delenv("FIRST_MATCH_THRESHOLD", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "hyst-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["correction_hysteresis"] == mod._DEFAULT_CORRECTION_HYSTERESIS
    eff = out["streaming"]["first_match_effective_threshold"]
    assert eff == mod._DEFAULT_FIRST_MATCH_THRESHOLD + mod._DEFAULT_CORRECTION_HYSTERESIS


def test_smoke_correction_hysteresis_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CORRECTION_HYSTERESIS", "0.0")
    monkeypatch.setenv("FIRST_MATCH_THRESHOLD", "0.05")
    mod = _load_smoke_run()
    audio = tmp_path / "hyst-zero.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["correction_hysteresis"] == 0.0
    assert out["streaming"]["first_match_effective_threshold"] == 0.05


def test_smoke_default_partial_match_margin_variant_07(monkeypatch, tmp_path):
    monkeypatch.delenv("PARTIAL_MATCH_MARGIN", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    monkeypatch.delenv("FIRST_MATCH_THRESHOLD", raising=False)
    mod = _load_smoke_run()
    audio = tmp_path / "partial-margin-default.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    eff = out["streaming"]["first_match_effective_threshold"]
    bar = out["streaming"]["first_match_lock_bar"]
    assert out["streaming"]["partial_match_margin"] == mod._DEFAULT_PARTIAL_MATCH_MARGIN
    assert bar == eff + mod._DEFAULT_PARTIAL_MATCH_MARGIN


def test_smoke_partial_match_margin_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTIAL_MATCH_MARGIN", "0.0")
    monkeypatch.setenv("FIRST_MATCH_THRESHOLD", "0.05")
    monkeypatch.setenv("CORRECTION_HYSTERESIS", "0.01")
    mod = _load_smoke_run()
    audio = tmp_path / "partial-margin-zero.wav"
    audio.write_bytes(b"")
    out = mod.predict(str(audio))
    assert out["streaming"]["partial_match_margin"] == 0.0
    assert out["streaming"]["first_match_lock_bar"] == out["streaming"]["first_match_effective_threshold"]


def test_windows_until_lock_increases_with_correction_hysteresis(monkeypatch, tmp_path):
    monkeypatch.delenv("SMOOTHING_WINDOW", raising=False)
    monkeypatch.delenv("OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("CHUNK_SECONDS", raising=False)
    monkeypatch.delenv("CORRECTION_HYSTERESIS", raising=False)
    mod_h0 = _load_smoke_run()
    monkeypatch.setenv("CORRECTION_HYSTERESIS", "0.08")
    mod_h8 = _load_smoke_run()
    audio = tmp_path / "s005-a005.wav"
    audio.write_bytes(b"")
    w0 = mod_h0.predict(str(audio))["streaming"]["windows_until_lock"]
    wh = mod_h8.predict(str(audio))["streaming"]["windows_until_lock"]
    assert wh >= w0
