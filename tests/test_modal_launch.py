"""Modal launch policy for model_only / joint tasks with modal_training."""

from __future__ import annotations

import lab_tools.autonomous_loop as al
from lab_tools.task_queue import Task


def test_modal_invocation_uses_python_minus_m():
    cmd = al._modal_invocation("job-x")
    assert cmd[:3] == [al.sys.executable, "-m", "modal"]
    assert any(
        isinstance(x, str) and x.startswith("training/train_fastconformer_phoneme_modal.py")
        for x in cmd
    )
    assert "--job-name" in cmd
    idx = cmd.index("--job-name")
    assert cmd[idx + 1] == "job-x"


def test_maybe_launch_modal_skips_when_disabled():
    t = Task(
        id="t1",
        status="running",
        kind="model_only",
        title="x",
        payload={"modal_training": True, "job_name": "jn"},
    )
    out = al._maybe_launch_modal(t, allow_modal=False)
    assert out is not None
    assert out.returncode == 77
    assert "-m" in out.cmd and "modal" in out.cmd


def test_maybe_launch_modal_respects_lab_autonomy_env(monkeypatch):
    monkeypatch.setenv("LAB_AUTONOMY_ALLOW_MODAL", "1")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    t = Task(
        id="t2",
        status="running",
        kind="model_only",
        title="x",
        payload={"modal_training": True, "job_name": "jn"},
    )
    out = al._maybe_launch_modal(t, allow_modal=False)
    assert out is not None
    assert out.returncode == 77


def test_maybe_launch_modal_runs_when_tokens_present(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "sk-test")

    def fake_run(cmd):
        return al.CommandResult(cmd=cmd, returncode=0)

    monkeypatch.setattr(al, "_run", fake_run)
    t = Task(
        id="t3",
        status="running",
        kind="model_only",
        title="x",
        payload={"modal_training": True, "job_name": "jn"},
    )
    out = al._maybe_launch_modal(t, allow_modal=True)
    assert out is not None
    assert out.returncode == 0
    assert out.cmd[1:3] == ["-m", "modal"]
