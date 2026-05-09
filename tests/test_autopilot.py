import lab_tools.autopilot as ap
import lab_tools.task_queue as tq


def test_plan_replenishes_deterministic_backlog(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())

    first = ap.plan(3)
    second = ap.plan(3)

    assert first["active"] == 3
    assert len(first["added"]) == 3
    assert second["active"] == 3
    assert second["added"] == []

    state = tq.load_state()
    assert len(state.tasks) == 3
    assert all(t.payload.get("autopilot_key") for t in state.tasks)
    assert all(t.payload.get("agent_instructions") for t in state.tasks)


def test_plan_retires_repeatedly_blocked_tasks_and_refills(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())
    task = tq.add_task_once(
        "model_only",
        "Evaluate phoneme FastConformer training candidate",
        {"autopilot_key": "model.fastconformer_phoneme_smoke"},
        key="model.fastconformer_phoneme_smoke",
    )
    assert task is not None

    failures = tmp_path / "artifacts" / "autonomy_failures"
    failures.mkdir(parents=True)
    for i in range(2):
        (failures / f"pr-{i}.json").write_text(
            (
                "{"
                f'"changed_files": ["artifacts/runs/{task.id}-20260509T00000{i}Z.json", '
                '"lab_tools/autonomous_loop.py"]'
                "}"
            ),
            encoding="utf-8",
        )

    result = ap.plan(3)

    state = tq.load_state()
    retired = next(t for t in state.tasks if t.id == task.id)
    assert retired.status == "rejected"
    assert retired.judge_reasons == ["autopilot_failure_memory_retired"]
    assert task.id in result["retired"]
    assert result["active"] == 3
    assert len(result["added"]) == 3


def test_adaptive_candidates_keep_runtime_backlog_available():
    keys = [c.key for c in ap.candidates()]

    assert "runtime.adaptive.chunk_seconds.01" in keys
    assert len([k for k in keys if k.startswith("runtime.adaptive.")]) >= 8
