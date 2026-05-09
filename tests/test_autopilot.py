import lab_tools.autopilot as ap
import lab_tools.task_queue as tq


def test_plan_replenishes_deterministic_backlog(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(ap, "read_entries", lambda: [])
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
    monkeypatch.setattr(ap, "read_entries", lambda: [])
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


def test_adaptive_candidates_keep_runtime_backlog_available(monkeypatch):
    monkeypatch.setattr(ap, "read_entries", lambda: [])
    keys = [c.key for c in ap.candidates()]

    assert "runtime.adaptive.chunk_seconds.01" in keys
    assert len([k for k in keys if k.startswith("runtime.adaptive.")]) >= 8


def test_plan_uses_ledger_champion_and_worst_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(
        ap,
        "read_entries",
        lambda: [
            {
                "schema": "offline-tarteel.experiment_ledger.v1",
                "run_id": "run-champ",
                "status": "promoted",
                "experiment_family": "runtime.threshold_sweep.first_match",
                "objective": 0.91,
                "parameters": {"experiment": "smoke", "param": "FIRST_MATCH_THRESHOLD"},
                "slices": {
                    "short": {"score": 0.95, "n": 17},
                    "multi": {"score": 0.62, "n": 9},
                },
            },
        ],
    )
    tq.save_state(tq.QueueState())

    result = ap.plan(2)

    assert result["champion"]["run_id"] == "run-champ"
    assert result["worst_slice"] == {"name": "multi", "score": 0.62}
    state = tq.load_state()
    keys = [t.payload["autopilot_key"] for t in state.tasks]
    assert keys == [
        "runtime.repair_slice.multi.run-champ",
        "runtime.exploit_champion.runtime.threshold_sweep.first_match.run-champ",
    ]
    assert state.tasks[0].payload["target_slice"] == "multi"
    assert state.tasks[1].payload["champion_run_id"] == "run-champ"


def test_plan_skips_repeatedly_failed_ledger_families(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(
        ap,
        "read_entries",
        lambda: [
            {
                "schema": "offline-tarteel.experiment_ledger.v1",
                "run_id": f"run-fail-{i}",
                "status": "rejected",
                "experiment_family": "runtime.threshold_sweep.first_match",
                "objective": 0.1,
            }
            for i in range(2)
        ],
    )
    tq.save_state(tq.QueueState())

    result = ap.plan(3)

    assert "runtime.threshold_sweep.first_match" in result["blocked_families"]
    keys = [t.payload["autopilot_key"] for t in tq.load_state().tasks]
    assert "runtime.threshold_sweep.first_match" not in keys
    assert "runtime.chunk_window_sweep" in keys
