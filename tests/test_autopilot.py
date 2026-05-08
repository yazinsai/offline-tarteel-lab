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
