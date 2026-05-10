import json

import lab_tools.task_queue as tq


def test_add_list_set_status(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())
    task = tq.add_task("runtime_only", "unit test task", {"k": "v"})
    st = tq.load_state()
    assert len(st.tasks) == 1
    assert st.tasks[0].status == "queued"

    ok = tq.set_status(task.id, "running", cursor_run_id="run-1", run_record_path="/tmp/r.json")
    assert ok
    st2 = tq.load_state()
    assert st2.tasks[0].status == "running"
    assert st2.tasks[0].cursor_run_id == "run-1"
    assert st2.tasks[0].run_record_path == "/tmp/r.json"

    path = tq.state_path()
    assert path.is_file()
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["schema"] == "offline-tarteel.task_queue.v1"


def test_seed_sweeps_increases_count(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())
    n = tq.seed_runtime_sweeps()
    assert n >= 1
    assert len(tq.load_state().tasks) == n

    assert tq.seed_runtime_sweeps() == 0
    assert len(tq.load_state().tasks) == n


def test_add_task_once_dedupes_by_key(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())

    first = tq.add_task_once("runtime_only", "one", {}, key="same")
    second = tq.add_task_once("runtime_only", "two", {}, key="same")

    assert first is not None
    assert second is None
    state = tq.load_state()
    assert len(state.tasks) == 1
    assert state.tasks[0].payload["autopilot_key"] == "same"


def test_add_task_once_dedupes_active_reference_port_family(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())

    first = tq.add_task_once(
        "joint_model_runtime",
        "Port shipped fastconformer-phoneme v4-tlog baseline from reference repo",
        {"blocked_family": "smoke_runtime_plateau"},
        key="baseline.reference_shipped_fastconformer_v4_tlog.28",
    )
    duplicate = tq.add_task_once(
        "joint_model_runtime",
        "Port shipped fastconformer-phoneme v4-tlog baseline from reference repo",
        {"blocked_family": "smoke_runtime_plateau"},
        key="baseline.reference_shipped_fastconformer_v4_tlog.31",
    )

    assert first is not None
    assert duplicate is None
    state = tq.load_state()
    assert len(state.tasks) == 1
    assert state.tasks[0].payload["autopilot_key"] == "baseline.reference_shipped_fastconformer_v4_tlog.28"


def test_next_queued_can_select_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())

    first = tq.add_task("runtime_only", "first", {})
    second = tq.add_task("runtime_only", "second", {})
    third = tq.add_task("runtime_only", "third", {})

    assert tq.next_queued(shard_index=0, shard_total=3).id == first.id
    assert tq.next_queued(shard_index=1, shard_total=3).id == second.id
    assert tq.next_queued(shard_index=2, shard_total=3).id == third.id

    tq.set_status(first.id, "running")

    assert tq.next_queued(shard_index=0, shard_total=2).id == second.id
    assert tq.next_queued(shard_index=1, shard_total=2).id == third.id
