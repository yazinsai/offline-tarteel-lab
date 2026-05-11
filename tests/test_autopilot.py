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
    # Two threshold-sweep failures also trip the smoke_runtime change_class guard, so other static
    # runtime smoke/chunk seeds are skipped; backlog should still seed model/joint probes.
    assert "model.fastconformer_phoneme_smoke" in keys
    assert "joint.model_runtime_export_contract" in keys


def _smoke_runtime_failure(i: int, param: str = "chunk_seconds") -> dict:
    return {
        "schema": "offline-tarteel.experiment_ledger.v1",
        "run_id": f"run-smoke-fail-{i}",
        "task_id": f"task-smoke-fail-{i}",
        "status": "rejected",
        "experiment_kind": "runtime_only",
        "experiment_family": f"runtime.adaptive.{param}.{i:02d}",
        "corpus_revision": "test_corpus_v3",
        "objective": 0.103516,
        "components": {
            "streaming_alignment_accuracy": 0.00390625,
            "correction_precision": 0.00390625,
            "verse_boundary_f1": 0.00390625,
            "latency_budget_score": 1.0,
        },
        "parameters": {
            "experiment": "smoke",
            "param": param,
            "full_corpus_gate": True,
            "autopilot_key": f"runtime.adaptive.{param}.{i:02d}",
        },
        "failure_modes": ["min_accuracy_not_met"],
    }


def test_smoke_runtime_plateau_suppresses_runtime_knobs(monkeypatch):
    entries = [_smoke_runtime_failure(i) for i in range(1, 5)]
    monkeypatch.setattr(ap, "read_entries", lambda: entries)

    planned = ap.candidates()
    keys = [c.key for c in planned]

    assert ap.smoke_runtime_plateau(entries) is True
    assert all(not key.startswith("runtime.adaptive.") for key in keys)
    assert "runtime.explore_diverse.v3" not in keys
    assert any(key.startswith("baseline.reference_shipped_fastconformer_v4_tlog.") for key in keys)
    assert any(key.startswith("escalate.non_smoke.") for key in keys)
    assert {c.kind for c in planned if c.key.startswith(("baseline.", "escalate.non_smoke."))} == {
        "model_only",
        "joint_model_runtime",
    }


def test_plateau_escape_keys_advance_after_non_smoke_rejections():
    entries = [_smoke_runtime_failure(i) for i in range(1, 5)]
    entries.append(
        {
            "status": "rejected",
            "experiment_kind": "joint_model_runtime",
            "experiment_family": "baseline.reference_shipped_fastconformer_v4_tlog.04",
            "parameters": {"blocked_family": "smoke_runtime_plateau"},
            "failure_modes": ["min_accuracy_not_met"],
        }
    )

    planned = ap.candidates(entries)

    assert planned[0].key == "baseline.reference_shipped_fastconformer_v4_tlog.05"
    assert planned[1].key == "escalate.non_smoke.model_only.05.01"


def test_plan_retires_queued_smoke_runtime_after_plateau(tmp_path, monkeypatch):
    entries = [_smoke_runtime_failure(i) for i in range(1, 5)]
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(ap, "read_entries", lambda: entries)
    tq.save_state(tq.QueueState())
    runtime = tq.add_task_once(
        "runtime_only",
        "Tune streaming smoothing window variant 29",
        {
            "experiment": "smoke",
            "param": "smoothing_window",
            "autopilot_key": "runtime.adaptive.smoothing_window.29",
        },
        key="runtime.adaptive.smoothing_window.29",
    )
    assert runtime is not None

    result = ap.plan(3)

    state = tq.load_state()
    retired = next(t for t in state.tasks if t.id == runtime.id)
    assert retired.status == "rejected"
    assert retired.judge_reasons == ["smoke_runtime_plateau_retired"]
    assert runtime.id in result["retired"]
    assert result["smoke_runtime_plateau"] is True

    active = [t for t in state.tasks if t.status == "queued"]
    assert len(active) == 3
    assert all(t.kind in {"model_only", "joint_model_runtime"} for t in active)
    assert all((t.payload or {}).get("experiment") != "smoke" for t in active)
    assert active[0].payload["reference_baseline"] == "fastconformer-phoneme v4-tlog browser/RN streaming"
    assert active[0].payload["reference_repo_url"] == "https://github.com/yazinsai/offline-tarteel.git"
    assert active[0].payload["target_correct_range"] == [223, 225]
    assert any((t.payload or {}).get("blocked_family") == "smoke_runtime_plateau" for t in active)


def test_plan_retires_stale_generic_plateau_escalations(tmp_path, monkeypatch):
    entries = [_smoke_runtime_failure(i) for i in range(1, 5)]
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(ap, "read_entries", lambda: entries)
    tq.save_state(tq.QueueState())
    stale = tq.add_task_once(
        "model_only",
        "Evaluate non-smoke ASR candidate on full corpus variant 01",
        {
            "blocked_family": "smoke_runtime_plateau",
            "autopilot_key": "escalate.non_smoke.model_only.04.01",
        },
        key="escalate.non_smoke.model_only.04.01",
    )
    assert stale is not None

    result = ap.plan(1)

    state = tq.load_state()
    retired = next(t for t in state.tasks if t.id == stale.id)
    active = [t for t in state.tasks if t.status == "queued"]
    assert retired.status == "rejected"
    assert stale.id in result["retired"]
    assert len(active) == 1
    assert active[0].payload["reference_baseline"] == "fastconformer-phoneme v4-tlog browser/RN streaming"


def test_failure_memory_does_not_retire_plateau_escape_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(ap, "read_entries", lambda: [])
    tq.save_state(tq.QueueState())
    task = tq.add_task_once(
        "joint_model_runtime",
        "Port shipped fastconformer-phoneme v4-tlog baseline from reference repo",
        {
            "blocked_family": "smoke_runtime_plateau",
            "reference_baseline": "fastconformer-phoneme v4-tlog browser/RN streaming",
            "autopilot_key": "baseline.reference_shipped_fastconformer_v4_tlog.17",
        },
        key="baseline.reference_shipped_fastconformer_v4_tlog.17",
    )
    assert task is not None

    failures = tmp_path / "artifacts" / "autonomy_failures"
    failures.mkdir(parents=True)
    for i in range(2):
        (failures / f"pr-{i}.json").write_text(
            '{"changed_files": ["lab_tools/autonomous_loop.py"]}',
            encoding="utf-8",
        )

    retired = ap._retire_repeatedly_blocked_tasks()

    state = tq.load_state()
    kept = next(t for t in state.tasks if t.id == task.id)
    assert retired == []
    assert kept.status == "queued"
