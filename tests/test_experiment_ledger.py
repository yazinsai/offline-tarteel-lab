import json

import lab_tools.experiment_ledger as ledger
from lab_tools.scorer import score_metrics


def test_score_metrics_emits_canonical_objective_and_slices():
    scored = score_metrics(
        {
            "streaming_alignment_accuracy": 0.8,
            "correction_precision": 0.9,
            "verse_boundary_f1": 0.7,
            "latency_ms": 120,
            "latency_budget_ms": 100,
            "onnx_mb": 250,
            "max_onnx_mb": 200,
            "slices": {
                "multi": {
                    "streaming_alignment_accuracy": 0.5,
                    "correction_precision": 0.6,
                    "verse_boundary_f1": 0.4,
                    "n": 9,
                },
            },
        },
    )

    assert scored["schema"] == "offline-tarteel.objective_score.v1"
    assert scored["components"]["latency_budget_score"] == 100 / 120
    assert scored["objective"] == 0.795833
    assert scored["slices"]["multi"]["score"] == 0.555
    assert scored["slices"]["multi"]["n"] == 9


def test_append_run_record_creates_ledger_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "lab_root", lambda: tmp_path)
    run_record = tmp_path / "artifacts" / "runs" / "run.json"
    run_record.parent.mkdir(parents=True)
    run_record.write_text(
        json.dumps(
            {
                "schema": "offline-tarteel.run_record.v1",
                "run_id": "run-1",
                "task_id": "task-1",
                "git_sha": "abc",
                "dataset_revision": "test_corpus_v3",
                "experiment_kind": "runtime_only",
                "parameter_vector": {
                    "autopilot_key": "runtime.foo",
                    "experiment": "smoke",
                },
                "metrics": {
                    "tier2_accuracy": 0.75,
                    "target_precision": 0.8,
                    "verse_boundary_f1": 0.7,
                },
                "artifact_hashes": {"config": "sha256:abc"},
            },
        ),
        encoding="utf-8",
    )

    entry = ledger.append_run_record(
        run_record,
        status="rejected",
        decision={"accept": False, "reasons": ["min_accuracy_not_met"]},
    )

    rows = ledger.read_entries()
    assert rows == [entry]
    assert entry["experiment_family"] == "runtime.foo"
    assert entry["corpus_revision"] == "test_corpus_v3"
    assert entry["objective"] == 0.7775
    assert entry["artifacts"]["run_record"] == "artifacts/runs/run.json"
    assert entry["failure_modes"] == ["min_accuracy_not_met"]


def test_champion_ignores_invalidated_promotions():
    entries = [
        {
            "schema": ledger.LEDGER_SCHEMA,
            "run_id": "bad",
            "status": "promoted",
            "objective": 0.99,
            "corpus_revision": "test_corpus_v3",
            "parameters": {"full_corpus_gate": True},
        },
        {
            "schema": ledger.LEDGER_SCHEMA,
            "run_id": "good",
            "status": "promoted",
            "objective": 0.5,
            "corpus_revision": "test_corpus_v3",
            "parameters": {"full_corpus_gate": True},
        },
        {
            "schema": ledger.LEDGER_SCHEMA,
            "status": "invalidated",
            "invalidates_run_id": "bad",
            "reason": "label_leakage",
        },
    ]

    assert ledger.champion(entries)["run_id"] == "good"


def test_champion_ignores_v3_promotions_without_full_corpus_gate():
    entries = [
        {
            "schema": ledger.LEDGER_SCHEMA,
            "run_id": "placeholder",
            "status": "promoted",
            "objective": 1.0,
            "corpus_revision": "test_corpus_v3",
            "parameters": {},
        },
        {
            "schema": ledger.LEDGER_SCHEMA,
            "run_id": "full",
            "status": "promoted",
            "objective": 0.4,
            "corpus_revision": "test_corpus_v3",
            "parameters": {"full_corpus_gate": True},
        },
    ]

    assert ledger.champion(entries)["run_id"] == "full"
