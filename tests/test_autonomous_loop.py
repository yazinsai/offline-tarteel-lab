import json

import lab_tools.autonomous_loop as al
import lab_tools.task_queue as tq


def test_judge_from_metrics_rejects_missing_accuracy():
    out = al._judge_from_metrics({"tier3_completed": True})
    assert out["accept"] is False
    assert "missing_tier2_accuracy" in out["reasons"]


def test_judge_from_metrics_rejects_below_min_accuracy():
    out = al._judge_from_metrics(
        {
            "tier2_accuracy": 0.4,
            "target_recall": 0.4,
            "min_accuracy": 0.5,
            "tier3_completed": True,
        },
    )
    assert out["accept"] is False
    assert "min_accuracy_not_met" in out["reasons"]


def test_judge_from_metrics_rejects_incomplete_promotion_corpus():
    out = al._judge_from_metrics(
        {
            "tier2_accuracy": 1.0,
            "target_recall": 1.0,
            "tier2_evaluated_samples": 1,
            "tier2_manifest_samples": 11,
            "tier3_completed": True,
            "requires_full_corpus_gate": True,
        },
    )
    assert out["accept"] is False
    assert "full_corpus_coverage_required" in out["reasons"]


def test_judge_from_metrics_rejects_non_improving_challenger():
    out = al._judge_from_metrics(
        {
            "tier2_accuracy": 1.0,
            "target_recall": 1.0,
            "tier2_evaluated_samples": 12,
            "tier2_manifest_samples": 12,
            "tier3_completed": True,
            "requires_full_corpus_gate": True,
            "requires_champion_improvement": True,
            "candidate_objective": 1.0,
            "champion_objective": 1.0,
        },
    )
    assert out["accept"] is False
    assert "champion_objective_not_improved" in out["reasons"]


def test_run_once_promotes_accepted_task(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(tq, "lab_root", lambda: tmp_path)
    tq.save_state(tq.QueueState())
    task = tq.add_task(
        "runtime_only",
        "smoke autonomous path",
        {"experiment": "smoke", "min_accuracy": 0.5},
    )

    tier1_path = tmp_path / "artifacts" / "tier1" / "tier1-test.json"
    tier2_path = tmp_path / "artifacts" / "tier2" / "tier2-test.json"
    tier3_path = tmp_path / "artifacts" / "tier3" / "tier3-test.json"
    tier1_path.parent.mkdir(parents=True)
    tier2_path.parent.mkdir(parents=True)
    tier3_path.parent.mkdir(parents=True)
    tier1_path.write_text(json.dumps({"passed": 1, "total": 1}), encoding="utf-8")
    tier2_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "experiment": "smoke",
                        "samples": 12,
                        "evaluated_samples": 12,
                        "manifest_samples": 12,
                        "correct": 12,
                        "accuracy": 1.0,
                        "failures": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    tier3_path.write_text(json.dumps({"completed": True}), encoding="utf-8")

    def fake_run(cmd):
        return al.CommandResult(cmd=cmd, returncode=0)

    def fake_latest_report(tier):
        return {1: tier1_path, 2: tier2_path, 3: tier3_path}[tier]

    monkeypatch.setattr(al, "_run", fake_run)
    monkeypatch.setattr(al, "_latest_report", fake_latest_report)
    promotion_path = tmp_path / "artifacts" / "promotions" / "promotion-test.json"
    promotion_path.parent.mkdir(parents=True)
    promotion_path.write_text(
        json.dumps(
            {
                "schema": "offline-tarteel.promotion.v2",
                "run_id": task.id,
                "gates": {
                    "corpus_qa": True,
                    "tier3_browser_required": True,
                },
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(al, "_promote_run", lambda *_args: (0, promotion_path))
    monkeypatch.setattr(al, "_git_sha", lambda: "abc123")

    assert al.run_once(limit=1) == 0
    state = tq.load_state()
    updated = next(t for t in state.tasks if t.id == task.id)
    assert updated.status == "promoted"
    assert updated.run_record_path

    record = json.loads((tmp_path / updated.run_record_path).read_text(encoding="utf-8"))
    assert record["metrics"]["tier2_accuracy"] == 1.0
    assert record["metrics"]["tier2_evaluated_samples"] == 12
    assert record["metrics"]["tier2_manifest_samples"] == 12
    assert record["metrics"]["requires_full_corpus_gate"] is True
    assert record["metrics"]["champion_objective"] is None
    assert record["parameter_vector"]["full_corpus_gate"] is True
    assert record["tier_completed"] == [1, 2, 3]
    assert any("--limit 0" in command for command in record["commands"])
    ledger_path = tmp_path / "artifacts" / "experiment_ledger.jsonl"
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert ledger_rows[-1]["run_id"] == record["run_id"]
    assert ledger_rows[-1]["artifacts"]["run_record"] == updated.run_record_path

    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion["schema"] == "offline-tarteel.promotion.v2"
    assert promotion["run_id"]
    assert promotion["gates"]["corpus_qa"] is True
    assert promotion["gates"]["tier3_browser_required"] is True
