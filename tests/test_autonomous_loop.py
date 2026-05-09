import json

import lab_tools.autonomous_loop as al
import lab_tools.task_queue as tq


def test_modal_allowed_respects_env(monkeypatch):
    monkeypatch.delenv("LAB_AUTONOMY_ALLOW_MODAL", raising=False)
    assert al._modal_allowed(False) is False
    monkeypatch.setenv("LAB_AUTONOMY_ALLOW_MODAL", "1")
    assert al._modal_allowed(False) is True


def test_maybe_launch_modal_skips_when_not_allowed(monkeypatch):
    monkeypatch.delenv("LAB_AUTONOMY_ALLOW_MODAL", raising=False)
    task = tq.Task(
        id="task-modal-test",
        status="queued",
        kind="model_only",
        title="modal smoke",
        payload={"modal_training": True, "job_name": "jn"},
    )
    out = al._maybe_launch_modal(task, allow_modal=False)
    assert len(out) == 1
    assert out[0].returncode == 77


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
                        "samples": 1,
                        "correct": 1,
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
    assert record["tier_completed"] == [1, 2, 3]

    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion["schema"] == "offline-tarteel.promotion.v2"
    assert promotion["run_id"]
    assert promotion["gates"]["corpus_qa"] is True
    assert promotion["gates"]["tier3_browser_required"] is True
