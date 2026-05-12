from pathlib import Path

import yaml


def _workflow() -> dict:
    path = Path(".github/workflows/cursor-automerge.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_automerge_job_runs_for_failed_pr_ci_so_blocked_prs_can_close():
    workflow = _workflow()
    job_if = workflow["jobs"]["merge"]["if"]

    assert "github.event.workflow_run.event == 'pull_request'" in job_if
    assert "github.event.workflow_run.pull_requests[0].number" in job_if
    assert "github.event.workflow_run.conclusion == 'success'" not in job_if


def test_automerge_gate_still_requires_success_before_merging():
    workflow = _workflow()
    gate_script = next(
        step["run"]
        for step in workflow["jobs"]["merge"]["steps"]
        if step.get("name") == "Evaluate promotion merge policy"
    )
    merge_step = next(
        step
        for step in workflow["jobs"]["merge"]["steps"]
        if step.get("name") == "Merge promotion PR"
    )

    assert 'CI did not succeed: ${WORKFLOW_RUN_CONCLUSION}' in gate_script
    assert "steps.gate.outcome == 'success'" in merge_step["if"]
