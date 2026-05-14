from pathlib import Path

import yaml


def _workflow() -> dict:
    path = Path(".github/workflows/cursor-automerge.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cloud_autonomy_workflow() -> dict:
    path = Path(".github/workflows/cursor-cloud-autonomy.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cloud_autonomy_source() -> str:
    return Path("orchestration/src/cloudAutonomy.ts").read_text(encoding="utf-8")


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


def test_state_pr_titles_must_not_lead_with_preflight_slice_scores():
    workflow = _workflow()
    gate_script = next(
        step["run"]
        for step in workflow["jobs"]["merge"]["steps"]
        if step.get("name") == "Evaluate promotion merge policy"
    )

    assert "Misleading state PR title reports preflight slice before champion full-corpus status" in gate_script
    assert "champion hold" in gate_script
    assert "full-corpus" in gate_script
    assert "slice" in gate_script


def test_cloud_autonomy_prompt_requires_state_titles_to_lead_with_full_corpus_champion():
    source = _cloud_autonomy_source()

    assert "For rejected/state-only PRs, do not put the preflight slice score first" in source
    assert "start with the current champion full-corpus result" in source
    assert "put preflight details after the dash" in source


def test_cloud_autonomy_is_manual_load_by_default():
    workflow = _cloud_autonomy_workflow()
    triggers = workflow.get("on") or workflow.get(True)
    source = _cloud_autonomy_source()

    assert "schedule" not in triggers
    assert triggers["workflow_dispatch"]["inputs"]["seed_sweeps"]["default"] == "false"
    assert triggers["workflow_dispatch"]["inputs"]["target_backlog"]["default"] == "0"
    assert 'readBoolDefault("LAB_AUTONOMY_PLAN", false)' in source
    assert "Process only tasks that were manually loaded" in source
