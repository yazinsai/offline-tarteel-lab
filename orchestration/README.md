# Lab orchestration (Cursor SDK)

## Dispatch a cloud worker

```bash
cd orchestration
npm install
export CURSOR_API_KEY=...
export OFFLINE_TARTEEL_REPO_URL=https://github.com/your-org/offline-tarteel-lab.git
export OFFLINE_TARTEEL_REF=main
npx tsx src/dispatch.ts "Your experiment prompt — edit tracker thresholds, run Tier 1, etc."
```

Optional:

- `CURSOR_AUTO_PR=1` — open PR from cloud agent
- `CURSOR_SKIP_REVIEWER=0` — allow reviewer request

## Autonomous cloud cycle

This dispatches a bounded Cursor Cloud worker that runs the lab controller and
opens a PR when it produces useful tracked changes or promotion metadata.

```bash
cd orchestration
npm install
export CURSOR_API_KEY=...
export OFFLINE_TARTEEL_REPO_URL=https://github.com/your-org/offline-tarteel-lab.git
export OFFLINE_TARTEEL_REF=main
LAB_AUTONOMY_MAX_CYCLES=1 npm run cloud:autonomy
```

Useful knobs:

- `LAB_AUTONOMY_MAX_CYCLES` — queued tasks to process; default `1`.
- `LAB_AUTONOMY_TARGET_BACKLOG` — autonomous queued/running task target; default `3`.
- `LAB_AUTONOMY_LIMIT` — benchmark sample limit; default `12`.
- `LAB_AUTONOMY_CORPUS` — corpus id; default `test_corpus_v3`.
- `LAB_AUTONOMY_PLAN=0` — disable autopilot backlog replenishment.
- `LAB_AUTONOMY_ALLOW_MODAL=1` — allow Modal training launches from model tasks.
- `CURSOR_MODEL_ID` — Cursor model id; default `composer-2`.

For unattended operation, set repository secret `CURSOR_API_KEY` and use the
`Cursor Cloud Autonomy` GitHub Actions workflow.

Promotion PRs can be merged by the `Cursor Promotion Auto-Merge` workflow after
`CI` passes. The merge gate requires a promotion manifest, a promotion PR title,
and changed files limited to experiment, benchmark, test, and promotion metadata
paths. State-only PRs titled `Autopilot offline-tarteel state:` can also merge
when they only update queue/run JSON, which prevents rejected ideas from being
retried forever.

## Judge (metrics JSON → accept/reject)

```bash
echo '{"target_recall":0.9,"baseline_recall":0.85,"blind_recall":0.88,"blind_baseline_recall":0.87,"onnx_mb":131,"tier3_completed":true}' \
  | npx tsx src/judgeCli.ts
```

Runs `lab_tools.judge_policy` in the lab venv.
