# Autonomous Lab Loop

This repo is set up around a queue-driven loop:

1. Seed or add experiment tasks.
2. Claim one task.
3. Run local tier gates.
4. Write a machine-readable run record.
5. Judge the result against baseline, blind-corpus, and artifact-budget policy.
6. Emit a promotion manifest for accepted runs.

The loop is intentionally file-backed under `artifacts/` so it can run from cron,
GitHub Actions, a long-lived VM, or a local machine without adding a database.

## Local Controller

Initialize the queue:

```bash
python -m lab_tools.task_queue init
```

Seed starter runtime sweeps:

```bash
python -m lab_tools.task_queue seed-sweeps
```

Add a targeted experiment:

```bash
python -m lab_tools.task_queue add \
  --kind runtime_only \
  --title "Tune first-verse streaming threshold" \
  --payload '{"experiment":"smoke","baseline_accuracy":0.75,"min_accuracy":0.8}'
```

Run one autonomous local cycle:

```bash
python -m lab_tools.autonomous_loop run-once --limit 12
```

Run until the queue is empty:

```bash
python -m lab_tools.autonomous_loop run --limit 12
```

Run as a bounded worker, for example from cron or CI:

```bash
python -m lab_tools.autonomous_loop run --max-cycles 5 --sleep-seconds 30
```

Dry-run the next cycle:

```bash
python -m lab_tools.autonomous_loop run-once --dry-run
```

Accepted runs become `promoted` tasks and receive:

- `artifacts/runs/*.json` run records
- `artifacts/promotions/*.json` promotion manifests

Rejected runs keep their run record and judge reasons in `artifacts/queue/state.json`.

## Promotion Policy

The local judge currently treats Tier-2 first-verse accuracy as the available proxy
for streaming recall. A task can also provide:

- `baseline_accuracy`: reject if the candidate regresses below this value.
- `min_accuracy`: reject unless the candidate reaches this absolute threshold.
- `onnx_mb`: reject if above `max_onnx_mb`.
- `max_onnx_mb`: defaults to `200`.

Tier 3 is required for promotion.

## Cloud Experiment Workers

The existing Cursor SDK dispatcher can still implement or tune experiments in
parallel. Queue tasks provide the durable source of truth; workers should write a
run record and set their task to `needs_eval` or leave a branch/PR for the local
controller to evaluate.

```bash
cd orchestration
npm install
export CURSOR_API_KEY=...
export OFFLINE_TARTEEL_REPO_URL=https://github.com/your-org/offline-tarteel-lab.git
npx tsx src/planner.ts
npx tsx src/dispatch.ts "Lab task ..."
```

For fully cloud-driven cycles, use the dedicated dispatcher:

```bash
cd orchestration
export CURSOR_API_KEY=...
export OFFLINE_TARTEEL_REPO_URL=https://github.com/your-org/offline-tarteel-lab.git
LAB_AUTONOMY_MAX_CYCLES=1 npm run cloud:autonomy
```

The GitHub Actions workflow `Cursor Cloud Autonomy` wraps the same command for
manual or scheduled runs. Store `CURSOR_API_KEY` as a repository secret before
enabling scheduled operation.

Each scheduled run replenishes the backlog with `lab_tools.autopilot` before
dispatching a bounded Cursor Cloud agent. The planner creates deterministic task
keys, so already-attempted ideas are not re-added on every fresh clone. Queue
state is tracked at `artifacts/queue/state.json`; large artifacts remain ignored.

## Autonomous Merge Gate

Promotion PRs can merge without human action through the `Cursor Promotion
Auto-Merge` workflow. It only merges after the `CI` workflow succeeds and the PR:

- targets `main`
- has a title starting with `Promote offline-tarteel experiment:` or
  `Autopilot offline-tarteel state:`
- for promotion PRs, changes at least one `artifacts/promotions/*.json` manifest
- for state PRs, changes only `artifacts/queue/` and `artifacts/runs/`
- keeps changed files inside the matching allowlist
- for promotion PRs, has promotion manifests with schema
  `offline-tarteel.promotion.v2`

Only `artifacts/queue/`, `artifacts/runs/`, `artifacts/promotions/`,
`experiments/`, `tests/`, `benchmark/`, and `benchmarks/` are auto-merge
eligible. Changes to workflows,
orchestration, package manifests, repo policy, training code, docs, or controller
code require human review.

## Modal Training

Model tasks can request a detached Modal run:

```bash
python -m lab_tools.task_queue add \
  --kind model_only \
  --title "Train phoneme FastConformer smoke model" \
  --payload '{"modal_training":true,"job_name":"fastconformer-smoke","min_accuracy":0.8}'
```

The controller only launches Modal when explicitly allowed:

```bash
python -m lab_tools.autonomous_loop run-once --allow-modal
```

Without `--allow-modal`, it records the intended Modal command and continues with
local gates when possible. This avoids accidental external compute spend.

## Production Target

The expected promotion artifact is an offline streaming package boundary:

- ONNX model artifact with hash and size budget.
- npm package wrapper for optimized streaming inference.
- benchmark reports proving non-regression on the blind corpus.
- promotion manifest tying artifact hashes, package versions, git SHA, and tier
  reports together.
