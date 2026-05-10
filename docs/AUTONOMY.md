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
- `artifacts/experiment_ledger.jsonl` entries for planner memory

Rejected runs keep their run record and judge reasons in `artifacts/queue/state.json`.

## Promotion Policy

The local judge currently treats Tier-2 first-verse accuracy as the available proxy
for streaming recall. A task can also provide:

- `baseline_accuracy`: reject if the candidate regresses below this value.
- `min_accuracy`: reject unless the candidate reaches this absolute threshold.
- `onnx_mb`: reject if above `max_onnx_mb`.
- `max_onnx_mb`: defaults to `200`.

Tier 3 is required for promotion.

### Mandatory Champion Preflight

Do not spend a full `test_corpus_v3` Tier-2 run (`--limit 0`) until the candidate
has beaten the current champion on the cheap fixed preflight slice.

Before editing, read the champion record:

```bash
python -m lab_tools.experiment_ledger champion
```

Open the champion's run record from `artifacts/runs` and state which subsystem
moved the needle versus the runner-up: ASR/decode, matcher, tracker/runtime, or
packaging/resource budget. The next mutation should target that subsystem. If
the champion is a joint stack such as ASR + matcher, matcher-only follow-ups
(reranks, blends, shortlist tweaks on the same decode, fragment-head reranks)
are not promotion candidates unless the preflight below shows a strict win.
Prefer changes that alter decoding/search (beam width/caps, hypothesis
merge/rescore, two-pass procedures with documented CPU/time caps) or true
ASR-side improvements when Modal is allowed.

After implementing a candidate, run candidate and champion on the same Tier-2
slice before the bounded controller's full-corpus gate:

```bash
python -m lab_tools.eval_tier --tier 2 --corpus test_corpus_v3 --limit 32 --experiment <candidate>
python -m lab_tools.eval_tier --tier 2 --corpus test_corpus_v3 --limit 32 --experiment <champion>
```

`--limit 32` is the repo-standard preflight slice unless a task explicitly names
another fixed slice. Promotion requires the candidate accuracy and composite
objective to be strictly greater than the champion on that same slice. A tie is
a failed run for promotion, even if it has "no regression"; record
`champion_objective_not_improved` and stop. With the current objective weights,
alignment metrics collapse mostly to Tier-2 accuracy: roughly +1 correct / 256
is about +0.35pp accuracy, so promotion needs a visible bump toward at least
230/256 rather than repeated 229/256 ties.

If preflight does not strictly beat the champion, do not run full Tier-2
promotion. Revert candidate code/test/manifest edits unless the user explicitly
wants the prototype kept, write an honest rejection run/state artifact, and keep
the PR state-only. If nothing durable needs recording, open no PR.

On failed probes, append compact negative memory under
`artifacts/autonomy_failures/` or in ledger metadata so later runs do not repeat
the same dead pattern:

```json
{
  "schema": "offline-tarteel.autonomy_probe_failure.v1",
  "autopilot_key": "<key>",
  "experiment_family": "<family>",
  "change_class": "matcher_only_fragment_head_rerank",
  "tier2_delta_correct": 0,
  "reason": "champion_objective_not_improved"
}
```

For shard ordering, inspect the queued tasks before implementing. If a shard
pulls a `joint_model_runtime` task while other queued tasks are reference-port or
`model_only` ASR work and the notes say the plateau is smoke/runtime, do not ship
another matcher-only probe. Either wait for the dependency task, or record
`blocked: wrong task ordering` in queue notes/autonomy failures and leave a
state-only update.

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
state is tracked at `artifacts/queue/state.json` and
`artifacts/experiment_ledger.jsonl`; large artifacts remain ignored.

## Autonomous Merge Gate

Promotion PRs can merge without human action through the `Cursor Promotion
Auto-Merge` workflow. It only merges after the `CI` workflow succeeds and the PR:

- targets `main`
- has a title starting with `Promote offline-tarteel experiment:` or
  `Autopilot offline-tarteel state:`
- for promotion PRs, changes at least one `artifacts/promotions/*.json` manifest
- for state PRs, changes only `artifacts/queue/`, `artifacts/runs/`, and
  `artifacts/autonomy_failures/`, and `artifacts/experiment_ledger.jsonl`
- keeps changed files inside the matching allowlist
- for promotion PRs, has promotion manifests with schema
  `offline-tarteel.promotion.v2`

Only `artifacts/queue/`, `artifacts/runs/`, `artifacts/promotions/`,
`artifacts/autonomy_failures/`, `artifacts/experiment_ledger.jsonl`,
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

Cursor Cloud autonomy exposes the same choice through `LAB_AUTONOMY_ALLOW_MODAL`:

- `auto` (default): launch Modal training when `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET` are present; otherwise treat Modal as unavailable.
- `true`: require Modal credentials and fail fast if they are missing.
- `false`: never launch Modal.

When Modal is available, `model_only` and ASR-side `joint_model_runtime` tasks may
launch detached training jobs with `modal run --detach`. Record the Modal app/run
ID, volume/checkpoint path, and intended follow-up evaluation in the queue/run
metadata. Do not commit model binaries or checkpoint dumps; trained artifacts
stay on Modal volumes until a promotion manifest references hashes and size
metadata.

Without Modal, do not replace a blocked ASR/model task with another matcher-only
probe. Record it as blocked or wait for a Modal-enabled shard.

## Production Target

The expected promotion artifact is an offline streaming package boundary:

- ONNX model artifact with hash and size budget.
- npm package wrapper for optimized streaming inference.
- benchmark reports proving non-regression on the blind corpus.
- promotion manifest tying artifact hashes, package versions, git SHA, and tier
  reports together.
