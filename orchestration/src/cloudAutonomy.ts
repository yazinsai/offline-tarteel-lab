/**
 * Dispatch a bounded autonomous lab cycle to Cursor Cloud.
 *
 * The cloud agent runs against a fresh GitHub clone and can open a PR containing
 * code changes plus small promotion metadata. Large model artifacts remain
 * ignored by the repo.
 */
import { Agent, CursorAgentError } from "@cursor/sdk";
import { loadConfigFromEnv } from "./config.js";
import { observeRun } from "./runObserver.js";

function readInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return n;
}

function readBool(name: string): boolean {
  return process.env[name] === "1" || process.env[name]?.toLowerCase() === "true";
}

function readModalMode(): "auto" | "false" | "true" {
  const raw = (process.env.LAB_AUTONOMY_ALLOW_MODAL ?? "auto").toLowerCase();
  if (raw === "1" || raw === "true") return "true";
  if (raw === "0" || raw === "false") return "false";
  if (raw === "auto") return "auto";
  throw new Error("LAB_AUTONOMY_ALLOW_MODAL must be auto, true, or false");
}

function readBoolDefault(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (!raw) return fallback;
  return raw === "1" || raw.toLowerCase() === "true";
}

const config = loadConfigFromEnv();
const maxCycles = readInt("LAB_AUTONOMY_MAX_CYCLES", 1);
const targetBacklog = readInt("LAB_AUTONOMY_TARGET_BACKLOG", 3);
const limit = readInt("LAB_AUTONOMY_LIMIT", 12);
const corpus = process.env.LAB_AUTONOMY_CORPUS ?? "test_corpus_v3";
const modalMode = readModalMode();
const hasModalCredentials = Boolean(process.env.MODAL_TOKEN_ID && process.env.MODAL_TOKEN_SECRET);
const allowModal = modalMode === "true" || (modalMode === "auto" && hasModalCredentials);
const autopilotPlan = readBoolDefault("LAB_AUTONOMY_PLAN", true);
const shardIndex = readInt("LAB_AUTONOMY_SHARD_INDEX", 0);
const shardTotal = readInt("LAB_AUTONOMY_SHARD_TOTAL", 1);

if (shardTotal < 1) {
  throw new Error("LAB_AUTONOMY_SHARD_TOTAL must be at least 1");
}

if (shardIndex >= shardTotal) {
  throw new Error("LAB_AUTONOMY_SHARD_INDEX must be less than LAB_AUTONOMY_SHARD_TOTAL");
}

if (modalMode === "true" && !hasModalCredentials) {
  throw new Error("Modal launches require MODAL_TOKEN_ID and MODAL_TOKEN_SECRET");
}

const modalEnv =
  allowModal && process.env.MODAL_TOKEN_ID && process.env.MODAL_TOKEN_SECRET
    ? {
        MODAL_TOKEN_ID: process.env.MODAL_TOKEN_ID,
        MODAL_TOKEN_SECRET: process.env.MODAL_TOKEN_SECRET,
        ...(process.env.MODAL_PROFILE ? { MODAL_PROFILE: process.env.MODAL_PROFILE } : {}),
        ...(process.env.MODAL_ENVIRONMENT
          ? { MODAL_ENVIRONMENT: process.env.MODAL_ENVIRONMENT }
          : {}),
      }
    : {};

const runCommand = [
  "python3",
  "-m",
  "lab_tools.autonomous_loop",
  "run",
  "--max-cycles",
  String(maxCycles),
  "--limit",
  String(limit),
  "--corpus",
  corpus,
  "--shard-index",
  String(shardIndex),
  "--shard-total",
  String(shardTotal),
  ...(allowModal ? ["--allow-modal"] : []),
].join(" ");

const prompt = `Run a bounded autonomous offline-tarteel lab cycle in Cursor Cloud.
This is queue shard ${shardIndex + 1}/${shardTotal}; only evaluate the queued task selected by that shard.
Modal mode is ${modalMode}; Modal launches are ${allowModal ? "enabled for this run" : "disabled because credentials are absent or the mode is false"}.

Follow this exact operating procedure:
1. Inspect the repository state and avoid unrelated refactors.
2. Install local Python deps if needed: python3 -m pip install -e ".[dev]".
3. Review prior blocked attempts: ls artifacts/autonomy_failures || true; if present, inspect the recent JSON files and avoid repeating rejected paths, binary artifacts, or policy violations.
4. If ${allowModal ? "a model_only or ASR-side joint_model_runtime task needs GPU training, install the Modal CLI if missing (python3 -m pip install modal) and launch the documented training command detached with modal run --detach." : "a task needs GPU training, record it as blocked without spending local time on matcher-only substitutes; Modal is not available in this run."}
5. Ensure the task queue exists: python3 -m lab_tools.task_queue init.
6. ${autopilotPlan ? `Replenish the autonomous backlog: python3 -m lab_tools.autopilot plan --target-backlog ${targetBacklog}.` : "Do not replenish the backlog unless explicitly needed by the queue state."}
7. Inspect the next queued task with python3 -m lab_tools.task_queue list.
8. Before editing, read the current champion: python3 -m lab_tools.experiment_ledger champion. Open the champion's run record from artifacts/runs and name which subsystem moved the needle versus the runner-up: ASR/decode, matcher, tracker/runtime, or packaging/resource budget. Mutate that subsystem unless a preflight proves another path wins.
9. If the champion is a joint ASR + matcher stack, do not ship matcher-only follow-ups (reranks, blends, shortlist tweaks, fragment-head reranks on the same decode) unless the preflight in step 12 strictly beats the champion. Prefer decode/search changes with documented CPU/time caps, hypothesis merge/rescore or two-pass procedures, or ASR-side work when Modal is allowed.
10. If this shard selected a joint_model_runtime task but queued reference-port or model_only ASR tasks are still the real dependency and queue notes say the plateau is smoke/runtime, do not implement another matcher-only probe. Record blocked: wrong task ordering in queue notes and/or artifacts/autonomy_failures, then stop with a state-only update.
11. Before running evaluation, implement or tune the next queued task according to its payload.agent_instructions. For model_only or ASR-side joint_model_runtime tasks, use Modal training when ${allowModal ? "the task calls for it; launch detached, record the app/run/checkpoint metadata, and do not commit model binaries" : "a future run has LAB_AUTONOMY_ALLOW_MODAL=auto with credentials or LAB_AUTONOMY_ALLOW_MODAL=true"}. Do not replace a blocked ASR/model task with another matcher-only probe.
12. Mandatory cheap Tier-2 preflight before full corpus: run the candidate and current champion on the same fixed slice, using --limit 32 unless the task names a different repo-standard slice. Example: python3 -m lab_tools.eval_tier --tier 2 --corpus ${corpus} --limit 32 --experiment <candidate>; then repeat with --experiment <champion>. Compare correct counts/accuracy from the Tier-2 reports and compute the composite objective using the current scorer weights if the report does not include it. The candidate must be strictly greater on both accuracy and objective. Tie = failed promotion with reason champion_objective_not_improved; "no regression" is not success.
13. If the preflight does not strictly beat the champion, do not run full Tier-2 --limit 0 or the bounded controller for promotion. Revert candidate code/test/benchmark-manifest edits, record an honest rejection in artifacts/runs, artifacts/queue, artifacts/experiment_ledger.jsonl, or artifacts/autonomy_failures, and keep any PR state-only. If nothing durable changed, open no PR.
14. On failed probes, append compact negative memory under artifacts/autonomy_failures/ or ledger metadata with autopilot_key or experiment_family, change_class, tier2_delta_correct, and reason. Be specific enough to block repeats such as "fragment head rerank on top of joint02".
15. If the next task payload has reference_repo_url or reference_baseline, a state-only smoke rerun is invalid. Clone the reference_repo_url if the sibling reference_repo path is missing, create or update a real runnable experiment under experiments/, register/select it by setting the queued task payload.experiment to that experiment name in artifacts/queue/state.json, and only then run the bounded controller. If you cannot clone or cannot create a real audio-driven experiment, stop with a non-zero exit instead of opening another rejected smoke/state-only PR.
16. Run the bounded controller only after the strict preflight passes: ${runCommand}
17. Run verification: pytest. If orchestration TypeScript changed, also run: cd orchestration && npx tsc --noEmit.
18. Commit only relevant promoted code changes plus JSON metadata under artifacts/queue, artifacts/runs, artifacts/promotions, artifacts/autonomy_failures, and artifacts/experiment_ledger.jsonl.
19. Open the PR against the ${config.startingRef} branch.
20. PR titles must include the measured corpus result and the delta versus the current best whenever evaluation produced both values. Use percentage points for accuracy/objective deltas, e.g. "86.3%, +1.5pp", "84.8%, tie", or "84.4%, -0.4pp". Put this near the front of the title so it is visible in GitHub lists.
21. If a task is promoted, make the PR title start with "Promote offline-tarteel experiment: <result>, <delta> —".
22. If the bounded controller rejects the task or no task is promoted, revert all candidate code, test, and benchmark-manifest edits before committing. The PR must be state-only: artifacts/queue, artifacts/runs, artifacts/autonomy_failures, and artifacts/experiment_ledger.jsonl.
23. If no task is promoted but queue/run/ledger/failure-memory state changed, make the PR title start with "Autopilot offline-tarteel state: <result>, <delta-or-reason> —" and include only artifacts/queue, artifacts/runs, artifacts/autonomy_failures, and artifacts/experiment_ledger.jsonl.
24. Do not open a normal code PR for a rejected autonomous task. Human-review code PRs are only for explicitly useful prototypes that are not rejected controller runs.
25. The PR description must explain the attempted experiment in concrete terms so reviewers do not need to read the diff to understand it. Include these sections:
   - "Attempted change": the exact parameter/model/tracker/rule change tried, with before -> after values when applicable.
   - "Files changed during the probe": files you edited while testing, including files later reverted for a rejected task.
   - "Champion forensics": champion run ID, champion experiment, and the subsystem that moved the needle versus runner-up.
   - "Preflight result": candidate vs champion command, sample count, correct counts, accuracy/objective, delta, and whether the strict improvement gate passed.
   - "Evaluation result": corpus, sample count, command, accuracy/objective, the four objective components (streaming_alignment_accuracy, correction_precision, verse_boundary_f1, latency_budget_score), gate outcome, and rejection/promotion reason.
   - "Final PR contents": what remains committed after promotion or after reverting a rejected probe.
   - "What to avoid next": any negative-memory lesson if the attempt failed.

Do not create, modify, or commit audio files, model binaries, tensor dumps, caches, node_modules, virtualenvs, or secrets.
Do not add files with extensions .wav, .mp3, .flac, .ogg, .m4a, .onnx, .pt, .pth, .ckpt, .bin, .safetensors, .npy, or .npz.
Do not create, modify, duplicate, pad, or otherwise game benchmark/test_corpus* manifests or datasets/registry.yaml. If ${corpus} is too small to satisfy promotion gates, reject/record the task; do not edit the corpus.
Do not infer surah/ayah labels from audio file names, paths, manifests, sidecar metadata, corpus IDs, or benchmark row ordering. Improvements must generalize from audio/model/tracker behavior, not corpus-label leakage.
Benchmark changes must be JSON metadata or manifests only, and should reference existing committed sample files.
Do not run unbounded loops.`;

const agent = await Agent.create({
  apiKey: config.apiKey,
  model: { id: config.modelId },
  cloud: {
    repos: [{ url: config.repoUrl, startingRef: config.startingRef }],
    workOnCurrentBranch: false,
    autoCreatePR: true,
    skipReviewerRequest: config.skipReviewerRequest,
    envVars: modalEnv,
  },
});

try {
  const run = await agent.send(prompt);
  console.error(`[cloud-autonomy] agent=${agent.agentId} run=${run.id}`);
  const result = await observeRun(run, "cloud-autonomy");
  if (result.status !== "finished") {
    console.error(`[cloud-autonomy] run ${result.id} ended as ${result.status}`);
    process.exit(2);
  }
  console.error(`[cloud-autonomy] done status=${result.status}`);
} catch (err) {
  if (err instanceof CursorAgentError) {
    console.error(
      `[cloud-autonomy] startup failed: ${err.message} retryable=${err.isRetryable}`,
    );
    process.exit(err.isRetryable ? 75 : 1);
  }
  throw err;
} finally {
  await agent[Symbol.asyncDispose]();
}
