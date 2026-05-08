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
const allowModal = readBool("LAB_AUTONOMY_ALLOW_MODAL");
const autopilotPlan = readBoolDefault("LAB_AUTONOMY_PLAN", true);

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
  ...(allowModal ? ["--allow-modal"] : []),
].join(" ");

const prompt = `Run a bounded autonomous offline-tarteel lab cycle in Cursor Cloud.

Follow this exact operating procedure:
1. Inspect the repository state and avoid unrelated refactors.
2. Install local Python deps if needed: python3 -m pip install -e ".[dev]".
3. Ensure the task queue exists: python3 -m lab_tools.task_queue init.
4. ${autopilotPlan ? `Replenish the autonomous backlog: python3 -m lab_tools.autopilot plan --target-backlog ${targetBacklog}.` : "Do not replenish the backlog unless explicitly needed by the queue state."}
5. Inspect the next queued task with python3 -m lab_tools.task_queue list.
6. Before running evaluation, implement or tune the next queued task according to its payload.agent_instructions. For model_only or joint_model_runtime tasks, launch Modal only when ${allowModal ? "allowed by this run" : "LAB_AUTONOMY_ALLOW_MODAL is enabled"}; otherwise prepare local training/eval improvements without spending external compute.
7. Run the bounded controller: ${runCommand}
8. Run verification: pytest. If orchestration TypeScript changed, also run: cd orchestration && npx tsc --noEmit.
9. Commit only relevant code changes plus JSON metadata under artifacts/queue, artifacts/runs, and artifacts/promotions.
10. If a task is promoted, make the PR title start with "Promote offline-tarteel experiment:".
11. If no task is promoted but the queue/run state changed, make the PR title start with "Autopilot offline-tarteel state:" and include only artifacts/queue and artifacts/runs JSON.
12. If no task is promoted and code changed, open a normal PR for human review rather than using an autonomous-merge title.

Do not commit large model files, audio files, caches, node_modules, virtualenvs, or secrets.
Do not run unbounded loops.`;

const agent = await Agent.create({
  apiKey: config.apiKey,
  model: { id: config.modelId },
  cloud: {
    repos: [{ url: config.repoUrl, startingRef: config.startingRef }],
    autoCreatePR: true,
    skipReviewerRequest: config.skipReviewerRequest,
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
