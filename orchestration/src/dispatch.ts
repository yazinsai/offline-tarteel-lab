/**
 * Cloud worker dispatch using Cursor SDK.
 * Usage: CURSOR_API_KEY=... OFFLINE_TARTEEL_REPO_URL=https://github.com/org/offline-tarteel.git npm run dispatch -- "prompt"
 */
import { Agent, CursorAgentError } from "@cursor/sdk";
import { loadConfigFromEnv } from "./config.js";
import { observeRun } from "./runObserver.js";

const prompt = process.argv.slice(2).join(" ").trim();
if (!prompt) {
  console.error('Usage: npm run dispatch -- "Implement experiment X"');
  process.exit(1);
}

const config = loadConfigFromEnv();

const agent = await Agent.create({
  apiKey: config.apiKey,
  model: { id: config.modelId },
  cloud: {
    repos: [{ url: config.repoUrl, startingRef: config.startingRef }],
    autoCreatePR: config.autoCreatePR,
    skipReviewerRequest: config.skipReviewerRequest,
  },
});

try {
  const run = await agent.send(prompt);
  console.error(`[dispatch] agent=${agent.agentId} run=${run.id}`);
  const result = await observeRun(run, "dispatch");
  if (result.status !== "finished") {
    console.error(`[dispatch] run ${result.id} ended as ${result.status}`);
    process.exit(2);
  }
  console.error(`[dispatch] done status=${result.status}`);
} catch (e) {
  if (e instanceof CursorAgentError) {
    console.error(`[dispatch] startup failed: ${e.message} retryable=${e.isRetryable}`);
    process.exit(e.isRetryable ? 75 : 1);
  }
  throw e;
} finally {
  await agent[Symbol.asyncDispose]();
}
