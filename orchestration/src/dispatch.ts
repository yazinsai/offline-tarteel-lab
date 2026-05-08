/**
 * Cloud worker dispatch using Cursor SDK.
 * Usage: CURSOR_API_KEY=... OFFLINE_TARTEEL_REPO_URL=https://github.com/org/offline-tarteel.git npm run dispatch -- "prompt"
 */
import { Agent, CursorAgentError } from "@cursor/sdk";

const prompt = process.argv.slice(2).join(" ").trim();
if (!prompt) {
  console.error('Usage: npm run dispatch -- "Implement experiment X"');
  process.exit(1);
}

const apiKey = process.env.CURSOR_API_KEY;
if (!apiKey) {
  console.error("Missing CURSOR_API_KEY");
  process.exit(1);
}

const repoUrl = process.env.OFFLINE_TARTEEL_REPO_URL!;
const startingRef = process.env.OFFLINE_TARTEEL_REF ?? "main";

const agent = Agent.create({
  apiKey,
  model: { id: "composer-2" },
  cloud: {
    repos: [{ url: repoUrl, startingRef }],
    autoCreatePR: process.env.CURSOR_AUTO_PR === "1",
    skipReviewerRequest: process.env.CURSOR_SKIP_REVIEWER !== "0",
  },
});

try {
  const run = await agent.send(prompt);
  console.error(`[dispatch] agent=${agent.agentId} run=${run.id}`);
  const result = await run.wait();
  if (result.status === "error") {
    console.error(`[dispatch] run failed: ${result.id}`);
    process.exit(2);
  }
  console.error(`[dispatch] done status=${result.status}`);
} catch (e) {
  if (e instanceof CursorAgentError) {
    console.error(`[dispatch] startup failed: ${e.message} retryable=${e.isRetryable}`);
    process.exit(1);
  }
  throw e;
} finally {
  await agent[Symbol.asyncDispose]();
}
