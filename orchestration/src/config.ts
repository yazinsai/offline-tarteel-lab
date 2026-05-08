export type OrchestrationConfig = {
  apiKey: string;
  repoUrl: string;
  startingRef: string;
  autoCreatePR: boolean;
  skipReviewerRequest: boolean;
  modelId: string;
};

export function loadConfigFromEnv(): OrchestrationConfig {
  const apiKey = process.env.CURSOR_API_KEY ?? "";
  const repoUrl = process.env.OFFLINE_TARTEEL_REPO_URL ?? "";
  const startingRef = process.env.OFFLINE_TARTEEL_REF ?? "main";
  const modelId = process.env.CURSOR_MODEL_ID ?? "composer-2";
  if (!apiKey) {
    throw new Error("Set CURSOR_API_KEY");
  }
  if (!repoUrl) {
    throw new Error("Set OFFLINE_TARTEEL_REPO_URL (GitHub HTTPS URL for cloud agents)");
  }
  return {
    apiKey,
    repoUrl,
    startingRef,
    autoCreatePR: process.env.CURSOR_AUTO_PR === "1",
    skipReviewerRequest: process.env.CURSOR_SKIP_REVIEWER !== "0",
    modelId,
  };
}
