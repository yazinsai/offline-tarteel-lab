export type OrchestrationConfig = {
  repoUrl: string;
  startingRef: string;
  skipReviewerRequest: boolean;
};

export function loadConfigFromEnv(): OrchestrationConfig {
  const repoUrl = process.env.OFFLINE_TARTEEL_REPO_URL ?? "";
  const startingRef = process.env.OFFLINE_TARTEEL_REF ?? "main";
  if (!repoUrl) {
    throw new Error("Set OFFLINE_TARTEEL_REPO_URL (GitHub HTTPS URL for cloud agents)");
  }
  return {
    repoUrl,
    startingRef,
    skipReviewerRequest: process.env.CURSOR_SKIP_REVIEWER !== "0",
  };
}
