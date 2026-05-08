import type { Run, RunResult } from "@cursor/sdk";

export async function observeRun(run: Run, label: string): Promise<RunResult> {
  if (!run.supports("stream")) {
    console.error(`[${label}] stream unsupported: ${run.unsupportedReason("stream") ?? "unknown"}`);
    return run.wait();
  }

  const streamDone = (async () => {
    for await (const event of run.stream()) {
      switch (event.type) {
        case "status":
          console.error(`[${label}] status=${event.status}`);
          break;
        case "task":
          if (event.text) console.error(`[${label}] task=${event.text}`);
          break;
        case "tool_call":
          if (event.status !== "running") {
            console.error(`[${label}] tool=${event.name} status=${event.status}`);
          }
          break;
        case "request":
          console.error(`[${label}] request=${event.request_id}`);
          break;
      }
    }
  })();

  const result = await run.wait();
  await streamDone;
  console.error(
    `[${label}] result status=${result.status} durationMs=${result.durationMs ?? "unknown"}`,
  );
  if (result.result) {
    console.error(`[${label}] result=${result.result}`);
  }
  if (result.git?.branches.length) {
    for (const branch of result.git.branches) {
      console.error(
        `[${label}] git repo=${branch.repoUrl} branch=${branch.branch ?? "none"} pr=${branch.prUrl ?? "none"}`,
      );
    }
  }
  return result;
}
