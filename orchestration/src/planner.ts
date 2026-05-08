/**
 * Read lab task queue state and print a short execution plan + dispatch snippet.
 *
 * Usage:
 *   npx tsx src/planner.ts [path/to/state.json]
 *   OFFLINE_TARTEEL_LAB_QUEUE=../artifacts/queue/state.json npx tsx src/planner.ts
 */
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

const argPath = process.argv[2];
const envPath = process.env.OFFLINE_TARTEEL_LAB_QUEUE;
const defaultPath = resolve(process.cwd(), "../artifacts/queue/state.json");
const statePath = argPath
  ? resolve(argPath)
  : envPath
    ? resolve(envPath)
    : defaultPath;

if (!existsSync(statePath)) {
  console.error(`No queue state at ${statePath}`);
  process.exit(2);
}

const state = JSON.parse(readFileSync(statePath, "utf-8")) as {
  schema?: string;
  tasks?: Array<{
    id: string;
    status: string;
    kind: string;
    title: string;
    payload?: Record<string, unknown>;
  }>;
};

const tasks = state.tasks ?? [];
const byStatus: Record<string, number> = {};
for (const t of tasks) {
  byStatus[t.status] = (byStatus[t.status] ?? 0) + 1;
}

const next = tasks.find((t) => t.status === "queued");
const running = tasks.filter((t) => t.status === "running");

console.log("## Queue summary\n");
console.log(JSON.stringify(byStatus, null, 2));
console.log("");

if (next) {
  const payload = JSON.stringify(next.payload ?? {});
  console.log("## Next queued\n");
  console.log(`- ${next.id} (${next.kind}): ${next.title}`);
  console.log(`- payload: ${payload}`);
  console.log("");
  console.log("## Suggested steps\n");
  console.log("1. Claim: `lab-loop tick` (or dry-run: `lab-loop tick --dry-run`)");
  console.log("2. After work: `python -m lab_tools.eval_tier --tier 1`");
  console.log("3. Record: `lab-task set-status --id ... --status needs_eval --run-record <path>`");
  console.log("");
  const hint =
    `Lab task ${next.id} (${next.kind}): ${next.title}\nPayload: ${payload}\n` +
    `Run tiered eval and attach run record JSON.`;
  console.log("## Dispatch one-liner (paste into dispatch.ts argv)\n");
  console.log(JSON.stringify(hint));
} else {
  console.log("No queued tasks.");
}

if (running.length) {
  console.log("\n## In progress\n");
  for (const t of running) {
    console.log(`- ${t.id}: ${t.title}`);
  }
}
