/**
 * Thin CLI: pipe JSON metrics to lab Python judge (single source of truth).
 * echo '{...}' | npm run judge
 */
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const labRoot = path.resolve(__dirname, "..", "..");
const py = path.join(labRoot, ".venv", "bin", "python");
const input = readFileSync(0, "utf-8");
const r = spawnSync(py, ["-m", "lab_tools.judge_policy"], {
  input,
  encoding: "utf-8",
  cwd: labRoot,
});
if (r.error) {
  console.error(r.error);
  process.exit(1);
}
process.stdout.write(r.stdout ?? "");
if (r.stderr) process.stderr.write(r.stderr);
process.exit(r.status ?? 1);
