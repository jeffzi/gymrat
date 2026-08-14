import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import type { ScratchRepo } from "./scratch-repo.js";

export const BENCH_FILE = "bench.js";
export const TUNING_FILE = "tuning.txt";
export const BASELINE_LATENCY = 100;

/**
 * A `metric-lines` bench that reports whatever `tuning.txt` holds, defaulting to
 * the untuned latency when the checkout has no tuning file.
 *
 * Written as CommonJS: the scratch repo has no `package.json`, so node reads a
 * `.js` file there as CommonJS on every platform.
 *
 * With `gateFile`, the bench blocks until that file appears — the only way to
 * hold a run open long enough for a second command to collide with it without
 * betting on a sleep being longer than the first run.
 */
export function benchScript(gateFile?: string): string {
  const lines = ['const fs = require("node:fs");'];
  if (gateFile !== undefined) {
    lines.push(
      `const gate = ${JSON.stringify(gateFile)};`,
      "const idle = new Int32Array(new SharedArrayBuffer(4));",
      "const deadline = Date.now() + 60000;",
      "while (!fs.existsSync(gate) && Date.now() < deadline) { Atomics.wait(idle, 0, 0, 25); }",
    );
  }
  lines.push(
    `const tuned = fs.existsSync(${JSON.stringify(TUNING_FILE)})`,
    `  ? fs.readFileSync(${JSON.stringify(TUNING_FILE)}, "utf8").trim()`,
    `  : "${String(BASELINE_LATENCY)}";`,
    'process.stdout.write("METRIC latency=" + tuned + "\\n");',
  );
  return `${lines.join("\n")}\n`;
}

/**
 * Commit the bench script, config, and gitignore into the scratch repo so
 * every worktree the loop checks out carries a runnable bench.
 */
export function commitProject(
  repo: ScratchRepo,
  options: { samples?: number; gateFile?: string } = {},
): void {
  const { samples = 5, gateFile } = options;
  const files: Record<string, string> = {
    ".gitignore": ".gymrat/\n",
    [BENCH_FILE]: benchScript(gateFile),
    "gymrat.json": `${JSON.stringify({
      bench: `node ${BENCH_FILE}`,
      adapter: "metric-lines",
      samples,
      timeoutSeconds: 120,
    })}\n`,
  };
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(repo.dir, name), content);
  }
  execFileSync("git", ["add", ...Object.keys(files)], { cwd: repo.dir, stdio: "pipe" });
  execFileSync("git", ["commit", "-m", "bench harness"], { cwd: repo.dir, stdio: "pipe" });
}
