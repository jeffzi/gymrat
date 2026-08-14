/* oxlint-disable typescript/require-await -- mock action callbacks are async to match the MockStep interface */
/* oxlint-disable typescript/no-unsafe-type-assertion -- narrowing test assertions on partial event objects */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import type { SessionPrompt } from "../../src/supervisor/driver.js";
import type { LaunchEvent } from "../../src/supervisor/events.js";
import { createMockDriver } from "../../src/supervisor/mock.js";
import type { MockStep } from "../../src/supervisor/mock.js";
import { supervise } from "../../src/supervisor/supervise.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** CLI binary path — the built dist entry point. */
const CLI = path.resolve("dist/cli.js");

const BENCH_FILE = "bench.js";
const TUNING_FILE = "tuning.txt";
const BASELINE_LATENCY = 100;
const TUNED_LATENCY = 90;
const SAMPLES = 5;

/** Generous timeout: real git + bench processes run under the mock driver. */
const LONG_TIMEOUT_MS = 120_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePrompt(cwd: string, overrides: Partial<SessionPrompt> = {}): SessionPrompt {
  return {
    kickoff: "optimize the decoder",
    cwd,
    ...overrides,
  };
}

function makeLaunch(overrides: Partial<LaunchEvent> = {}): LaunchEvent {
  return {
    type: "launch",
    timestamp: Date.now(),
    headSha: "abc123def",
    dirty: false,
    maxMinutes: 10,
    maxUsd: undefined,
    model: undefined,
    runbookPath: "/path/to/runbook.md",
    kickoffSummary: "integration test",
    ...overrides,
  };
}

function makeTempLogPath(dir: string): string {
  return path.join(dir, "supervisor-events.jsonl");
}

function readLogLines(logPath: string): unknown[] {
  return fs
    .readFileSync(logPath, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as unknown);
}

/**
 * A `metric-lines` bench that reports whatever `tuning.txt` holds, defaulting
 * to the baseline latency when the checkout has no tuning file.
 *
 * Written as CommonJS: the scratch repo has no `package.json`, so node reads
 * a `.js` file there as CommonJS on every platform.
 */
function benchScript(): string {
  const lines = [
    'const fs = require("node:fs");',
    `const tuned = fs.existsSync(${JSON.stringify(TUNING_FILE)})`,
    `  ? fs.readFileSync(${JSON.stringify(TUNING_FILE)}, "utf8").trim()`,
    `  : "${String(BASELINE_LATENCY)}";`,
    'process.stdout.write("METRIC latency=" + tuned + "\\n");',
  ];
  return `${lines.join("\n")}\n`;
}

/**
 * Commit the bench script, config, and gitignore into the scratch repo so
 * every worktree the loop checks out carries a runnable bench.
 */
function commitProject(repo: ScratchRepo): void {
  const files: Record<string, string> = {
    ".gitignore": ".gymrat/\n",
    [BENCH_FILE]: benchScript(),
    "gymrat.json": `${JSON.stringify({
      bench: `node ${BENCH_FILE}`,
      adapter: "metric-lines",
      samples: SAMPLES,
      timeoutSeconds: 120,
    })}\n`,
  };
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(repo.dir, name), content);
  }
  execFileSync("git", ["add", ...Object.keys(files)], { cwd: repo.dir, stdio: "pipe" });
  execFileSync("git", ["commit", "-m", "bench harness"], { cwd: repo.dir, stdio: "pipe" });
}

/**
 * Run a gymrat CLI command inside the given working directory.
 *
 * Uses execFileSync so it blocks the mock driver's action step until the
 * command finishes — exactly how a real agent drives gymrat.
 */
function gymrat(args: string[], cwd: string): void {
  execFileSync("node", [CLI, ...args], {
    cwd,
    stdio: "pipe",
    encoding: "utf-8",
    timeout: 60_000,
  });
}

/**
 * Tune the experiment worktree to `latency`, simulating the edit an agent
 * would make between iterations.
 */
function tuneExperiment(repoDir: string, latency: number): void {
  const experimentDir = path.join(repoDir, ".gymrat", "worktrees", "experiment");
  fs.writeFileSync(path.join(experimentDir, TUNING_FILE), `${String(latency)}\n`);
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

describe("supervise – integration", () => {
  describe("when a mock agent drives a complete session via real CLI commands", () => {
    let repo: ScratchRepo;
    let logPath: string;
    let result: Awaited<ReturnType<typeof supervise>>;
    let logLines: unknown[];

    beforeAll(async () => {
      repo = createScratchRepo();
      commitProject(repo);
      logPath = makeTempLogPath(repo.dir);

      const cwd = repo.dir;
      const steps: MockStep[] = [
        {
          action: async () => {
            gymrat(["start", "main"], cwd);
          },
        },
        {
          action: async () => {
            tuneExperiment(cwd, TUNED_LATENCY);
            gymrat(["iterate"], cwd);
          },
        },
        {
          action: async () => {
            gymrat(["keep", "-m", "tune latency to 90"], cwd);
          },
        },
        {
          action: async () => {
            gymrat(["finalize"], cwd);
          },
        },
        // Cost step so result.costUsd is non-zero
        { costUsd: 0.42 },
      ];

      const driver = createMockDriver(steps);
      const launch = makeLaunch();

      result = await supervise({
        driver,
        prompt: makePrompt(cwd),
        maxMinutes: 30,
        logPath,
        launch,
      });

      logLines = readLogLines(logPath);
    }, LONG_TIMEOUT_MS);

    afterAll(() => {
      repo.cleanup();
    });

    it("completes the session normally", () => {
      expect(result.endedBy).toBe("session");
      expect(result.outcome.reason).toBe("completed");
    });

    it("writes the launch event as the first log line", () => {
      expect(logLines.length).toBeGreaterThanOrEqual(2);
      expect(logLines[0]).toMatchObject({ type: "launch" });
    });

    it("writes session events in order after the launch", () => {
      const types = logLines.map((line) =>
        typeof line === "object" && line !== null && "type" in line ? line.type : undefined,
      );
      expect(types[0]).toBe("launch");
      // The remaining lines are session events (usage_update from the cost step)
      const rest = types.slice(1);
      expect(rest.length).toBeGreaterThanOrEqual(1);
      expect(rest).toContain("usage_update");
    });

    it("leaves the session log on disk with a finalize record", () => {
      const sessionLog = path.join(repo.dir, ".gymrat", "session.jsonl");
      expect(fs.existsSync(sessionLog)).toBe(true);
      const records = fs
        .readFileSync(sessionLog, "utf-8")
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as { type: string });
      const types = records.map((r) => r.type);
      expect(types).toContain("session");
      expect(types).toContain("finalize");
    });
  });

  describe("when the wall-clock cap fires before the session finishes", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it("interrupts the session and reports endedBy wall-clock", async () => {
      vi.useFakeTimers();

      // A step that takes 5 minutes, but the cap is 1 minute
      const steps: MockStep[] = [{ costUsd: 0.01, delayMs: 300_000 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath(
        fs.mkdtempSync(path.join(fs.realpathSync.native(os.tmpdir()), "sv-int-")),
      );

      const resultPromise = supervise({
        driver,
        prompt: makePrompt("/tmp/test"),
        maxMinutes: 1,
        logPath,
        launch: makeLaunch({ maxMinutes: 1 }),
      });

      await vi.advanceTimersByTimeAsync(60_000);
      const wallClockResult = await resultPromise;

      expect(wallClockResult.endedBy).toBe("wall-clock");
      expect(wallClockResult.outcome.reason).toBe("interrupted");
    });
  });
});
