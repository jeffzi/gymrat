/* oxlint-disable typescript/require-await -- mock action callbacks are async to match the MockStep interface */
/* oxlint-disable typescript/no-unsafe-type-assertion -- narrowing test assertions on partial event objects */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { supervise } from "../../src/supervisor/supervise.js";
import { commitProject, TUNING_FILE } from "../fixtures/bench-harness.js";
import { createMockDriver } from "../fixtures/mock-driver.js";
import type { MockStep } from "../fixtures/mock-driver.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";
import { makeLaunch, makePrompt } from "../fixtures/supervisor.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** CLI binary path — the built dist entry point. */
const CLI = path.resolve("dist/cli.js");

const TUNED_LATENCY = 90;

/** Generous timeout: real git + bench processes run under the mock driver. */
const LONG_TIMEOUT_MS = 120_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
        prompt: makePrompt({ cwd }),
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
        prompt: makePrompt({ cwd: "/tmp/test" }),
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
