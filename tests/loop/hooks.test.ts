import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { FAILURE_EXIT_CODE } from "../../src/exec.js";
import type { HookInvocation, HookStage } from "../../src/loop/hooks.js";
import { runHook } from "../../src/loop/hooks.js";
import { SESSION_ID } from "../fixtures/constants.js";
import type { HookScripts } from "../fixtures/hook-scripts.js";
import { hookScripts } from "../fixtures/hook-scripts.js";
import { expectedHookRecord, iterationRecord, sessionRecord } from "../fixtures/session-records.js";

/** The cap the runner holds hook stdout to before it reaches gymrat's own output. */
const STDOUT_LIMIT_BYTES = 8192;

let tempDir: string;
let experimentDir: string;
let hookCommand: HookScripts["hookCommand"];
let printing: HookScripts["printing"];

/** Park `content` beside the hook and give back a command that prints it verbatim. */
function printingContentOf(name: string, content: string): string {
  const dataPath = path.join(tempDir, name);
  fs.writeFileSync(dataPath, content);
  return hookCommand(
    `import fs from "node:fs";\n` +
      `process.stdout.write(fs.readFileSync(${JSON.stringify(dataPath)}, "utf-8"));\n`,
  );
}

/** A `before` invocation on the scratch worktree, overridable field by field. */
function invocationOf(command: string, overrides: Partial<HookInvocation> = {}): HookInvocation {
  return {
    command,
    stage: "before",
    seq: 2,
    session: sessionRecord({
      sessionId: SESSION_ID,
      worktrees: {
        experiment: experimentDir,
        baseline: path.join(tempDir, "side-baseline"),
      },
    }),
    lastIteration: null,
    iterationCount: 1,
    ...overrides,
  };
}

/**
 * The report's lines with their `[stage]` label stripped off.
 *
 * Every line the runner emits carries the label, so a line without one is a
 * leak of unlabeled hook output rather than something to quietly pass through.
 */
function labeledLines(report: string, stage: HookStage): string[] {
  if (report === "") {
    return [];
  }
  const prefix = `[${stage}] `;
  return report.split("\n").map((line) => {
    if (!line.startsWith(prefix)) {
      throw new Error(`expected every hook report line to be labeled ${prefix.trim()}: ${line}`);
    }
    return line.slice(prefix.length);
  });
}

beforeEach(() => {
  tempDir = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-hooks-")));
  experimentDir = path.join(tempDir, "side-experiment");
  fs.mkdirSync(experimentDir, { recursive: true });
  ({ hookCommand, printing } = hookScripts(tempDir));
});

afterEach(() => {
  fs.rmSync(tempDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
});

describe("runHook", () => {
  describe("when the configured command runs", () => {
    it.each([
      {
        stage: "before" as const,
        seq: 3,
        lastIteration: null,
        iterationCount: 2,
      },
      {
        stage: "after" as const,
        seq: 3,
        lastIteration: iterationRecord({ seq: 3 }),
        iterationCount: 3,
      },
    ])(
      "hands the $stage hook a payload naming the stage, the worktree and the session",
      async ({ stage, seq, lastIteration, iterationCount }) => {
        // Arrange
        const command = hookCommand("process.stdin.pipe(process.stdout);\n");

        // Act
        const run = await runHook(
          invocationOf(command, { stage, seq, lastIteration, iterationCount }),
        );

        // Assert
        const payload: unknown = JSON.parse(labeledLines(run.report, stage).join("\n"));
        expect(payload).toStrictEqual({
          stage,
          experimentDir,
          seq,
          lastIteration,
          session: {
            sessionId: SESSION_ID,
            baseline: { ref: "main", sha: "a".repeat(40) },
            branch: `gymrat/${SESSION_ID}`,
            iterationCount,
          },
        });
      },
    );

    it("runs the command in the experiment worktree", async () => {
      // Arrange
      const command = hookCommand(
        `import fs from "node:fs";\nfs.writeFileSync("landed.txt", "here");\n`,
      );

      // Act
      await runHook(invocationOf(command));

      // Assert
      expect(fs.readFileSync(path.join(experimentDir, "landed.txt"), "utf-8")).toBe("here");
    });

    it("labels every line the hook printed with its stage", async () => {
      // Arrange
      const command = printing("archived the samples", "pushed the branch");

      // Act
      const run = await runHook(invocationOf(command, { stage: "after" }));

      // Assert
      expect(run.report).toBe("[after] archived the samples\n[after] pushed the branch");
    });

    it("reports nothing at all for a hook that printed nothing", async () => {
      // Arrange
      const command = hookCommand("");

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect(run.report).toBe("");
    });

    it("keeps a successful hook's stderr out of the report", async () => {
      // Arrange
      const command = hookCommand(
        `process.stdout.write("warmed the cache\\n");\n` +
          `process.stderr.write("cache was already warm\\n");\n`,
      );

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect(run.report).toBe("[before] warmed the cache");
    });

    it("records the invocation with the bytes the hook printed", async () => {
      // Arrange
      const command = printing("hello");

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect(run.record).toStrictEqual(
        expectedHookRecord({ stage: "before", seq: 2, exitCode: 0, stdoutBytes: 6 }),
      );
    });
  });

  describe("when the hook prints more than the 8 KiB gymrat will relay", () => {
    it("stops at the last whole line that fits", async () => {
      // Arrange
      const line = "a".repeat(100);
      const text = `${line}\n`.repeat(200);
      const command = printingContentOf("many-lines.txt", text);

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      const lines = labeledLines(run.report, "before");
      expect.soft(lines).toStrictEqual(Array.from({ length: 81 }, () => line));
      expect(run.record.stdoutBytes).toBe(20200);
    });

    it("stops mid-line without splitting a multi-byte character", async () => {
      // Arrange
      const command = printingContentOf("one-long-line.txt", "é".repeat(5000));

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      const lines = labeledLines(run.report, "before");
      expect.soft(lines).toStrictEqual(["é".repeat(STDOUT_LIMIT_BYTES / 2)]);
      expect(run.record.stdoutBytes).toBe(10000);
    });
  });

  describe("when the hook cannot steer the loop the way it meant to", () => {
    it("reports a non-zero exit alongside its stderr instead of failing the iterate", async () => {
      // Arrange
      const command = hookCommand(
        `process.stdout.write("checked the cache\\n");\n` +
          `process.stderr.write("no warm copy\\n");\n` +
          `process.exitCode = 3;\n`,
      );

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect
        .soft(labeledLines(run.report, "before"))
        .toStrictEqual(["checked the cache", "hook exited 3", "no warm copy"]);
      expect(run.record).toStrictEqual(
        expectedHookRecord({ stage: "before", seq: 2, exitCode: 3, stdoutBytes: 18 }),
      );
    });

    it("kills a hook that outruns its timeout and says so", async () => {
      // Arrange
      const command = hookCommand("setTimeout(() => {}, 5000);\n");

      // Act
      const run = await runHook(invocationOf(command, { timeoutMs: 200 }));

      // Assert
      expect.soft(labeledLines(run.report, "before")).toStrictEqual(["hook timed out after 200ms"]);
      expect.soft(run.record.timedOut).toBe(true);
      expect.soft(run.record.exitCode).toBe(FAILURE_EXIT_CODE);
      expect(run.record.durationMs).toBeLessThan(4000);
    });

    it("reports a hook that never started instead of raising", async () => {
      // Arrange
      const command = printing("never runs");
      const vanishedDir = path.join(tempDir, "vanished");

      // Act
      const run = await runHook(
        invocationOf(command, {
          session: sessionRecord({
            sessionId: SESSION_ID,
            worktrees: {
              experiment: vanishedDir,
              baseline: path.join(tempDir, "side-baseline"),
            },
          }),
        }),
      );

      // Assert
      const lines = labeledLines(run.report, "before");
      expect.soft(lines[0]).toBe(`hook exited ${FAILURE_EXIT_CODE}`);
      // Windows maps a missing cwd to ENOTDIR; POSIX to ENOENT.
      expect.soft(lines.slice(1).join("\n")).toMatch(/ENOENT|ENOTDIR/);
      expect.soft(run.record.stdoutBytes).toBe(0);
      expect(run.record.timedOut).toBe(false);
    });
  });
});
