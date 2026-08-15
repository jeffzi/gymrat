import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { FAILURE_EXIT_CODE } from "../../src/exec.js";
import type { HookInvocation, HookStage } from "../../src/loop/hooks.js";
import { runHook } from "../../src/loop/hooks.js";
import { SESSION_ID } from "../fixtures/constants.js";
import type { HookScripts } from "../fixtures/hook-scripts.js";
import { hookScripts } from "../fixtures/hook-scripts.js";
import { expectedHookRecord, iterationRecord, sessionRecord } from "../fixtures/session-records.js";

/** The cap the runner holds each of a hook's channels to before it reaches gymrat's own output. */
const RELAY_LIMIT_BYTES = 8192;

let tempDir: string;
let experimentDir: string;
let hookCommand: HookScripts["hookCommand"];
let printing: HookScripts["printing"];

/**
 * Park `content` in `name` beside the hook and give back the line that prints it
 * verbatim on `channel`.
 *
 * The text lives in a file rather than inside the script so a payload far larger
 * than a source literal wants to be still reaches the runner byte for byte.
 */
function printingLine(name: string, channel: "stdout" | "stderr", content: string): string {
  const dataPath = path.join(tempDir, name);
  fs.writeFileSync(dataPath, content);
  return `process.${channel}.write(fs.readFileSync(${JSON.stringify(dataPath)}, "utf-8"));\n`;
}

/** Park `content` beside the hook and give back a command that prints it verbatim. */
function printingContentOf(fileName: string, content: string): string {
  return hookCommand(`import fs from "node:fs";\n${printingLine(fileName, "stdout", content)}`);
}

/** A command that prints parked text on both channels and then exits 3. */
function failingContentOf(stem: string, streams: { stdout: string; stderr: string }): string {
  return hookCommand(
    `import fs from "node:fs";\n` +
      printingLine(`${stem}-stdout.txt`, "stdout", streams.stdout) +
      printingLine(`${stem}-stderr.txt`, "stderr", streams.stderr) +
      `process.exitCode = 3;\n`,
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
      expect(run.record.stdoutBytes).toBe(Buffer.byteLength(text, "utf-8"));
    });

    it("stops mid-line without splitting a multi-byte character", async () => {
      // Arrange
      const content = "é".repeat(5000);
      const command = printingContentOf("one-long-line.txt", content);

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      const lines = labeledLines(run.report, "before");
      expect.soft(lines).toStrictEqual(["é".repeat(RELAY_LIMIT_BYTES / 2)]);
      expect(run.record.stdoutBytes).toBe(Buffer.byteLength(content, "utf-8"));
    });
  });

  describe("when a failing hook prints more than the 8 KiB gymrat will relay on either channel", () => {
    it("holds each channel to its own cap, cut at the last whole line that fits", async () => {
      // Arrange
      const outLine = "a".repeat(100);
      const errLine = "b".repeat(100);
      const stdout = `${outLine}\n`.repeat(200);
      const command = failingContentOf("both-channels", {
        stdout,
        stderr: `${errLine}\n`.repeat(200),
      });

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect
        .soft(labeledLines(run.report, "before"))
        .toStrictEqual([
          ...Array.from({ length: 81 }, () => outLine),
          "hook exited 3",
          ...Array.from({ length: 81 }, () => errLine),
        ]);
      expect(run.record.stdoutBytes).toBe(Buffer.byteLength(stdout, "utf-8"));
    });

    it("stops stderr mid-line without splitting a multi-byte character", async () => {
      // Arrange
      const command = failingContentOf("long-stderr-line", {
        stdout: "",
        stderr: "é".repeat(5000),
      });

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect(labeledLines(run.report, "before")).toStrictEqual([
        "hook exited 3",
        "é".repeat(RELAY_LIMIT_BYTES / 2),
      ]);
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

    it("records exit code 127 when the hook command does not exist", async () => {
      // Arrange — a command that cannot be found
      const command = "nonexistent-command-abc123xyz";

      // Act
      const run = await runHook(invocationOf(command));

      // Assert
      expect(run.record.exitCode).toBe(127);
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

  describe("when the invocation carries an abort signal", () => {
    it("kills the hook and records the outcome instead of rejecting", async () => {
      // Arrange
      const controller = new AbortController();
      const command = hookCommand("setTimeout(() => {}, 10000);\n");
      setTimeout(() => {
        controller.abort();
      }, 50);
      const invocation = { ...invocationOf(command), signal: controller.signal };

      // Act
      const run = await runHook(invocation);

      // Assert
      expect.soft(run.record.exitCode).toBe(FAILURE_EXIT_CODE);
      expect.soft(run.record.timedOut).toBe(false);
      expect(run.record.durationMs).toBeLessThan(4000);
    }, 15_000);
  });
});
