import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { HookInvocation, HookRun, HookStage } from "../../src/loop/hooks.js";
import { runHook } from "../../src/loop/hooks.js";
import type { IterationRecord, SessionRecord } from "../../src/session/records.js";
import {
  iterationRecord,
  sessionRecord as sessionRecordDefaults,
} from "../fixtures/session-records.js";
import { removeTempRoot } from "../setup/temp-root.js";

const SESSION_ID = "20260808-141530-abcd";

/** The cap the runner holds hook stdout to before it reaches gymrat's own output. */
const STDOUT_LIMIT_BYTES = 8192;

let tempDir: string;
let hooksDir: string;

/** The session every invocation here describes to its hook. */
function sessionRecord(): SessionRecord {
  return sessionRecordDefaults({
    sessionId: SESSION_ID,
    worktrees: {
      experiment: path.join(tempDir, "side-experiment"),
      baseline: path.join(tempDir, "side-baseline"),
    },
  });
}

/** A measured iteration numbered `seq`, the shape a hook is handed as `lastIteration`. */
function iteration(seq: number): IterationRecord {
  return iterationRecord({ seq });
}

/** A `before` invocation on the scratch hooks directory, overridable field by field. */
function invocationOf(overrides: Partial<HookInvocation> = {}): HookInvocation {
  return {
    hooksDir,
    stage: "before",
    seq: 2,
    session: sessionRecord(),
    lastIteration: null,
    iterationCount: 1,
    ...overrides,
  };
}

/** Write `body` as `<stage>.sh`, executable unless `mode` says otherwise. */
function writeHookScript(stage: HookStage, body: string, mode = 0o755): void {
  const scriptPath = path.join(hooksDir, `${stage}.sh`);
  fs.writeFileSync(scriptPath, `#!/bin/sh\n${body}\n`);
  fs.chmodSync(scriptPath, mode);
}

/** Park `content` beside the hooks and give back a hook body that writes it to stdout. */
function catBody(name: string, content: string): string {
  fs.writeFileSync(path.join(hooksDir, name), content);
  return `cat "$(dirname "$0")/${name}"`;
}

/** Run the invocation and fail the test when the runner skipped the hook. */
async function runFired(overrides: Partial<HookInvocation> = {}): Promise<HookRun> {
  const run = await runHook(invocationOf(overrides));
  if (run === undefined) {
    throw new Error("expected the hook to have fired");
  }
  return run;
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
  hooksDir = path.join(tempDir, "gymrat.hooks");
  fs.mkdirSync(hooksDir, { recursive: true });
});

afterEach(() => {
  removeTempRoot(tempDir);
});

describe("runHook", () => {
  describe("when the stage has no runnable script", () => {
    it("skips a hooks directory holding no script for the stage", async () => {
      // Act
      const run = await runHook(invocationOf());

      // Assert
      expect(run).toBeUndefined();
    });

    it.skipIf(process.platform === "win32")(
      "skips a script the filesystem does not mark executable",
      async () => {
        // Arrange
        writeHookScript("before", 'echo "should never run"', 0o644);

        // Act
        const run = await runHook(invocationOf());

        // Assert
        expect(run).toBeUndefined();
      },
    );
  });

  describe("when the stage's script is executable", () => {
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
        lastIteration: iteration(3),
        iterationCount: 3,
      },
    ])(
      "hands the $stage hook a payload naming the stage, the worktree and the session",
      async ({ stage, seq, lastIteration, iterationCount }) => {
        // Arrange
        writeHookScript(stage, "cat");

        // Act
        const run = await runFired({ stage, seq, lastIteration, iterationCount });

        // Assert
        const payload: unknown = JSON.parse(labeledLines(run.report, stage).join("\n"));
        expect(payload).toStrictEqual({
          stage,
          experimentDir: sessionRecord().worktrees.experiment,
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

    it("labels every line the hook printed with its stage", async () => {
      // Arrange
      writeHookScript("after", 'echo "archived the samples"\necho "pushed the branch"');

      // Act
      const run = await runFired({ stage: "after" });

      // Assert
      expect(run.report).toBe("[after] archived the samples\n[after] pushed the branch");
    });

    it("reports nothing at all for a hook that printed nothing", async () => {
      // Arrange
      writeHookScript("before", "exit 0");

      // Act
      const run = await runFired();

      // Assert
      expect(run.report).toBe("");
    });

    it("records the invocation with the bytes the hook printed", async () => {
      // Arrange
      writeHookScript("before", 'echo "hello"');

      // Act
      const run = await runFired();

      // Assert
      expect(run.record).toStrictEqual({
        type: "hook",
        stage: "before",
        seq: 2,
        exitCode: 0,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        durationMs: expect.any(Number),
        stdoutBytes: 6,
        timedOut: false,
      });
    });
  });

  describe("when the hook prints more than the 8 KiB gymrat will relay", () => {
    it("stops at the last whole line that fits", async () => {
      // Arrange
      const line = "a".repeat(100);
      const text = `${line}\n`.repeat(200);
      writeHookScript("before", catBody("many-lines.txt", text));

      // Act
      const run = await runFired();

      // Assert
      const lines = labeledLines(run.report, "before");
      expect.soft(lines).toStrictEqual(Array.from({ length: 81 }, () => line));
      expect(run.record.stdoutBytes).toBe(20200);
    });

    it("stops mid-line without splitting a multi-byte character", async () => {
      // Arrange
      writeHookScript("before", catBody("one-long-line.txt", "é".repeat(5000)));

      // Act
      const run = await runFired();

      // Assert
      const lines = labeledLines(run.report, "before");
      expect.soft(lines).toStrictEqual(["é".repeat(STDOUT_LIMIT_BYTES / 2)]);
      expect(run.record.stdoutBytes).toBe(10000);
    });
  });

  describe("when the hook cannot steer the loop the way it meant to", () => {
    it("reports a non-zero exit alongside its stderr instead of failing the iterate", async () => {
      // Arrange
      writeHookScript("before", 'echo "checked the cache"\necho "no warm copy" >&2\nexit 3');

      // Act
      const run = await runFired();

      // Assert
      expect
        .soft(labeledLines(run.report, "before"))
        .toStrictEqual(["checked the cache", "hook exited 3", "no warm copy"]);
      expect(run.record).toStrictEqual({
        type: "hook",
        stage: "before",
        seq: 2,
        exitCode: 3,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        durationMs: expect.any(Number),
        stdoutBytes: 18,
        timedOut: false,
      });
    });

    it("kills a hook that outruns its timeout and says so", async () => {
      // Arrange
      writeHookScript("before", "exec sleep 5");

      // Act
      const run = await runFired({ timeoutMs: 200 });

      // Assert
      expect.soft(labeledLines(run.report, "before")).toStrictEqual(["hook timed out after 200ms"]);
      expect.soft(run.record.timedOut).toBe(true);
      expect.soft(run.record.exitCode).not.toBe(0);
      expect(run.record.durationMs).toBeLessThan(4000);
    });
  });
});
