import { execFileSync, type ChildProcess, type SpawnOptions } from "node:child_process";
import { getEventListeners } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";

import { describe, it, expect, beforeEach, afterEach, onTestFinished, vi } from "vitest";

import type { ExecOptions, ExecResult, ExecTimeoutError } from "../src/exec.js";
import { exec } from "../src/exec.js";
import { isAlive, readPid, waitForPid } from "./fixtures/process-probe.js";

/** Every child `exec` has spawned, so a test can reach into its stdio pipes. */
const spawnedChildren = vi.hoisted((): ChildProcess[] => []);

// `spawn` stays real — only the Windows-only `execFileSync("taskkill", ...)` is
// faked, because it is the call whose failure modes are under test.
vi.mock("node:child_process", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:child_process")>();
  return {
    ...actual,
    execFileSync: vi.fn(),
    spawn: (command: string, options: SpawnOptions): ChildProcess => {
      const child = actual.spawn(command, options);
      spawnedChildren.push(child);
      return child;
    },
  };
});

const PENDING = Symbol("pending");

function assertIsExecResult(result: ExecResult | ExecTimeoutError): asserts result is ExecResult {
  if ("kind" in result) {
    throw new Error(`Expected ExecResult, got timeout error after ${result.timeoutMs}ms`);
  }
}

function assertIsTimeoutError(
  result: ExecResult | ExecTimeoutError,
): asserts result is ExecTimeoutError {
  if (!("kind" in result)) {
    throw new Error("Expected timeout error, got success result");
  }
}

async function settleWithin<T>(promise: Promise<T>, ms: number): Promise<T | typeof PENDING> {
  return await Promise.race([promise, delay(ms, PENDING, { ref: false })]);
}

/** A bare shell writes its pid promptly; this only has to outlast process startup. */
const PID_WAIT_MS = 3000;

/** The most recent child `exec` spawned, once the spawn has happened. */
async function waitForSpawnedChild(): Promise<ChildProcess> {
  return await vi.waitFor(
    () => {
      const child = spawnedChildren.at(-1);
      if (child === undefined) {
        throw new Error("exec() has not spawned a child yet");
      }
      return child;
    },
    { timeout: PID_WAIT_MS, interval: 25 },
  );
}

/**
 * Reap the children of tests that deliberately stop `exec` from killing its own
 * process group. Signal only a group whose leader is still alive, so a pid the
 * OS has already recycled is never signalled.
 */
function killSpawnedChildren(): void {
  for (const { pid } of spawnedChildren) {
    if (pid !== undefined && isAlive(pid)) {
      try {
        process.kill(-pid, "SIGKILL");
      } catch {
        // Raced: the group exited between the liveness check and the kill.
      }
    }
  }
}

function taskkillFailure(status: number): Error {
  return Object.assign(new Error(`taskkill exited with ${status}`), { status });
}

// exec uses POSIX process groups (kill(-pid, SIGKILL)) and the tests drive
// sh-only constructs ($$, >&2, for/do/done). Neither works under cmd.exe.
describe.skipIf(process.platform === "win32")("exec", () => {
  const runInTmpdir = (command: string, options: Omit<ExecOptions, "cwd"> = {}) =>
    exec(command, { cwd: os.tmpdir(), ...options });

  describe("when the command runs to completion", () => {
    it.each([
      {
        description: "captures stdout",
        command: "echo hello",
        expected: { stdout: "hello\n", stderr: "", exitCode: 0 },
      },
      {
        description: "separates stdout from stderr",
        command: "echo stdout && echo stderr >&2",
        expected: { stdout: "stdout\n", stderr: "stderr\n", exitCode: 0 },
      },
      {
        description: "reports a non-zero exit code",
        command: "exit 42",
        expected: { stdout: "", stderr: "", exitCode: 42 },
      },
      {
        description: "captures every line of multi-line output",
        command: 'echo "line1" && echo "line2" && echo "line3"',
        expected: { stdout: "line1\nline2\nline3\n", stderr: "", exitCode: 0 },
      },
      {
        description: "waits for a slow command with no timeout set",
        command: "sleep 0.1 && echo done",
        expected: { stdout: "done\n", stderr: "", exitCode: 0 },
      },
    ])("$description", async ({ command, expected }) => {
      const result = await runInTmpdir(command);

      expect(result).toStrictEqual(expected);
    });
  });

  describe("when stdin is provided", () => {
    it("delivers the text to the command's standard input", async () => {
      const result = await runInTmpdir("cat", { stdin: "piped input\n" });

      expect(result).toStrictEqual({ stdout: "piped input\n", stderr: "", exitCode: 0 });
    });

    it("settles with the command's own result when the command never reads stdin", async () => {
      // Larger than any OS pipe buffer, so the write cannot complete on its own:
      // the child exits first and the pending write breaks with EPIPE.
      const unreadPayload = "x".repeat(1024 * 1024);

      const result = await runInTmpdir("exit 3", { stdin: unreadPayload });

      expect(result).toStrictEqual({ stdout: "", stderr: "", exitCode: 3 });
    });
  });

  describe("when stdin is omitted", () => {
    it("gives the command an immediately closed standard input", async () => {
      const result = await runInTmpdir("cat");

      expect(result).toStrictEqual({ stdout: "", stderr: "", exitCode: 0 });
    });
  });

  describe("when a descendant writes after the shell exits", () => {
    it("captures output flushed once the shell has already returned", async () => {
      const result = await runInTmpdir("(sleep 0.2; echo METRIC) &");

      expect(result).toStrictEqual({ stdout: "METRIC\n", stderr: "", exitCode: 0 });
    });
  });

  describe("when a multi-byte character is split across pipe reads", () => {
    it("decodes the halves as a single character", async () => {
      // The two bytes of U+00B5 are flushed by separate printf processes, so
      // they land in separate pipe reads.
      const result = await runInTmpdir("printf '\\302'; sleep 0.2; printf '\\265'");

      expect(result).toStrictEqual({ stdout: "µ", stderr: "", exitCode: 0 });
    });
  });

  describe("when the shell cannot be spawned", () => {
    it("reports the spawn failure on stderr", async () => {
      const missingDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-exec-gone-"));
      fs.rmSync(missingDir, { recursive: true });

      const result = await exec("echo hello", { cwd: missingDir });
      assertIsExecResult(result);

      expect(result.stdout).toBe("");
      expect(result.exitCode).not.toBe(0);
      expect(result.stderr).toContain("ENOENT");
    });

    it("resolves with an ExecResult when spawn throws synchronously", async () => {
      // A regular file as cwd fails the lookup before the child exists, so Node
      // throws out of spawn() instead of emitting an "error" event.
      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-exec-sync-"));
      onTestFinished(() => {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      });
      const filePath = path.join(tmpDir, "not-a-directory");
      fs.writeFileSync(filePath, "");

      const result = await exec("echo hello", { cwd: filePath });
      assertIsExecResult(result);

      expect(result.stdout).toBe("");
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain("ENOTDIR");
    });
  });

  describe("when working directory is specified", () => {
    it("runs the command in the given directory", async () => {
      const result = await runInTmpdir("pwd");
      assertIsExecResult(result);

      expect(result.exitCode).toBe(0);
      expect(result.stdout.trim()).toBe(fs.realpathSync(os.tmpdir()));
    });
  });

  describe("when timeout is exceeded", () => {
    it("returns a timeout error when the process exceeds the timeout", async () => {
      const result = await runInTmpdir("sleep 10", { timeoutMs: 500 });

      expect(result).toStrictEqual({
        kind: "timeout",
        stdout: "",
        stderr: "",
        timeoutMs: 500,
      });
    });

    it("captures partial output before timeout", async () => {
      const result = await runInTmpdir('for i in 1 2 3; do echo "line $i"; sleep 1; done', {
        timeoutMs: 1500,
      });
      assertIsTimeoutError(result);

      expect(result.kind).toBe("timeout");
      expect(result.stderr).toBe("");
      expect(result.timeoutMs).toBe(1500);
      expect(result.stdout).toContain("line 1");
    });
  });

  describe("when aborted", () => {
    let tmpDir: string;

    beforeEach(() => {
      tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-exec-"));
    });

    afterEach(() => {
      // Safety net: a test that fails before the abort lands must not leak a process group.
      let leader = Number.NaN;
      try {
        leader = readPid(path.join(tmpDir, "shell.pid"));
      } catch {
        leader = Number.NaN;
      }
      // Signal only a group whose leader is still alive. On the passing path the
      // group is already dead and the kill would exist purely to raise ESRCH into
      // an empty catch — but if the OS has recycled that pid, it would instead
      // signal a group this test does not own.
      if (leader > 0 && isAlive(leader)) {
        try {
          process.kill(-leader, "SIGKILL");
        } catch {
          // Raced: the group exited between the liveness check and the kill.
        }
      }
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    describe("while the command is running", () => {
      it("kills the whole process group, leaving no grandchild running", async () => {
        const controller = new AbortController();
        const running = exec("sleep 30 & echo $! > grandchild.pid; echo $$ > shell.pid; wait", {
          cwd: tmpDir,
          signal: controller.signal,
        });
        const grandchildPid = await waitForPid(path.join(tmpDir, "grandchild.pid"), PID_WAIT_MS);

        controller.abort();

        await vi.waitFor(
          () => {
            expect(isAlive(grandchildPid)).toBe(false);
          },
          { timeout: 3000, interval: 25 },
        );
        await running;
      }, 20_000);

      it("settles the promise with a result that is not a timeout", async () => {
        const controller = new AbortController();
        const running = exec("echo $$ > shell.pid; sleep 30", {
          cwd: tmpDir,
          signal: controller.signal,
        });
        await waitForPid(path.join(tmpDir, "shell.pid"), PID_WAIT_MS);

        controller.abort();

        const settled = await settleWithin(running, 3000);
        if (settled === PENDING) {
          throw new Error("exec() stayed pending after abort");
        }
        // Fully determined: the command redirects its only output to a file, and a
        // SIGKILLed child reports a null code that exec maps to 1. An exact match
        // also rules out the timeout shape, which carries `kind` and `timeoutMs`.
        expect(settled).toStrictEqual({ stdout: "", stderr: "", exitCode: 1 });
      }, 20_000);
    });

    describe("before the call", () => {
      beforeEach(() => {
        spawnedChildren.length = 0;
      });

      it("never spawns the shell, settling as a cancelled run", async () => {
        const controller = new AbortController();
        controller.abort();

        const settled = await settleWithin(
          exec("echo $$ > shell.pid; sleep 30; echo done > completed.marker", {
            cwd: tmpDir,
            signal: controller.signal,
          }),
          3000,
        );

        if (settled === PENDING) {
          throw new Error("exec() stayed pending for an already-aborted signal");
        }
        expect(spawnedChildren).toHaveLength(0);
        // Same shape as a mid-run abort: no output, and exec's failure exit code.
        // An exact match also rules out the timeout shape, which carries `kind`
        // and `timeoutMs`.
        expect(settled).toStrictEqual({ stdout: "", stderr: "", exitCode: 1 });
        expect(fs.existsSync(path.join(tmpDir, "completed.marker"))).toBe(false);
      }, 20_000);
    });
  });

  describe("when aborted after the command completed", () => {
    it("leaves no abort listener on the caller's signal", async () => {
      const controller = new AbortController();
      await runInTmpdir("echo done", { signal: controller.signal });

      expect(getEventListeners(controller.signal, "abort")).toHaveLength(0);
    });
  });

  describe("when a run settles while the child's pipes are still open", () => {
    /** A run under way, plus the trigger that settles it early. */
    interface StartedRun {
      readonly running: Promise<ExecResult | ExecTimeoutError>;
      /** Absent for a run that settles on its own, such as one that times out. */
      readonly settleRun?: () => void;
    }

    beforeEach(() => {
      spawnedChildren.length = 0;
    });

    afterEach(() => {
      killSpawnedChildren();
    });

    it.each([
      {
        description: "a timeout",
        start: (): StartedRun => ({ running: runInTmpdir("sleep 30", { timeoutMs: 500 }) }),
      },
      {
        description: "an abort",
        start: (): StartedRun => {
          const controller = new AbortController();
          return {
            running: runInTmpdir("sleep 30", { signal: controller.signal }),
            settleRun: () => {
              controller.abort();
            },
          };
        },
      },
    ])(
      "destroys the child's stdio pipes after $description, so a surviving descendant cannot grow the settled buffers",
      async ({ start }) => {
        const { running, settleRun } = start();
        const child = await waitForSpawnedChild();
        const { stdout, stderr } = child;
        if (stdout === null || stderr === null) {
          throw new Error("child was spawned without stdio pipes");
        }

        settleRun?.();
        await running;

        expect({ stdout: stdout.destroyed, stderr: stderr.destroyed }).toStrictEqual({
          stdout: true,
          stderr: true,
        });
      },
      20_000,
    );
  });

  describe("when taskkill fails on Windows", () => {
    const realPlatform = process.platform;

    const setPlatform = (value: NodeJS.Platform) => {
      Object.defineProperty(process, "platform", { value, configurable: true });
    };

    /**
     * Drives one abort through the Windows kill path with a taskkill that fails.
     *
     * `process.platform` is stubbed after the spawn, not before: `spawn` reads it
     * too, and a win32 platform at spawn time would send Node looking for
     * cmd.exe instead of the sh this suite needs. `killTree` reads it later, when
     * the abort lands, which is the only read this test wants to redirect.
     */
    const abortWithFailingTaskkill = async (failure: Error) => {
      vi.mocked(execFileSync).mockImplementation(() => {
        throw failure;
      });
      const controller = new AbortController();
      const running = runInTmpdir("sleep 0.5", { signal: controller.signal });
      await waitForSpawnedChild();

      setPlatform("win32");
      controller.abort();
      await running;
    };

    beforeEach(() => {
      spawnedChildren.length = 0;
    });

    afterEach(() => {
      setPlatform(realPlatform);
      killSpawnedChildren();
      vi.restoreAllMocks();
    });

    it("stays silent when taskkill reports that the process is gone", async () => {
      const emitWarning = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
      const failure = taskkillFailure(128);

      await abortWithFailingTaskkill(failure);

      expect(emitWarning).not.toHaveBeenCalledWith(failure);
    });

    it("warns when taskkill fails for any other reason", async () => {
      const emitWarning = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
      const failure = taskkillFailure(5);

      await abortWithFailingTaskkill(failure);

      expect(emitWarning).toHaveBeenCalledWith(failure);
    });
  });

  describe("when a stdio stream emits an error", () => {
    beforeEach(() => {
      spawnedChildren.length = 0;
    });

    afterEach(() => {
      killSpawnedChildren();
    });

    it.each([{ stream: "stdout" as const }, { stream: "stderr" as const }])(
      "settles as a failed run when $stream errors",
      async ({ stream }) => {
        const running = runInTmpdir("sleep 0.5");
        const child = await waitForSpawnedChild();
        const pipe = stream === "stdout" ? child.stdout : child.stderr;
        if (pipe === null) {
          throw new Error(`child was spawned without a ${stream} pipe`);
        }

        pipe.emit("error", new Error("stream exploded"));

        const settled = await settleWithin(running, 3000);
        expect(settled).toStrictEqual({
          stdout: "",
          stderr: "stream exploded\n",
          exitCode: 1,
        });
      },
    );

    it("kills the whole process group, leaving no grandchild running", async () => {
      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-exec-stream-"));
      onTestFinished(() => {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      });
      const running = exec("sleep 30 & echo $! > grandchild.pid; wait", { cwd: tmpDir });
      const child = await waitForSpawnedChild();
      if (child.stdout === null) {
        throw new Error("child was spawned without a stdout pipe");
      }
      const grandchildPid = await waitForPid(path.join(tmpDir, "grandchild.pid"), PID_WAIT_MS);

      child.stdout.emit("error", new Error("stream exploded"));

      await vi.waitFor(
        () => {
          expect(isAlive(grandchildPid)).toBe(false);
        },
        { timeout: 3000, interval: 25 },
      );
      await running;
    }, 20_000);
  });
});
