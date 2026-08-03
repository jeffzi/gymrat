import { getEventListeners } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import type { ExecOptions, ExecResult, ExecTimeoutError } from "../src/exec.js";
import { exec } from "../src/exec.js";
import { isAlive, readPid, waitForPid } from "./fixtures/process-probe.js";

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

describe("exec", () => {
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
    it("kills the process and returns timeout error", async () => {
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
      it("kills the command instead of running it, settling as a cancelled run", async () => {
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
        // A `sleep 30` that settles inside three seconds can only have been
        // signalled, so this doubles as the proof the command was killed rather
        // than left running. Same shape as a mid-run abort: no output, and a
        // SIGKILLed child reports a null code that exec maps to 1. An exact match
        // also rules out the timeout shape, which carries `kind` and `timeoutMs`.
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
});
