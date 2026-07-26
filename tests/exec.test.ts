import fs from "node:fs";
import os from "node:os";

import { describe, it, expect } from "vitest";

import type { ExecResult, ExecTimeoutError } from "../src/exec.js";
import { exec } from "../src/exec.js";

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

describe("exec", () => {
  const runInTmpdir = (command: string, options: { timeoutMs?: number } = {}) =>
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
});
