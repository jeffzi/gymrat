import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { GymratError } from "../../src/errors.js";
import { acquireLock } from "../../src/session/lock.js";

/** Shape every lockfile assertion expects, with the wall-clock field left open. */
const HOLDER_RECORD = {
  pid: process.pid,
  command: "compare",
  // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
  at: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/),
};

/** The `at` value `writeLockfile` stamps every fixture lockfile with. */
const WRITTEN_LOCK_AT = "2026-01-01T00:00:00.000Z";

/** A lock path inside its own temp directory, so tests never share a lockfile. */
function freshLockPath(...segments: string[]): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lock-test-"));
  return path.join(dir, ...segments, "gymrat.lock.json");
}

/**
 * A pid that is certainly gone: the child ran to completion and was reaped
 * before `spawnSync` returned.
 */
function deadPid(): number {
  return spawnSync(process.execPath, ["-e", ""]).pid;
}

function writeLockfile(lockPath: string, holder: { pid: number; command: string }): void {
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  fs.writeFileSync(lockPath, JSON.stringify({ ...holder, at: WRITTEN_LOCK_AT }));
}

function readLockfile(lockPath: string): unknown {
  return JSON.parse(fs.readFileSync(lockPath, "utf8"));
}

/** Run `act` and hand back the GymratError it threw, failing the test if it threw none. */
function captureGymratError(act: () => unknown): GymratError {
  try {
    act();
  } catch (error) {
    if (error instanceof GymratError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected the call to throw a GymratError");
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("acquireLock", () => {
  describe("when no lockfile exists", () => {
    it("records the holding process, its command, and the start time", () => {
      // Arrange
      const lockPath = freshLockPath();

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual(HOLDER_RECORD);
    });

    it("creates the directories leading to the lockfile", () => {
      // Arrange
      const lockPath = freshLockPath("nested", "deeper");

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect(fs.existsSync(lockPath)).toBe(true);
    });
  });

  describe("when the lockfile is held by a live process", () => {
    it("throws a GymratError naming the holder and pointing at the lockfile", () => {
      // Arrange
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid: process.pid, command: "measure" });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain(`PID ${String(process.pid)}`);
      expect.soft(error.message).toContain("measure");
      expect.soft(error.hint).toMatch(/another gymrat run/i);
      expect.soft(error.hint).toContain(lockPath);
    });

    it("leaves the holder record of an unreadable liveness probe in place", () => {
      // Arrange: EPERM means the process exists but belongs to another user.
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid: deadPid(), command: "measure" });
      vi.spyOn(process, "kill").mockImplementation(() => {
        throw Object.assign(new Error("operation not permitted"), { code: "EPERM" });
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain("measure");
      expect.soft(readLockfile(lockPath)).toStrictEqual({
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        pid: expect.any(Number),
        command: "measure",
        at: WRITTEN_LOCK_AT,
      });
    });
  });

  describe("when the lockfile is held by a process that has exited", () => {
    it("steals the lock and overwrites the stale holder record", () => {
      // Arrange
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid: deadPid(), command: "measure" });

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual(HOLDER_RECORD);
    });
  });
});

describe("the release handle", () => {
  it("removes the lockfile", () => {
    // Arrange
    const lockPath = freshLockPath();
    const release = acquireLock(lockPath, "compare");

    // Act
    release();

    // Assert
    expect(fs.existsSync(lockPath)).toBe(false);
  });

  it("stays silent when the lockfile is already gone", () => {
    // Arrange
    const lockPath = freshLockPath();
    const release = acquireLock(lockPath, "compare");
    release();

    // Act & Assert
    expect(() => {
      release();
    }).not.toThrow();
  });
});
