import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { acquireLock } from "../../src/session/lock.js";
import { captureGymratError } from "../fixtures/errors.js";

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

    it("publishes the record only once it is whole, leaving no scratch file behind", () => {
      // Arrange: a rival reader peeks at the lock path every time a record is
      // written out, so a half-written lockfile would be caught in the act.
      const lockPath = freshLockPath();
      const realWriteFileSync = fs.writeFileSync.bind(fs);
      const peeks: (string | undefined)[] = [];
      vi.spyOn(fs, "writeFileSync").mockImplementation((file, data, options) => {
        realWriteFileSync(file, data, options);
        peeks.push(fs.existsSync(lockPath) ? fs.readFileSync(lockPath, "utf8") : undefined);
      });

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect.soft(peeks).toStrictEqual([undefined]);
      expect.soft(fs.readdirSync(path.dirname(lockPath))).toStrictEqual([path.basename(lockPath)]);
      expect.soft(readLockfile(lockPath)).toStrictEqual(HOLDER_RECORD);
    });
  });

  describe("when a rival publishes first", () => {
    it("leaves the rival's lockfile in place and reports it as held", () => {
      // Arrange: the rival's lockfile lands after our record is written but
      // before it is published into place.
      const lockPath = freshLockPath();
      const rivalPid = process.pid;
      const realWriteFileSync = fs.writeFileSync.bind(fs);
      vi.spyOn(fs, "writeFileSync").mockImplementationOnce((file, data, options) => {
        realWriteFileSync(file, data, options);
        writeLockfile(lockPath, { pid: rivalPid, command: "rival" });
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain(`PID ${String(rivalPid)}`);
      expect.soft(error.message).toContain("rival");
      expect.soft(readLockfile(lockPath)).toStrictEqual({
        pid: rivalPid,
        command: "rival",
        at: WRITTEN_LOCK_AT,
      });
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

  describe("when the lockfile cannot be read", () => {
    it.each([
      { shape: "an empty", contents: "" },
      { shape: "a truncated", contents: '{"pid":4242,"comm' },
    ])("reclaims $shape lockfile no run could have published", ({ contents }) => {
      // Arrange: a whole record is what a reader sees, so an incomplete one is
      // the leftover of a run that died mid-write.
      const lockPath = freshLockPath();
      fs.mkdirSync(path.dirname(lockPath), { recursive: true });
      fs.writeFileSync(lockPath, contents);

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual(HOLDER_RECORD);
    });
  });

  describe("when the stale lockfile belongs to another user", () => {
    it("names the lockfile and the manual remedy instead of raising EPERM", () => {
      // Arrange: a sticky /tmp — the stale file is another user's, so gymrat
      // cannot claim it out of the way.
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid: deadPid(), command: "measure" });
      vi.spyOn(fs, "renameSync").mockImplementation(() => {
        throw Object.assign(new Error("operation not permitted"), { code: "EPERM" });
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain(lockPath);
      expect.soft(error.hint).toMatch(/remove/i);
      expect.soft(error.hint).toContain(lockPath);
    });
  });

  describe("when the steal race is lost", () => {
    it("leaves the winner's lockfile in place and reports it as held", () => {
      // Arrange: a rival takes the lock between our claim of the stale file and
      // our exclusive re-creation of it.
      const lockPath = freshLockPath();
      const rivalPid = process.pid;
      writeLockfile(lockPath, { pid: deadPid(), command: "measure" });

      const realRename = fs.renameSync.bind(fs);
      vi.spyOn(fs, "renameSync").mockImplementationOnce((from, to) => {
        realRename(from, to);
        writeLockfile(lockPath, { pid: rivalPid, command: "rival" });
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain(`PID ${String(rivalPid)}`);
      expect.soft(error.message).toContain("rival");
      expect.soft(error.hint).toContain(lockPath);
      expect.soft(readLockfile(lockPath)).toStrictEqual({
        pid: rivalPid,
        command: "rival",
        at: WRITTEN_LOCK_AT,
      });
    });

    it("acquires the lock once the winner releases it", () => {
      // Arrange: a rival claims the stale file first — our claim finds it gone —
      // and has released the lock by the time we retry.
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid: deadPid(), command: "measure" });

      const realUnlink = fs.unlinkSync.bind(fs);
      vi.spyOn(fs, "renameSync").mockImplementationOnce((from) => {
        realUnlink(from);
        throw Object.assign(new Error("no such file or directory"), { code: "ENOENT" });
      });

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
