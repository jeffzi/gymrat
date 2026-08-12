import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { acquireLock } from "../../src/session/lock.js";
import { captureGymratError, captureThrown } from "../fixtures/errors.js";

/** Shape every lockfile assertion expects, with the wall-clock field left open. */
const HOLDER_RECORD = {
  pid: process.pid,
  command: "compare",
  // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
  at: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/),
};

/** The `at` value `writeLockfile` stamps every fixture lockfile with. */
const WRITTEN_LOCK_AT = "2026-01-01T00:00:00.000Z";

/**
 * The remedy for a lock whose holder answered a liveness probe.
 *
 * Deleting the lockfile is not offered: the holder is provably alive, so the
 * only correct move is to wait for it.
 */
const LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish.";

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

/** An error carrying the errno `code` a syscall mock needs to report. */
function errnoError(code: string, message: string): Error {
  return Object.assign(new Error(message), { code });
}

/** Make every open of `lockPath` fail with `failure`, leaving other paths alone. */
function refuseOpen(lockPath: string, failure: Error): void {
  const realOpenSync = fs.openSync.bind(fs);
  vi.spyOn(fs, "openSync").mockImplementation((file, flags, mode) => {
    if (file === lockPath) {
      throw failure;
    }
    return realOpenSync(file, flags, mode);
  });
}

/** A fresh lockfile whose holder process, `pid`, has already exited. */
function staleLockPath(pid = deadPid()): { lockPath: string; holderPid: number } {
  const lockPath = freshLockPath();
  writeLockfile(lockPath, { pid, command: "measure" });
  return { lockPath, holderPid: pid };
}

/** The real `fs.linkSync`, captured so wedging works under a `linkSync` spy. */
const realLinkSync = fs.linkSync.bind(fs);

/**
 * Leave behind the claim link a run that died mid-takeover would have left.
 *
 * The link is the lockfile's own inode under the claim name derived from that
 * inode, so every steal attempt is blocked by an occupied claim path while the
 * identity behind the lockfile stays constant. Hands back the claim path.
 */
function wedgeTakeover(lockPath: string): string {
  const { dev, ino } = fs.statSync(lockPath, { bigint: true });
  const claimPath = `${lockPath}.${String(dev)}-${String(ino)}.claim`;
  realLinkSync(lockPath, claimPath);
  return claimPath;
}

/**
 * Put a different run's lockfile where `lockPath` is, as a takeover would.
 *
 * The replacement is built beside the lock and renamed over it, so the
 * filesystem cannot hand it the inode of the file it displaces — the two
 * lockfiles are guaranteed to be distinguishable.
 */
function replaceLockfile(lockPath: string, holder: { pid: number; command: string }): void {
  const scratchPath = `${lockPath}.replacement`;
  writeLockfile(scratchPath, holder);
  fs.renameSync(scratchPath, lockPath);
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
    it("throws a GymratError naming the holder and telling the caller to wait it out", () => {
      // Arrange
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid: process.pid, command: "measure" });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain(`PID ${String(process.pid)}`);
      expect.soft(error.message).toContain("measure");
      expect.soft(error.hint).toBe(LIVE_HOLDER_HINT);
    });

    it("leaves the holder record of an unreadable liveness probe in place", () => {
      // Arrange: EPERM means the process exists but belongs to another user.
      const { lockPath } = staleLockPath();
      vi.spyOn(process, "kill").mockImplementation(() => {
        throw errnoError("EPERM", "operation not permitted");
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
      const { lockPath } = staleLockPath();

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

  describe("when the lockfile names an impossible process", () => {
    it.each([
      { shape: "the whole process group", pid: 0 },
      { shape: "every process at once", pid: -1 },
      { shape: "a fraction of a process", pid: 3.5 },
      { shape: "a number no pid can reach", pid: 4_294_967_296 },
    ])("reclaims a lockfile whose holder is $shape", ({ pid }) => {
      // Arrange: no run can be identified by this pid, so the record is damaged
      // exactly as a truncated one is — and probing it would answer for
      // something other than the holder.
      const lockPath = freshLockPath();
      writeLockfile(lockPath, { pid, command: "measure" });

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual(HOLDER_RECORD);
    });
  });

  describe("when the lockfile cannot be opened", () => {
    it.each([{ code: "EACCES" }, { code: "EPERM" }])(
      "names the lockfile and the manual remedy instead of raising $code",
      ({ code }) => {
        // Arrange: the lockfile is another user's, readable by them alone, so
        // gymrat cannot even learn whose run holds it.
        const { lockPath } = staleLockPath();
        refuseOpen(lockPath, errnoError(code, "permission denied"));

        // Act
        const error = captureGymratError(() => acquireLock(lockPath, "compare"));

        // Assert
        expect.soft(error.message).toContain(lockPath);
        expect.soft(error.hint).toMatch(/remove/i);
        expect.soft(error.hint).toContain(lockPath);
      },
    );

    it("lets an unexpected open failure reach the caller unwrapped", () => {
      // Arrange
      const { lockPath } = staleLockPath();
      const failure = errnoError("EIO", "input/output error");
      refuseOpen(lockPath, failure);

      // Act & Assert
      expect(captureThrown(() => acquireLock(lockPath, "compare"))).toBe(failure);
    });
  });

  describe("when the stale lockfile belongs to another user", () => {
    it("names the lockfile and the manual remedy instead of raising EPERM", () => {
      // Arrange: a sticky /tmp — the stale file is another user's, so gymrat
      // cannot claim it out of the way.
      const { lockPath } = staleLockPath();
      vi.spyOn(fs, "renameSync").mockImplementation(() => {
        throw errnoError("EPERM", "operation not permitted");
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
      const { lockPath } = staleLockPath();
      const rivalPid = process.pid;

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
      expect.soft(error.hint).toBe(LIVE_HOLDER_HINT);
      expect.soft(readLockfile(lockPath)).toStrictEqual({
        pid: rivalPid,
        command: "rival",
        at: WRITTEN_LOCK_AT,
      });
    });

    it("acquires the lock once the winner releases it", () => {
      // Arrange: a rival claims the stale file first — our claim finds it gone —
      // and has released the lock by the time we retry.
      const { lockPath } = staleLockPath();

      const realUnlink = fs.unlinkSync.bind(fs);
      vi.spyOn(fs, "renameSync").mockImplementationOnce((from) => {
        realUnlink(from);
        throw errnoError("ENOENT", "no such file or directory");
      });

      // Act
      acquireLock(lockPath, "compare");

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual(HOLDER_RECORD);
    });
  });

  describe("when a rival reaches the same staleness verdict first", () => {
    it("leaves the rival's fresh lockfile in place and reports it as held", () => {
      // Arrange: a rival run takes the very same stale lockfile over, and holds
      // it, in the window between our read of that lockfile and our own steal.
      const { lockPath } = staleLockPath();

      const realKill = process.kill.bind(process);
      let rivalHasRun = false;
      vi.spyOn(process, "kill").mockImplementation((pid, signal) => {
        if (!rivalHasRun) {
          rivalHasRun = true;
          acquireLock(lockPath, "rival");
        }
        return realKill(pid, signal);
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect.soft(error.message).toContain(`PID ${String(process.pid)}`);
      expect.soft(error.message).toContain("rival");
      expect.soft(readLockfile(lockPath)).toStrictEqual({ ...HOLDER_RECORD, command: "rival" });
    });
  });

  describe("when a takeover died and left its claim behind", () => {
    it("names the dead holder and both paths to delete", () => {
      // Arrange
      const { lockPath, holderPid } = staleLockPath();
      const claimPath = wedgeTakeover(lockPath);

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect
        .soft(error.message)
        .toBe(`Lock at ${lockPath} was left behind by a run that died while taking it over.`);
      expect
        .soft(error.hint)
        .toBe(
          `No gymrat process holds this lock (PID ${String(holderPid)} is dead). ` +
            `To unblock, delete ${lockPath} and ${claimPath}, then rerun.`,
        );
    });

    it("leaves the holder unnamed when the lockfile cannot be read", () => {
      // Arrange: a truncated record names no process, so the remedy cannot
      // either.
      const lockPath = freshLockPath();
      fs.mkdirSync(path.dirname(lockPath), { recursive: true });
      fs.writeFileSync(lockPath, '{"pid":4242,"comm');
      const claimPath = wedgeTakeover(lockPath);

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect
        .soft(error.message)
        .toBe(`Lock at ${lockPath} was left behind by a run that died while taking it over.`);
      expect
        .soft(error.hint)
        .toBe(
          `No gymrat process holds this lock. ` +
            `To unblock, delete ${lockPath} and ${claimPath}, then rerun.`,
        );
    });

    it("reports contention when an attempt failed for another reason", () => {
      // Arrange: the first steal fails because the lockfile itself went missing
      // — not because a leftover claim blocked it — so the run is contended
      // rather than wedged.
      const { lockPath } = staleLockPath();
      const claimPath = wedgeTakeover(lockPath);
      const realLink = fs.linkSync.bind(fs);
      let claimRefused = false;
      vi.spyOn(fs, "linkSync").mockImplementation((existing, target) => {
        if (target === claimPath && !claimRefused) {
          claimRefused = true;
          throw errnoError("ENOENT", "no such file or directory");
        }
        realLink(existing, target);
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect
        .soft(error.message)
        .toBe(`Lock at ${lockPath} was claimed by another process on every attempt.`);
      expect
        .soft(error.hint)
        .toBe(`${LIVE_HOLDER_HINT} If no gymrat process is running, delete ${lockPath}.`);
    });

    it("reports contention when the lockfile it wedged on is no longer the one on disk", () => {
      // Arrange: the leftover claim wedges every steal, and a rival publishes
      // its own lockfile as the last claim is refused — so the file left at the
      // lock path is not the one the attempts were wedged on, and no remedy may
      // name it for deletion. Which refusal is the last one is not knowable from
      // here, so the rival publishes after every refusal, and the wedged
      // lockfile — still reachable through its own claim link — goes back
      // whenever a further attempt begins.
      const { lockPath, holderPid } = staleLockPath();
      const claimPath = wedgeTakeover(lockPath);
      const restoreWedgedLockfile = (): void => {
        const scratchPath = `${lockPath}.restored`;
        realLinkSync(claimPath, scratchPath);
        fs.renameSync(scratchPath, lockPath);
        fs.rmSync(scratchPath, { force: true });
      };
      vi.spyOn(fs, "linkSync").mockImplementation((existing, target) => {
        if (target === lockPath) {
          restoreWedgedLockfile();
        }
        try {
          realLinkSync(existing, target);
        } finally {
          if (target === claimPath) {
            replaceLockfile(lockPath, { pid: holderPid, command: "rival" });
          }
        }
      });

      // Act
      const error = captureGymratError(() => acquireLock(lockPath, "compare"));

      // Assert
      expect
        .soft(error.message)
        .toBe(`Lock at ${lockPath} was claimed by another process on every attempt.`);
      expect
        .soft(error.hint)
        .toBe(`${LIVE_HOLDER_HINT} If no gymrat process is running, delete ${lockPath}.`);
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

  describe("when another run has replaced the lockfile", () => {
    it("leaves the replacement in place, however often it is called", () => {
      // Arrange: our lock was taken over, so the file at the lock path is now
      // the new holder's and deleting it would unlock a live run.
      const lockPath = freshLockPath();
      const release = acquireLock(lockPath, "compare");
      replaceLockfile(lockPath, { pid: process.pid, command: "rival" });

      // Act
      release();
      release();

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual({
        pid: process.pid,
        command: "rival",
        at: WRITTEN_LOCK_AT,
      });
    });

    it("leaves the replacement in place when the released lock was stolen", () => {
      // Arrange: the same takeover, but our own lock came from stealing a stale
      // lockfile rather than from publishing a fresh one.
      const { lockPath } = staleLockPath();
      const release = acquireLock(lockPath, "compare");
      replaceLockfile(lockPath, { pid: process.pid, command: "rival" });

      // Act
      release();

      // Assert
      expect(readLockfile(lockPath)).toStrictEqual({
        pid: process.pid,
        command: "rival",
        at: WRITTEN_LOCK_AT,
      });
    });
  });
});
