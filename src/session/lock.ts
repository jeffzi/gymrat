import fs from "node:fs";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";

import { GymratError, hasErrorCode } from "../errors.js";
import { compile, expected } from "../schema.js";

const lockHolderSchema = Type.Object(
  {
    // The compiled check requires a finite number, so a `pid` that overflowed
    // JSON's number range reads as a foreign shape rather than as a holder no
    // signal can ever reach.
    pid: Type.Number(expected("a number")),
    command: Type.String(expected("a string")),
    at: Type.String(expected("a string")),
  },
  // Deliberately not strict about unknown keys: a lockfile carrying a field
  // this version does not know is still a live holder, and rejecting it would
  // steal the lock out from under whoever wrote it.
  expected("an object"),
);

/** The process a lockfile records as its holder. */
type LockHolder = Static<typeof lockHolderSchema>;

const lockHolderValidator = compile(lockHolderSchema);

/** Gives up an acquired lock. Calling it more than once is harmless. */
export type ReleaseLock = () => void;

/**
 * Read the holder a lockfile names, or `undefined` when it names nobody usable.
 *
 * A file that has vanished, cannot be parsed, or carries a foreign shape is
 * treated the same as one whose holder has exited: a lock nobody can be shown to
 * hold must not wedge the repository forever.
 */
function readHolder(lockPath: string): LockHolder | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(lockPath, "utf8"));
  } catch {
    return undefined;
  }
  return lockHolderValidator.check(parsed) ? parsed : undefined;
}

/**
 * Whether a process with `pid` still exists.
 *
 * Signal `0` runs the kernel's permission and existence checks without
 * delivering anything. Only `ESRCH` means no such process: `EPERM` says the
 * process is there but owned by another user, which is still a live holder.
 */
function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !hasErrorCode(error, "ESRCH");
  }
}

function removeLockFile(lockPath: string): void {
  try {
    fs.unlinkSync(lockPath);
  } catch (error) {
    if (!hasErrorCode(error, "ENOENT")) {
      throw error;
    }
  }
}

function heldByError(holder: LockHolder, lockPath: string): GymratError {
  return new GymratError(
    `Lock held by PID ${String(holder.pid)} (${holder.command}, started ${holder.at})`,
    `Another gymrat run is active in this repo. Wait for it to finish or remove ${lockPath} if it crashed.`,
  );
}

function stealLock(lockPath: string, record: string): void {
  removeLockFile(lockPath);
  try {
    fs.writeFileSync(lockPath, record, { flag: "wx" });
  } catch (retryError) {
    if (!hasErrorCode(retryError, "EEXIST")) {
      throw retryError;
    }
    const winner = readHolder(lockPath);
    throw winner === undefined
      ? new GymratError(
          `Lock at ${lockPath} was claimed by another process while stealing it.`,
          `Another gymrat run is active in this repo. Wait for it to finish or remove ${lockPath} if it crashed.`,
        )
      : heldByError(winner, lockPath);
  }
}

/**
 * Take the single-flight lock at `lockPath` on behalf of `command`.
 *
 * The lockfile is created exclusively, so two processes racing for it cannot
 * both win. A lockfile left behind by a process that has since exited is stolen
 * silently — a crashed run must not need manual cleanup.
 *
 * @throws GymratError when the lock is held by a process that is still running.
 */
export function acquireLock(lockPath: string, command: string): ReleaseLock {
  const holder: LockHolder = { pid: process.pid, command, at: new Date().toISOString() };
  const record = JSON.stringify(holder);

  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  try {
    fs.writeFileSync(lockPath, record, { flag: "wx" });
  } catch (error) {
    if (!hasErrorCode(error, "EEXIST")) {
      throw error;
    }
    const existing = readHolder(lockPath);
    if (existing !== undefined && isAlive(existing.pid)) {
      throw heldByError(existing, lockPath);
    }
    stealLock(lockPath, record);
  }

  return () => {
    removeLockFile(lockPath);
  };
}
