import fs from "node:fs";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";

import { assertNever, GymratError, hasErrorCode, messageOf } from "../errors.js";
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

/** How many times acquisition re-reads a lockfile it lost a race for. */
const MAX_ACQUIRE_ATTEMPTS = 3;

/** What a lockfile says at the moment it was read. */
type LockfileState =
  | { readonly kind: "absent" }
  | { readonly kind: "held"; readonly holder: LockHolder }
  | { readonly kind: "unreadable" };

/**
 * Read what the lockfile at `lockPath` currently says.
 *
 * A file that cannot be parsed, or that carries a foreign shape, is reported as
 * `unreadable`. Publication is atomic, so no reader ever catches a holder
 * mid-write: an unreadable file is debris — a run killed between writing its
 * record and publishing it, or a foreign file at the lock path — not a lock
 * somebody is in the middle of taking.
 */
function readLockfile(lockPath: string): LockfileState {
  let contents: string;
  try {
    contents = fs.readFileSync(lockPath, "utf8");
  } catch (error) {
    if (hasErrorCode(error, "ENOENT")) {
      return { kind: "absent" };
    }
    throw error;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(contents);
  } catch {
    return { kind: "unreadable" };
  }
  return lockHolderValidator.check(parsed)
    ? { kind: "held", holder: parsed }
    : { kind: "unreadable" };
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

/**
 * Publish `record` at `lockPath`, or report that someone else got there first.
 *
 * The record is written to a scratch file beside the lock and only then linked
 * into place, so the lock path never exists holding half a record: readers see a
 * whole holder or nothing at all. The link is also the exclusive step — it fails
 * with `EEXIST` when the path is taken — so exactly one racer publishes.
 */
function publishLockRecord(lockPath: string, record: string): boolean {
  const scratchPath = `${lockPath}.${String(process.pid)}.record`;
  fs.writeFileSync(scratchPath, record);
  try {
    fs.linkSync(scratchPath, lockPath);
    return true;
  } catch (error) {
    if (hasErrorCode(error, "EEXIST")) {
      return false;
    }
    throw error;
  } finally {
    removeLockFile(scratchPath);
  }
}

/**
 * Move the stale lockfile aside, handing back the path it now lives at.
 *
 * The rename is the atomic step of a steal: exactly one racer can move a given
 * file, so the ones that lose get `ENOENT` and are told so with `undefined`
 * rather than going on to overwrite the winner's fresh lock.
 *
 * @throws GymratError when the file belongs to another user (a sticky `/tmp`),
 *   which no retry can resolve.
 */
function claimStaleLock(lockPath: string): string | undefined {
  const claimPath = `${lockPath}.${String(process.pid)}.claim`;
  try {
    fs.renameSync(lockPath, claimPath);
  } catch (error) {
    if (hasErrorCode(error, "ENOENT")) {
      return undefined;
    }
    if (hasErrorCode(error, "EPERM") || hasErrorCode(error, "EACCES")) {
      throw new GymratError(
        `Stale lock file ${lockPath} could not be removed: ${messageOf(error)}`,
        `It belongs to another user. Remove ${lockPath} yourself, then rerun.`,
        { cause: error },
      );
    }
    throw error;
  }
  return claimPath;
}

/** Take over a lockfile no live process holds. Returns whether this process won. */
function stealLock(lockPath: string, record: string): boolean {
  const claimPath = claimStaleLock(lockPath);
  if (claimPath === undefined) {
    return false;
  }
  try {
    return publishLockRecord(lockPath, record);
  } finally {
    removeLockFile(claimPath);
  }
}

/**
 * Take the single-flight lock at `lockPath` on behalf of `command`.
 *
 * The lockfile is published exclusively, so two processes racing for it cannot
 * both win. A lockfile no live process holds is stolen silently — a crashed run
 * must not need manual cleanup, whether it left a holder record behind or a file
 * too damaged to read — and losing that steal re-enters acquisition, where the
 * winner is either a live holder to report or a lock released again in the
 * meantime.
 *
 * @throws GymratError when the lock is held by a process that is still running,
 *   or when a lockfile free to steal belongs to another user.
 */
export function acquireLock(lockPath: string, command: string): ReleaseLock {
  const holder: LockHolder = { pid: process.pid, command, at: new Date().toISOString() };
  const record = JSON.stringify(holder);
  const release: ReleaseLock = () => {
    removeLockFile(lockPath);
  };

  fs.mkdirSync(path.dirname(lockPath), { recursive: true });

  for (let attempt = 0; attempt < MAX_ACQUIRE_ATTEMPTS; attempt++) {
    if (publishLockRecord(lockPath, record)) {
      return release;
    }

    const state = readLockfile(lockPath);
    switch (state.kind) {
      case "absent":
        continue;
      case "unreadable":
        if (stealLock(lockPath, record)) {
          return release;
        }
        continue;
      case "held":
        if (isAlive(state.holder.pid)) {
          throw heldByError(state.holder, lockPath);
        }
        if (stealLock(lockPath, record)) {
          return release;
        }
        continue;
      default:
        return assertNever(state);
    }
  }

  throw new GymratError(
    `Lock at ${lockPath} was claimed by another process on every attempt.`,
    `Another gymrat run is active in this repo. Wait for it to finish or remove ${lockPath} if it crashed.`,
  );
}
