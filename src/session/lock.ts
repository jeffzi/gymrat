import fs from "node:fs";
import path from "node:path";

import { GymratError, hasErrorCode } from "../errors.js";

/** The process a lockfile records as its holder. */
interface LockHolder {
  pid: number;
  command: string;
  at: string;
}

/** Gives up an acquired lock. Calling it more than once is harmless. */
export type ReleaseLock = () => void;

function isLockHolder(value: unknown): value is LockHolder {
  return (
    typeof value === "object" &&
    value !== null &&
    "pid" in value &&
    typeof value.pid === "number" &&
    "command" in value &&
    typeof value.command === "string" &&
    "at" in value &&
    typeof value.at === "string"
  );
}

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
  return isLockHolder(parsed) ? parsed : undefined;
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
      throw new GymratError(
        `Lock held by PID ${String(existing.pid)} (${existing.command}, started ${existing.at})`,
        `Another gymrat run is active in this repo. Wait for it to finish or remove ${lockPath} if it crashed.`,
      );
    }
    fs.writeFileSync(lockPath, record);
  }

  return () => {
    try {
      fs.unlinkSync(lockPath);
    } catch (error) {
      if (!hasErrorCode(error, "ENOENT")) {
        throw error;
      }
    }
  };
}
