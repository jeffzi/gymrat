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

/**
 * Gives up an acquired lock. Calling it more than once is harmless.
 *
 * Only the lockfile this run published is removed. A lock taken over since
 * belongs to whoever holds it now, and is left where it stands.
 */
export type ReleaseLock = () => void;

/** How many times acquisition re-reads a lockfile it lost a race for. */
const MAX_ACQUIRE_ATTEMPTS = 3;

/**
 * Which file a lockfile read came from, as the filesystem identifies it.
 *
 * A path says nothing about which file it names from one moment to the next, so
 * a steal that only knows `lockPath` cannot tell the record it judged stale from
 * whatever a rival published there since. Device and inode do tell them apart.
 */
type LockfileIdentity = {
  readonly dev: bigint;
  readonly ino: bigint;
};

/** Whether two reads came from one and the same file. */
function sameFile(one: LockfileIdentity, other: LockfileIdentity): boolean {
  return one.dev === other.dev && one.ino === other.ino;
}

/** Which file an open descriptor refers to, whatever its path names by now. */
function identityOf(fd: number): LockfileIdentity {
  const stats = fs.fstatSync(fd, { bigint: true });
  return { dev: stats.dev, ino: stats.ino };
}

/** What a lockfile says at the moment it was read. */
type LockfileState =
  | { readonly kind: "absent" }
  | { readonly kind: "held"; readonly holder: LockHolder; readonly identity: LockfileIdentity }
  | { readonly kind: "unreadable"; readonly identity: LockfileIdentity };

/**
 * Read what the lockfile at `lockPath` currently says, and which file said it.
 *
 * The identity is taken from the open descriptor the contents are read through,
 * so the two describe one file even when the lock path is taken over mid-read.
 *
 * A file that cannot be parsed, or that carries a foreign shape, is reported as
 * `unreadable`. Publication is atomic, so no reader ever catches a holder
 * mid-write: an unreadable file is debris — a run killed between writing its
 * record and publishing it, or a foreign file at the lock path — not a lock
 * somebody is in the middle of taking.
 *
 * @throws GymratError when the lockfile belongs to another user.
 */
function readLockfile(lockPath: string): LockfileState {
  let fd: number;
  try {
    fd = fs.openSync(lockPath, "r");
  } catch (error) {
    if (hasErrorCode(error, "ENOENT")) {
      return { kind: "absent" };
    }
    // A lockfile another user owns is unreadable to every later run, so the
    // steal path below can never reach it — the only way out is by hand.
    if (hasErrorCode(error, "EPERM") || hasErrorCode(error, "EACCES")) {
      throw new GymratError(
        `Lock file ${lockPath} could not be read: ${messageOf(error)}`,
        `It belongs to another user. Remove ${lockPath} yourself, then rerun.`,
        { cause: error },
      );
    }
    throw error;
  }

  let identity: LockfileIdentity;
  let contents: string;
  try {
    identity = identityOf(fd);
    contents = fs.readFileSync(fd, "utf8");
  } finally {
    fs.closeSync(fd);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(contents);
  } catch {
    return { kind: "unreadable", identity };
  }
  return lockHolderValidator.check(parsed)
    ? { kind: "held", holder: parsed, identity }
    : { kind: "unreadable", identity };
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

/** Delete `filePath` if it exists. Used on the lockfile and its scratch siblings alike. */
function unlinkIfExists(filePath: string): void {
  try {
    fs.unlinkSync(filePath);
  } catch (error) {
    if (!hasErrorCode(error, "ENOENT")) {
      throw error;
    }
  }
}

/**
 * Delete `lockPath` only while it still names the file `identity` came from.
 *
 * A path is not a lock: the file a run published can be displaced by a takeover
 * at any moment, and deleting whatever answers to the path would hand the next
 * holder's lock away to nobody. A different file there — or none at all — is
 * left exactly as found.
 */
function unlinkIfSameFile(lockPath: string, identity: LockfileIdentity): void {
  let fd: number;
  try {
    fd = fs.openSync(lockPath, "r");
  } catch (error) {
    if (hasErrorCode(error, "ENOENT")) {
      return;
    }
    throw error;
  }

  let current: LockfileIdentity;
  try {
    current = identityOf(fd);
  } finally {
    fs.closeSync(fd);
  }

  if (sameFile(current, identity)) {
    unlinkIfExists(lockPath);
  }
}

/**
 * Report a lock whose holder answered a liveness probe.
 *
 * Deleting the lockfile is deliberately not offered: the holder is provably
 * alive, so the only safe remedy is to wait it out.
 */
function heldByError(holder: LockHolder): GymratError {
  return new GymratError(
    `Lock held by PID ${String(holder.pid)} (${holder.command}, started ${holder.at})`,
    "Another gymrat run is active in this repo. Wait for it to finish.",
  );
}

/**
 * Publish `record` at `lockPath`, handing back the identity of the file now
 * standing there, or `undefined` when someone else got there first.
 *
 * The record is written to a scratch file beside the lock and only then linked
 * into place, so the lock path never exists holding half a record: readers see a
 * whole holder or nothing at all. The link is also the exclusive step — it fails
 * with `EEXIST` when the path is taken — so exactly one racer publishes.
 *
 * The identity is taken from the descriptor the record was written through,
 * which is the same file the link names, so it stays true no matter what takes
 * the lock path over afterwards.
 */
function publishLockRecord(lockPath: string, record: string): LockfileIdentity | undefined {
  const scratchPath = `${lockPath}.${String(process.pid)}.record`;
  const fd = fs.openSync(scratchPath, "w");
  let identity: LockfileIdentity;
  try {
    fs.writeFileSync(fd, record);
    identity = identityOf(fd);
  } finally {
    fs.closeSync(fd);
  }

  try {
    fs.linkSync(scratchPath, lockPath);
    return identity;
  } catch (error) {
    if (hasErrorCode(error, "EEXIST")) {
      return undefined;
    }
    throw error;
  } finally {
    unlinkIfExists(scratchPath);
  }
}

/**
 * Rethrow the failure to displace the stale lockfile at `lockPath`.
 *
 * `EPERM` and `EACCES` say the file belongs to another user — a sticky `/tmp` —
 * which no retry can resolve, so they are framed for whoever has to clean up.
 */
function rethrowDisplacementFailure(lockPath: string, error: unknown): never {
  if (hasErrorCode(error, "EPERM") || hasErrorCode(error, "EACCES")) {
    throw new GymratError(
      `Stale lock file ${lockPath} could not be removed: ${messageOf(error)}`,
      `It belongs to another user. Remove ${lockPath} yourself, then rerun.`,
      { cause: error },
    );
  }
  throw error;
}

/**
 * What asking for the right to displace a stale lockfile turned up.
 *
 * `gone` covers both ways the judged file can stop being the one at the lock
 * path: it vanished before the claim, or the claim came back naming a different
 * file.
 */
type ClaimOutcome =
  | { readonly kind: "claimed"; readonly claimPath: string }
  | { readonly kind: "blocked"; readonly claimPath: string }
  | { readonly kind: "gone" };

/**
 * Claim the right to displace the file `identity` was read from, handing back
 * the path the claim lives at.
 *
 * The claim is a second name for the lockfile itself, spelled out of that file's
 * identity, which makes it both halves of a safe steal. It is exclusive: two
 * racers reaching the same staleness verdict about one file ask for the same
 * name, `EEXIST` tells the loser so, and only the winner goes on to displace
 * anything. And it is proof: a racer whose claim turns out to name a different
 * file is holding a lock somebody published after the verdict, so it drops the
 * claim and leaves that lock where it stands.
 *
 * A steal that is not this racer's to make is reported as `blocked` — the claim
 * name is taken — or as `gone`, leaving acquisition to read the lock path
 * afresh. The two are told apart because a claim that stays taken across every
 * attempt is the fingerprint of a run that died holding it.
 *
 * @throws GymratError when the lockfile belongs to another user.
 */
function claimStaleLock(lockPath: string, identity: LockfileIdentity): ClaimOutcome {
  const claimPath = `${lockPath}.${String(identity.dev)}-${String(identity.ino)}.claim`;
  try {
    fs.linkSync(lockPath, claimPath);
  } catch (error) {
    if (hasErrorCode(error, "EEXIST")) {
      return { kind: "blocked", claimPath };
    }
    if (hasErrorCode(error, "ENOENT")) {
      return { kind: "gone" };
    }
    rethrowDisplacementFailure(lockPath, error);
  }

  const claimed = fs.statSync(claimPath, { bigint: true });
  if (sameFile(claimed, identity)) {
    return { kind: "claimed", claimPath };
  }
  unlinkIfExists(claimPath);
  return { kind: "gone" };
}

/**
 * Clear the claimed lockfile off the lock path. Returns whether it went away.
 *
 * Only the holder of the claim gets here, and nothing else may displace a
 * claimed file, so the file moved is the one that was judged stale. It is moved
 * rather than deleted so a run killed mid-steal leaves the lock path free rather
 * than holding a record it never published.
 *
 * @throws GymratError when the lockfile belongs to another user.
 */
function displaceStaleLock(lockPath: string): boolean {
  const asidePath = `${lockPath}.${String(process.pid)}.stale`;
  try {
    fs.renameSync(lockPath, asidePath);
  } catch (error) {
    if (hasErrorCode(error, "ENOENT")) {
      return false;
    }
    rethrowDisplacementFailure(lockPath, error);
  }
  unlinkIfExists(asidePath);
  return true;
}

/**
 * What taking over a lockfile no live process holds turned up.
 *
 * `blocked` carries the claim path that stood in the way, because a claim
 * nobody is behind can only be cleared by hand.
 */
type StealOutcome =
  | { readonly kind: "won"; readonly identity: LockfileIdentity }
  | { readonly kind: "lost" }
  | { readonly kind: "blocked"; readonly claimPath: string };

/** Take over a lockfile no live process holds. */
function stealLock(lockPath: string, identity: LockfileIdentity, record: string): StealOutcome {
  const claim = claimStaleLock(lockPath, identity);
  switch (claim.kind) {
    case "gone":
      return { kind: "lost" };
    case "blocked":
      return { kind: "blocked", claimPath: claim.claimPath };
    case "claimed":
      try {
        const published = displaceStaleLock(lockPath)
          ? publishLockRecord(lockPath, record)
          : undefined;
        return published === undefined ? { kind: "lost" } : { kind: "won", identity: published };
      } finally {
        unlinkIfExists(claim.claimPath);
      }
    default:
      return assertNever(claim);
  }
}

/**
 * A takeover that died holding its claim, leaving the lock impossible to steal.
 *
 * The claim outlives the run that made it, so every later steal of that same
 * file is refused the claim name forever. `holderPid` is the dead process the
 * lockfile named, or `undefined` when the lockfile was too damaged to name one.
 */
type WedgedTakeover = {
  readonly identity: LockfileIdentity;
  readonly claimPath: string;
  readonly holderPid: number | undefined;
};

function wedgedTakeoverError(lockPath: string, wedge: WedgedTakeover): GymratError {
  const nobodyHolds =
    wedge.holderPid === undefined
      ? "No gymrat process holds this lock."
      : `No gymrat process holds this lock (PID ${String(wedge.holderPid)} is dead).`;
  return new GymratError(
    `Lock at ${lockPath} was left behind by a run that died while taking it over.`,
    `${nobodyHolds} To unblock, delete ${lockPath} and ${wedge.claimPath}, then rerun.`,
  );
}

/** What one pass at the lock path turned up. */
type AttemptOutcome =
  | { readonly kind: "acquired"; readonly identity: LockfileIdentity }
  | { readonly kind: "retry" }
  | { readonly kind: "blocked"; readonly wedge: WedgedTakeover };

/** Read a steal that did not win as either worth retrying or as wedged. */
function attemptFromSteal(
  steal: StealOutcome,
  identity: LockfileIdentity,
  holderPid: number | undefined,
): AttemptOutcome {
  switch (steal.kind) {
    case "won":
      return { kind: "acquired", identity: steal.identity };
    case "lost":
      return { kind: "retry" };
    case "blocked":
      return { kind: "blocked", wedge: { identity, claimPath: steal.claimPath, holderPid } };
    default:
      return assertNever(steal);
  }
}

/**
 * Make one bid for the lock at `lockPath`: publish, or judge what is there.
 *
 * @throws GymratError when the lock is held by a process that is still running,
 *   or when the lockfile belongs to another user.
 */
function attemptAcquire(lockPath: string, record: string): AttemptOutcome {
  const published = publishLockRecord(lockPath, record);
  if (published !== undefined) {
    return { kind: "acquired", identity: published };
  }

  const state = readLockfile(lockPath);
  switch (state.kind) {
    case "absent":
      return { kind: "retry" };
    case "unreadable":
      return attemptFromSteal(
        stealLock(lockPath, state.identity, record),
        state.identity,
        undefined,
      );
    case "held": {
      if (isAlive(state.holder.pid)) {
        throw heldByError(state.holder);
      }
      const steal = stealLock(lockPath, state.identity, record);
      return attemptFromSteal(steal, state.identity, state.holder.pid);
    }
    default:
      return assertNever(state);
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
 * One exception: a run killed between claiming the right to displace a stale
 * lockfile and completing that displacement leaves a state no later run can
 * clear on its own. The thrown error names both files to delete.
 *
 * @throws GymratError when the lock is held by a process that is still running,
 *   when the lockfile belongs to another user, or when every attempt was refused
 *   by the claim of a takeover that never finished.
 */
export function acquireLock(lockPath: string, command: string): ReleaseLock {
  const holder: LockHolder = { pid: process.pid, command, at: new Date().toISOString() };
  const record = JSON.stringify(holder);

  fs.mkdirSync(path.dirname(lockPath), { recursive: true });

  let wedge: WedgedTakeover | undefined;
  let wedgedEveryAttempt = true;

  for (let attempt = 0; attempt < MAX_ACQUIRE_ATTEMPTS; attempt++) {
    const outcome = attemptAcquire(lockPath, record);
    if (outcome.kind === "acquired") {
      const { identity } = outcome;
      return () => {
        unlinkIfSameFile(lockPath, identity);
      };
    }

    // Only one file blocking every single attempt rules out the rival that
    // takes the lock, works, and releases it between two of our reads.
    const blocked = outcome.kind === "blocked" ? outcome.wedge : undefined;
    if (blocked === undefined) {
      wedgedEveryAttempt = false;
    } else if (wedge === undefined) {
      wedge = blocked;
    } else if (!sameFile(wedge.identity, blocked.identity)) {
      wedgedEveryAttempt = false;
    }
  }

  if (wedgedEveryAttempt && wedge !== undefined) {
    throw wedgedTakeoverError(lockPath, wedge);
  }

  throw new GymratError(
    `Lock at ${lockPath} was claimed by another process on every attempt.`,
    `Another gymrat run is active in this repo. Wait for it to finish. ` +
      `If no gymrat process is running, delete ${lockPath}.`,
  );
}
