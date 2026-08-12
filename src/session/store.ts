import fs from "node:fs";
import path from "node:path";

import { assertNever, GymratError, messageOf } from "../errors.js";
import { sessionJsonlPath } from "./paths.js";
import type {
  FinalizeRecord,
  IterationRecord,
  SessionLogRecord,
  SessionRecord,
} from "./records.js";
import { parseRecord } from "./records.js";

/** What a session log adds up to: the state every loop command reads before acting. */
export interface SessionState {
  /** The header opening the log, absent while no session has been started. */
  session: SessionRecord | undefined;
  /** How many edits have been measured. */
  iterationCount: number;
  /** The most recently measured edit, absent while nothing has been measured. */
  lastIteration: IterationRecord | undefined;
  /** Whether a measured edit is waiting to be kept or discarded. */
  unsettled: boolean;
  /** How many edits were committed. */
  keepCount: number;
  /** How many edits were reverted. */
  discardCount: number;
  /** Whether the last committed keep settled an edit that reached the target metric. */
  targetReachedAndKept: boolean;
  /**
   * The highest number any iteration or settling record has taken, `0` while the log has none.
   *
   * A refusal that settles nothing still claims a number, so this is a high-water mark
   * rather than the last iteration's number: the next record to need a number of its own
   * takes `lastSeq + 1` and cannot alias one the log already carries.
   */
  lastSeq: number;
  /**
   * The record that closed the session, absent while the session is still open.
   *
   * A finalized session accepts no further loop commands — see
   * {@link requireOpenSession} — but still reads and renders.
   */
  finalized: FinalizeRecord | undefined;
}

/** An open session, with everything reading its log already produced. */
export interface RequiredSession {
  /** The header the log opens with. */
  session: SessionRecord;
  /** What the whole log folds to, `session` included. */
  state: SessionState;
  /** The log the session was read from. */
  jsonlPath: string;
  /** Every record the log holds, in file order — the same ones `state` folds. */
  records: SessionLogRecord[];
}

/**
 * Append `record` to the session log at `jsonlPath`, creating its directory if needed.
 *
 * One record is one `appendFileSync` call: a single write of a single line is what lets
 * a reader treat the log as append-only truth rather than a file that may be caught
 * half-written.
 *
 * @throws GymratError when `record` would not read back — the log is left untouched,
 * and no directory is created for a log that does not exist yet.
 */
export function appendRecord(jsonlPath: string, record: SessionLogRecord): void {
  const line = serializeRecord(record);
  fs.mkdirSync(path.dirname(jsonlPath), { recursive: true });
  fs.appendFileSync(jsonlPath, `${line}\n`);
}

/**
 * The line `record` is written as, once proven to parse back into the same record.
 *
 * A record can satisfy the compiler and still not survive JSON: `NaN` and `Infinity`
 * are `number` to TypeScript and `null` on the wire, and any measurement an adapter
 * hands back unchecked can carry one. So the check runs the serialized line — not the
 * object — through the very parser {@link readRecords} will use, reusing its compiled
 * validators. Nothing is written until that round trip succeeds, which is what stops a
 * single bad measurement from leaving the whole session log unreadable.
 */
function serializeRecord(record: SessionLogRecord): string {
  let line: string;
  try {
    line = JSON.stringify(record);
    const readBack: unknown = JSON.parse(line);
    parseRecord(readBack);
  } catch (error) {
    throw new GymratError(
      `Refusing to log an unreadable ${record.type} record: ${messageOf(error)}`,
      "Nothing was written. A metric that is NaN or Infinity becomes null in JSON and no longer reads back.",
      { cause: error },
    );
  }
  return line;
}

/**
 * Read every record from the session log at `jsonlPath`, in file order.
 *
 * A log that does not exist reads as no session — an empty array — because the loop
 * commands distinguish "no session" from "corrupt session" and only the latter is a
 * failure.
 *
 * @throws GymratError when a line is not JSON, when a line matches no record schema, or
 * when the first record is not a session header. Every message names the log and the
 * 1-based line at fault.
 */
export function readRecords(jsonlPath: string): SessionLogRecord[] {
  if (!fs.existsSync(jsonlPath)) {
    return [];
  }

  const records: SessionLogRecord[] = [];
  const lines = fs.readFileSync(jsonlPath, "utf-8").split("\n");

  for (const [index, line] of lines.entries()) {
    if (line.trim() === "") {
      continue;
    }
    const at = `${jsonlPath}:${index + 1}`;

    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      throw new GymratError(`Invalid JSON at ${at}`, `Line ${index + 1} is not a JSON object.`, {
        cause: error,
      });
    }

    let record: SessionLogRecord;
    try {
      record = parseRecord(value);
    } catch (error) {
      const hint = error instanceof GymratError ? error.hint : undefined;
      throw new GymratError(`${messageOf(error)} (at ${at})`, hint, { cause: error });
    }

    if (records.length === 0 && record.type !== "session") {
      throw new GymratError(
        `Expected session header at ${at}, got a ${record.type} record`,
        "The session log is corrupt; start a new session.",
      );
    }
    records.push(record);
  }

  return records;
}

/**
 * The commit made by the last keep in `records` that committed, absent when none did.
 *
 * The session's baseline worktree advances to every kept commit, so this — not the
 * session header's pinned SHA — is where the baseline stands once a keep has landed.
 * A blocked keep committed nothing and leaves the baseline where it was.
 */
export function lastKeptCommit(records: SessionLogRecord[]): string | undefined {
  let commit: string | undefined;

  for (const record of records) {
    if (record.type === "keep" && record.status === "committed" && record.commit !== undefined) {
      commit = record.commit;
    }
  }

  return commit;
}

/**
 * Whether `records` ends on a keep the loop blocked for a gating regression.
 *
 * The block settles the iteration it refused — {@link foldSession} clears
 * `unsettled` for it — but the edit it would not commit is still standing in the
 * experiment worktree, so `discard` accepts this as the one settled state it may
 * still revert, making the refusal's own hint true. Any iteration, keep, or
 * discard written after the block supersedes it — except a keep refused for want
 * of a measurement, which is what retrying `keep` after the block appends and
 * which must not wedge the edit in place by closing the window the block opened.
 */
export function endsOnGatingBlock(records: SessionLogRecord[]): boolean {
  let blocked = false;

  // The header, baseline samples, and hook runs neither measure nor settle an
  // edit, so they leave the answer where the last keep, iteration, or discard put it.
  for (const record of records) {
    if (record.type === "keep") {
      // A "nothing-measured" refusal commits nothing and settles nothing, so it
      // leaves the standing edit — and the window to revert it — as it found them.
      if (record.reason !== "nothing-measured") {
        blocked = record.status === "blocked" && record.reason === "gating-regression";
      }
    } else if (record.type === "iteration" || record.type === "discard") {
      blocked = false;
    }
  }

  return blocked;
}

/**
 * Fold `records` into the state they describe.
 *
 * Folds whatever it is given: validating the log — that it parses, and that it opens with
 * a session header — belongs to {@link readRecords}.
 */
export function foldSession(records: SessionLogRecord[]): SessionState {
  /** Whether the iteration numbered `seq` reached the target metric. */
  const targetReachedBySeq = new Map<number, boolean>();

  let session: SessionRecord | undefined;
  let iterationCount = 0;
  let lastIteration: IterationRecord | undefined;
  let unsettled = false;
  let keepCount = 0;
  let discardCount = 0;
  let targetReachedAndKept = false;
  let lastSeq = 0;
  let finalized: FinalizeRecord | undefined;

  for (const record of records) {
    switch (record.type) {
      case "session":
        session ??= record;
        break;
      case "iteration":
        iterationCount += 1;
        lastSeq = Math.max(lastSeq, record.seq);
        lastIteration = record;
        unsettled = true;
        targetReachedBySeq.set(record.seq, record.targetReached);
        break;
      case "keep":
        lastSeq = Math.max(lastSeq, record.seq);
        if (record.status === "committed") {
          unsettled = false;
          keepCount += 1;
          // The keep settles the iteration it shares a seq with, so the stop condition
          // follows the last commit rather than the last measurement.
          targetReachedAndKept = targetReachedBySeq.get(record.seq) ?? false;
        } else if (record.reason !== undefined && record.reason !== "checks-failed") {
          // A blocked keep leaves the edit uncommitted, so it settles the iteration
          // too — unblocking `iterate` — except for "checks-failed", where the user
          // is expected to fix the failure and retry `keep` on the same iteration.
          // A blocked keep with no reason at all is held to the same rule: nothing
          // says the iteration is beyond recovery, so it stays unsettled.
          unsettled = false;
        }
        break;
      case "discard":
        lastSeq = Math.max(lastSeq, record.seq);
        unsettled = false;
        discardCount += 1;
        break;
      case "finalize":
        // Closing the session settles nothing and measures nothing: it collapses
        // work the counters have already counted.
        finalized = record;
        break;
      case "baseline":
      case "hook":
        // A hook's seq names the iteration it runs around rather than claiming a
        // number of its own: a run whose hook fired and then failed must leave that
        // number free for the retry.
        break;
      default:
        assertNever(record);
    }
  }

  return {
    session,
    iterationCount,
    lastIteration,
    unsettled,
    keepCount,
    discardCount,
    targetReachedAndKept,
    lastSeq,
    finalized,
  };
}

/**
 * The session open in `root`, or the error telling the caller to open one.
 *
 * `verb` names what the caller was about to do — "measuring an edit" — and becomes
 * the thing the hint says no session was open for, so every loop command refuses in
 * its own words while sharing one guard.
 *
 * @throws GymratError when no session has been started, or when the log is corrupt —
 * every parse failure names the log and the line at fault.
 */
export function requireSession(root: string, verb: string): RequiredSession {
  const jsonlPath = sessionJsonlPath(root);
  const records = readRecords(jsonlPath);
  const state = foldSession(records);

  if (state.session === undefined) {
    throw new GymratError(`No session in ${root}`, `Run gymrat start to open one before ${verb}.`);
  }

  return { session: state.session, state, jsonlPath, records };
}

/**
 * The *open* session in `root`, or the error telling the caller why there is none.
 *
 * Every command that writes to the log goes through this rather than
 * {@link requireSession}: a finalized session has had its kept work collapsed and
 * its worktrees removed, so appending to it would record work no branch carries.
 * `status` keeps the unguarded path — reading a closed session is exactly what it
 * is for.
 *
 * @throws GymratError when no session has been started, when the log is corrupt, or
 * when the session was already finalized.
 */
export function requireOpenSession(root: string, verb: string): RequiredSession {
  const required = requireSession(root, verb);
  const { finalized } = required.state;

  if (finalized !== undefined) {
    throw new GymratError(
      `Session ${required.session.sessionId} was finalized onto ${finalized.branch}`,
      `Run gymrat start to open a new session before ${verb}.`,
    );
  }

  return required;
}
