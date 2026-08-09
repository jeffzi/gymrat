import fs from "node:fs";
import path from "node:path";

import { assertNever, GymratError, messageOf } from "../errors.js";
import type { IterationRecord, SessionLogRecord, SessionRecord } from "./records.js";
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
  /** Number of the most recently measured edit, `0` while nothing has been measured. */
  lastSeq: number;
}

/**
 * Append `record` to the session log at `jsonlPath`, creating its directory if needed.
 *
 * One record is one `appendFileSync` call: a single write of a single line is what lets
 * a reader treat the log as append-only truth rather than a file that may be caught
 * half-written.
 */
export function appendRecord(jsonlPath: string, record: SessionLogRecord): void {
  fs.mkdirSync(path.dirname(jsonlPath), { recursive: true });
  fs.appendFileSync(jsonlPath, `${JSON.stringify(record)}\n`);
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

  for (const record of records) {
    switch (record.type) {
      case "session":
        session ??= record;
        break;
      case "iteration":
        iterationCount += 1;
        lastIteration = record;
        unsettled = true;
        targetReachedBySeq.set(record.seq, record.targetReached);
        break;
      case "keep":
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
        unsettled = false;
        discardCount += 1;
        break;
      case "baseline":
      case "hook":
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
    lastSeq: lastIteration?.seq ?? 0,
  };
}
