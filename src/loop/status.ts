/**
 * What `gymrat status` reports: the session, rebuilt from its log and nothing else.
 *
 * Every other loop command writes to the log; this one only reads it, which is
 * why it takes no repository lock. A status that agreed with the log only while
 * some process kept state in memory would be worth nothing to an agent whose
 * every command is a fresh process.
 */

import type { ResolvedConfig } from "../config.js";
import { GymratError } from "../errors.js";
import type { SettleState } from "../report/loop.js";
import {
  formatStatusBaseline,
  formatStatusFooter,
  formatStatusHeader,
  formatStatusIteration,
} from "../report/loop.js";
import { sessionJsonlPath } from "../session/paths.js";
import type { KeepRecord, SessionLogRecord } from "../session/records.js";
import { foldSession, readRecords } from "../session/store.js";

/** What became of the iteration a keep record settled. */
function keepSettleState(record: KeepRecord): SettleState {
  return record.status === "committed"
    ? { kind: "kept", commit: record.commit }
    : { kind: "keep-blocked", reason: record.reason };
}

/**
 * What became of each measured iteration, keyed by its number.
 *
 * The last settling record wins: an iteration whose keep the checks blocked, and
 * which a later keep committed, is a kept iteration — the blocked attempt is
 * history the log still holds, not the state the iteration rests in.
 */
function settleStates(records: readonly SessionLogRecord[]): Map<number, SettleState> {
  const states = new Map<number, SettleState>();
  for (const record of records) {
    switch (record.type) {
      case "keep":
        states.set(record.seq, keepSettleState(record));
        break;
      case "discard":
        states.set(record.seq, { kind: "discarded" });
        break;
      default:
        break;
    }
  }
  return states;
}

/**
 * The session's whole history, as the agent reads it back.
 *
 * The records are walked in file order rather than summarized, so the report
 * tells the session's story in the order it happened; the folded state supplies
 * only the totals. Hook records are left out — a hook is machinery around an
 * iteration, not a step of the loop the agent decides on.
 *
 * The stop condition comes from live configuration rather than the session
 * record's snapshot, matching every other loop command: an agent that raised its
 * iteration budget mid-session must see the budget it now runs under.
 *
 * @throws GymratError when no session has been started, or when the log is
 *   corrupt — every parse failure names the log and the line at fault.
 */
export function statusSession(root: string, config: ResolvedConfig): string {
  const records = readRecords(sessionJsonlPath(root));
  const state = foldSession(records);
  if (state.session === undefined) {
    throw new GymratError(
      `No session in ${root}`,
      "Run gymrat start to open one before asking for its status.",
    );
  }

  const settled = settleStates(records);
  const history = records.flatMap((record) => {
    switch (record.type) {
      case "baseline":
        return [formatStatusBaseline(record)];
      case "iteration":
        return [
          formatStatusIteration({
            seq: record.seq,
            deltaPct: record.primary.deltaPct,
            outcome: record.outcome,
            settle: settled.get(record.seq) ?? { kind: "unsettled" },
          }),
        ];
      default:
        return [];
    }
  });

  return [
    ...formatStatusHeader(state.session),
    ...history,
    ...formatStatusFooter({
      iterationCount: state.iterationCount,
      keepCount: state.keepCount,
      discardCount: state.discardCount,
      targetReached: state.targetReachedAndKept,
      ...(config.stop === undefined ? {} : { stop: config.stop }),
    }),
  ].join("\n");
}
