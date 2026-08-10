/**
 * What `gymrat status` reports: the session, rebuilt from its log and nothing else.
 *
 * Every other loop command writes to the log; this one only reads it, which is
 * why it takes no repository lock. A status that agreed with the log only while
 * some process kept state in memory would be worth nothing to an agent whose
 * every command is a fresh process.
 */

import type { BenchlessConfig } from "../config.js";
import type { SettleState } from "../report/loop.js";
import {
  formatStatusBaseline,
  formatStatusFooter,
  formatStatusHeader,
  formatStatusIteration,
  formatStatusSettle,
} from "../report/loop.js";
import type { DiscardRecord, KeepRecord, SessionLogRecord } from "../session/records.js";
import { readRecords, requireSession } from "../session/store.js";

/** What a single settling record says became of the iteration it settles. */
function settleStateOf(record: KeepRecord | DiscardRecord): SettleState {
  if (record.type === "discard") {
    return { kind: "discarded" };
  }
  return record.status === "committed"
    ? { kind: "kept", commit: record.commit }
    : { kind: "keep-blocked", reason: record.reason };
}

/**
 * The settle state each record's own line states, keyed by that record's
 * position in the log.
 *
 * A settling record settles the iteration it follows, not merely the one it
 * shares a number with. `keep` numbers a refusal that measured nothing past
 * every iteration on file, so the log can hold a blocked keep bearing the number
 * an iteration was minted with only afterwards; matching on the number alone
 * would hand that stale refusal to the new iteration and call it settled.
 *
 * The last settling record wins: an iteration whose keep the checks blocked, and
 * which a later keep committed, is a kept iteration — the blocked attempt is
 * history the log still holds, not the state the iteration rests in.
 *
 * A record that settled no iteration keeps an entry under its own position, and
 * so a line of its own: a refused keep is a decision the log reads back, never
 * one it erases.
 */
function settleStates(records: readonly SessionLogRecord[]): Map<number, SettleState> {
  const states = new Map<number, SettleState>();
  let pending: { position: number; seq: number } | undefined;

  for (const [position, record] of records.entries()) {
    switch (record.type) {
      case "iteration":
        pending = { position, seq: record.seq };
        break;
      case "keep":
      case "discard":
        if (pending !== undefined && pending.seq === record.seq) {
          states.set(pending.position, settleStateOf(record));
        } else {
          states.set(position, settleStateOf(record));
        }
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
export function statusSession(root: string, config: BenchlessConfig): string {
  const { session, state, jsonlPath } = requireSession(root, "asking for its status");
  const records = readRecords(jsonlPath);

  const settled = settleStates(records);
  const history = records.flatMap((record, position) => {
    switch (record.type) {
      case "baseline":
        return [formatStatusBaseline(record)];
      case "iteration":
        return [
          formatStatusIteration({
            seq: record.seq,
            deltaPct: record.primary.deltaPct,
            outcome: record.outcome,
            settle: settled.get(position) ?? { kind: "unsettled" },
          }),
        ];
      case "keep":
      case "discard": {
        const settle = settled.get(position);
        return settle === undefined ? [] : [formatStatusSettle(settle)];
      }
      default:
        return [];
    }
  });

  return [
    ...formatStatusHeader(session),
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
