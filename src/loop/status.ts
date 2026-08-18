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
  formatStatusFinalized,
  formatStatusFooter,
  formatStatusHeader,
  formatStatusIteration,
  formatStatusSettle,
} from "../report/loop.js";
import type { DiscardRecord, KeepRecord, SessionLogRecord } from "../session/records.js";
import { requireSession } from "../session/store.js";

/** What a single settling record says became of the iteration it settles. */
function settleStateOf(record: KeepRecord | DiscardRecord): SettleState {
  if (record.type === "discard") {
    return { kind: "discarded" };
  }
  return record.status === "committed"
    ? { kind: "kept", ...(record.commit !== undefined && { commit: record.commit }) }
    : { kind: "keep-blocked", ...(record.reason !== undefined && { reason: record.reason }) };
}

/** The iteration a settling record is still waiting to hear about. */
interface PendingIteration {
  position: number;
  seq: number;
}

/**
 * The last keep-blocked that settled the pending iteration, so a discard
 * that supersedes it can relocate the block to a standalone line and take
 * the iteration's display for itself.
 */
interface LastGatingBlock {
  iterationPosition: number;
  blockPosition: number;
  blockState: SettleState;
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
  let pending: PendingIteration | undefined;
  let lastBlock: LastGatingBlock | undefined;

  for (const [position, record] of records.entries()) {
    switch (record.type) {
      case "iteration":
        pending = { position, seq: record.seq };
        lastBlock = undefined;
        break;
      case "keep":
      case "discard":
        lastBlock = applySettleRecord({ states, pending, lastBlock, position }, record);
        break;
      default:
        break;
    }
  }
  return states;
}

/**
 * Record `record`'s settle state and report the gating block still standing
 * afterwards, if any.
 *
 * A record settling the pending iteration writes there directly, and starts
 * tracking the block if it is one. A discard with no pending match instead
 * looks for a standing block: that is a discard that follows a gating block,
 * which supersedes the block's claim on the iteration it refused — the block
 * moves to its own position and the iteration shows as discarded. Anything
 * else settles no iteration and keeps its own line.
 */
interface SettleRecordContext {
  states: Map<number, SettleState>;
  pending: PendingIteration | undefined;
  lastBlock: LastGatingBlock | undefined;
  position: number;
}

function applySettleRecord(
  ctx: SettleRecordContext,
  record: KeepRecord | DiscardRecord,
): LastGatingBlock | undefined {
  const { states, pending, lastBlock, position } = ctx;
  const settle = settleStateOf(record);

  if (pending !== undefined && pending.seq === record.seq) {
    states.set(pending.position, settle);
    return settle.kind === "keep-blocked"
      ? { iterationPosition: pending.position, blockPosition: position, blockState: settle }
      : undefined;
  }

  if (record.type === "discard" && lastBlock !== undefined) {
    states.set(lastBlock.iterationPosition, settle);
    states.set(lastBlock.blockPosition, lastBlock.blockState);
    return undefined;
  }

  states.set(position, settle);
  return lastBlock;
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
  const { session, state, records } = requireSession(root, "asking for its status");

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
    ...(config.runbook === undefined ? [] : [`runbook ${config.runbook}`]),
    ...history,
    ...formatStatusFooter({
      iterationCount: state.iterationCount,
      keepCount: state.keepCount,
      discardCount: state.discardCount,
      targetReached: state.targetReachedAndKept,
      ...(config.stop === undefined ? {} : { stop: config.stop }),
    }),
    ...(state.finalized === undefined ? [] : [formatStatusFinalized(state.finalized)]),
  ].join("\n");
}
