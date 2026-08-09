import crypto from "node:crypto";

import type { ResolvedConfig } from "../config.js";
import { GymratError } from "../errors.js";
import { sessionJsonlPath } from "../session/paths.js";
import type { SessionRecord } from "../session/records.js";
import { appendRecord, foldSession, readRecords, type SessionState } from "../session/store.js";
import { createWorkspace, detectWorkspace, recreateWorkspace } from "../session/workspace.js";
import { resolveTarget } from "../targets.js";

/** Ref the baseline is pinned to when the caller names none. */
const DEFAULT_BASELINE_REF = "HEAD";

/** Hex digits distinguishing two sessions started within the same second. */
const SESSION_ID_ENTROPY_BYTES = 2;

/** A session ready to iterate in, together with everything its log already holds. */
export interface StartResult {
  /** The header the session's log opens with, newly written or read back. */
  session: SessionRecord;
  /** What the log adds up to, so a resumed session can report its history. */
  state: SessionState;
  /** Whether the session was already on disk. */
  resumed: boolean;
}

/**
 * Create the repository's optimization session, or resume the one it already has.
 *
 * Resuming is what makes a second `start` safe: an existing log is never appended
 * to, and only a worktree that went missing is put back. Holding the repository
 * lock across the call is the caller's job — this touches `.gymrat/` and the
 * repository's branches, so two concurrent runs must not reach it.
 *
 * @throws GymratError when `ref` names no commit, when the log is corrupt, or
 *   when git refuses to create or recreate the workspace.
 */
export function startSession(
  root: string,
  ref: string | undefined,
  config: ResolvedConfig,
): StartResult {
  const jsonlPath = sessionJsonlPath(root);
  const state = foldSession(readRecords(jsonlPath));
  const { session } = state;

  if (session === undefined) {
    return createSession(root, jsonlPath, ref ?? DEFAULT_BASELINE_REF, config);
  }

  if (detectWorkspace(root) !== "present") {
    recreateWorkspace(root, session.branch, session.baseline.sha);
  }
  return { session, state, resumed: true };
}

/**
 * Pin the baseline, build the workspace, and write the header that opens the log.
 *
 * The header lands last: a session the log claims exists but whose branch git
 * never created would send every later command looking for a workspace that
 * is not there.
 */
function createSession(
  root: string,
  jsonlPath: string,
  ref: string,
  config: ResolvedConfig,
): StartResult {
  const sha = resolveBaselineSha(ref, root);
  const now = new Date();
  const sessionId = newSessionId(now);
  const workspace = createWorkspace(root, sessionId, { ref, sha });

  const session: SessionRecord = {
    type: "session",
    schemaVersion: 1,
    sessionId,
    createdAt: now.toISOString(),
    baseline: workspace.baseline,
    branch: workspace.branch,
    worktrees: workspace.worktrees,
    config: snapshotConfig(config),
  };
  appendRecord(jsonlPath, session);

  return { session, state: foldSession([session]), resumed: false };
}

/**
 * The commit `ref` names, peeled through the same resolution `compare` uses.
 *
 * A baseline is always a ref: the session's own worktrees are where benchmarks
 * run, so a directory — which {@link resolveTarget} otherwise accepts, and
 * prefers over a ref of the same name — has nothing to pin a branch at.
 *
 * @throws GymratError when `ref` resolves to a directory or to no commit at all.
 */
function resolveBaselineSha(ref: string, root: string): string {
  const target = resolveTarget(ref, root);
  if (target.kind !== "ref") {
    throw new GymratError(
      `Cannot start a session at '${ref}': it names a directory, not a git ref`,
      "Pass a branch, tag, or commit the session's baseline is pinned to.",
    );
  }
  return target.resolvedSha;
}

/**
 * `<YYYYMMDD-HHmmss>-<4 hex>` in UTC.
 *
 * The timestamp sorts sessions the way they were started and reads back as a
 * date; the random suffix keeps two sessions started in the same second — and
 * therefore their branches — apart.
 */
function newSessionId(now: Date): string {
  const iso = now.toISOString();
  const date = iso.slice(0, 10).replaceAll("-", "");
  const time = iso.slice(11, 19).replaceAll(":", "");
  const suffix = crypto.randomBytes(SESSION_ID_ENTROPY_BYTES).toString("hex");
  return `${date}-${time}-${suffix}`;
}

/**
 * The settings the header records as provenance.
 *
 * Only the keys the session schema declares survive, and an absent optional key
 * stays absent rather than becoming an explicit `undefined`. Every command
 * re-reads `gymrat.json` for the settings it acts on, so this snapshot answers
 * "what was this session started with", never "what runs next".
 */
function snapshotConfig(config: ResolvedConfig): SessionRecord["config"] {
  return {
    bench: config.bench,
    ...(config.prepare === undefined ? {} : { prepare: config.prepare }),
    adapter: config.adapter,
    samples: config.samples,
    timeoutSeconds: config.timeoutSeconds,
    primary: config.primary,
    ...(config.filter === undefined ? {} : { filter: config.filter }),
    ...(config.hooks === undefined ? {} : { hooks: config.hooks }),
  };
}
