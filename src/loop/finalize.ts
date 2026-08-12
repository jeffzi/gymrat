import { GymratError } from "../errors.js";
import { runGit } from "../git.js";
import { pluralize } from "../report/format.js";
import { SHORT_SHA_LENGTH } from "../report/loop.js";
import type { FinalizeRecord, KeepRecord, SessionLogRecord } from "../session/records.js";
import { appendRecord, requireOpenSession } from "../session/store.js";
import { isWorktreeDirty, removeWorktrees, runGitStep } from "../session/workspace.js";

const SETTLE_FIRST_HINT = "Run gymrat keep or gymrat discard before closing the session.";

/** What a caller can hand a finalize beyond the repository it runs in. */
export interface FinalizeOptions {
  /** The squash commit's message; absent, one is generated from the kept commits. */
  message?: string;
  /** The branch to point at the squash commit; absent, `<session branch>-final`. */
  branch?: string;
}

/** One closed session: what was written to the log, and what to print about it. */
export interface FinalizeResult {
  /** The record appended to the session log. */
  record: FinalizeRecord;
  /** The finalize as the agent reads it: the branch, the commit, and any cleanup left to do. */
  report: string;
}

/**
 * Collapse a session's kept commits into one commit on the pinned baseline and close the session.
 *
 * The squash is built with plumbing — `commit-tree` against the session branch's
 * tree — so nothing is ever checked out: the user's own working copy stays on
 * whatever branch it was on, which a `merge --squash` could not promise. The
 * session branch is left in place too, so the iteration-by-iteration history
 * that earned the squash stays readable.
 *
 * Holding the repository lock across the call is the caller's job: the record
 * and the worktree removal it explains must not be separable by another run.
 *
 * @throws GymratError when no open session exists, when the session kept nothing,
 *   when an iteration is still unsettled, when the experiment worktree carries
 *   uncommitted work, when the target branch already exists, or when git refuses
 *   to build the squash commit.
 */
export function finalizeSession(root: string, options: FinalizeOptions = {}): FinalizeResult {
  const { session, state, jsonlPath, records } = requireOpenSession(root, "closing the session");

  if (state.keepCount === 0) {
    throw new GymratError(
      `Finalize refused: session ${session.sessionId} has kept nothing to squash.`,
      "Run gymrat keep on a measured edit before closing the session.",
    );
  }
  if (state.unsettled) {
    throw new GymratError(
      `Finalize refused: iteration ${state.lastSeq} has been neither kept nor discarded.`,
      SETTLE_FIRST_HINT,
    );
  }
  if (isWorktreeDirty(session.worktrees.experiment)) {
    throw new GymratError(
      `Finalize refused: the experiment worktree at ${session.worktrees.experiment} carries uncommitted work.`,
      SETTLE_FIRST_HINT,
    );
  }

  const branch = options.branch ?? `${session.branch}-final`;
  if (branchExists(root, branch)) {
    throw new GymratError(
      `Finalize refused: the branch '${branch}' already exists.`,
      `Name another with --branch <name>, or delete it with: git branch -D ${branch}`,
    );
  }

  const message = options.message ?? generatedMessage(records);
  const commit = squashOntoBaseline(root, session.branch, session.baseline.sha, message);
  runGitStep(
    ["branch", branch, commit],
    root,
    `Cannot point '${branch}' at the squash commit ${commit}`,
    `Inspect the branches this repository has with: git branch --list`,
  );

  const record: FinalizeRecord = {
    type: "finalize",
    at: new Date().toISOString(),
    branch,
    commit,
    message,
  };
  // Record the squash before clearing the worktrees: the branch already carries
  // the work, so a removal that fails must not leave the session open on a log
  // that never mentions the commit the agent is about to be told to look at.
  appendRecord(jsonlPath, record);

  const warnings = removeWorktrees(root, session.worktrees);

  return { record, report: finalizeReport(record, state.keepCount, session.branch, warnings) };
}

/**
 * Build the one commit that carries the session branch's tree onto the pinned baseline.
 *
 * Reading the tree and writing the commit are both plumbing, so neither needs —
 * or moves — a checkout. The single parent is the baseline the session started
 * from, which is what makes the result a squash rather than a merge.
 */
function squashOntoBaseline(
  root: string,
  sessionBranch: string,
  baselineSha: string,
  message: string,
): string {
  const tree = runGitStep(
    ["rev-parse", `${sessionBranch}^{tree}`],
    root,
    `Cannot read the tree of the session branch '${sessionBranch}'`,
    `Check that the branch is still there: git branch --list ${sessionBranch}`,
  ).trim();

  return runGitStep(
    ["commit-tree", tree, "-p", baselineSha, "-m", message],
    root,
    `Cannot build the squash commit from ${sessionBranch} onto ${baselineSha}`,
    `Check that ${baselineSha} is a commit this repository has: git cat-file -t ${baselineSha}`,
  ).trim();
}

/** Whether `root` already has a local branch named `branch`. */
function branchExists(root: string, branch: string): boolean {
  try {
    runGit(["show-ref", "--verify", "--quiet", `refs/heads/${branch}`], root);
    return true;
  } catch {
    return false;
  }
}

/** The body line a committed keep gets when it names neither a message nor a commit. */
const UNNAMED_KEEP_LINE = "(no message)";

/**
 * The squash message written when the caller supplied none.
 *
 * The subject counts what was collapsed and the body lists the kept commits in
 * the order they landed, so the one commit that reaches the user's branch still
 * says what the session did — the per-iteration history stays on the session
 * branch, which nothing but the reader's curiosity brings back.
 *
 * A keep whose `message` the log omits — which gymrat never writes, but a
 * hand-edited or older log can hold — falls back to its short commit rather
 * than dropping its line and leaving the subject claiming more than the body
 * shows.
 */
function generatedMessage(records: SessionLogRecord[]): string {
  const kept = records
    .filter(
      (record): record is KeepRecord => record.type === "keep" && record.status === "committed",
    )
    .map(
      (record) => record.message ?? record.commit?.slice(0, SHORT_SHA_LENGTH) ?? UNNAMED_KEEP_LINE,
    );

  const subject = `gymrat: squash ${pluralize(kept.length, "kept iteration")}`;
  return `${subject}\n\n${kept.join("\n")}`;
}

/** The finalize as the agent reads it, with anything git would not clean up spelled out. */
function finalizeReport(
  record: FinalizeRecord,
  keepCount: number,
  sessionBranch: string,
  warnings: string[],
): string {
  const lines = [
    `Finalized onto ${record.branch} as ${record.commit.slice(0, SHORT_SHA_LENGTH)}`,
    `  squashed ${pluralize(keepCount, "kept iteration")} into one commit`,
    `  the session is closed; ${sessionBranch} is left in place for its history`,
    ...warnings,
  ];
  return lines.join("\n");
}
