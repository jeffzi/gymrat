import crypto from "node:crypto";
import os from "node:os";
import path from "node:path";

import { repositoryLookupError, runGit } from "../git.js";

/** Directory under the repo root holding session state and its worktrees. */
export const SESSION_DIR_NAME = ".gymrat";

/** Number of hex digits of the repo-root digest that name a lockfile. */
const LOCK_DIGEST_LENGTH = 12;

/**
 * Locate the top level of the git repository containing `cwd`.
 *
 * @throws NotAGitRepositoryError when git places `cwd` outside every repository.
 * @throws GymratError when git could not answer where `cwd` sits.
 */
export function repoRoot(cwd: string = process.cwd()): string {
  try {
    // git returns forward slashes on every platform; normalize so the
    // path hashes and compares identically to native paths elsewhere.
    return path.normalize(runGit(["rev-parse", "--show-toplevel"], cwd).trim());
  } catch (error) {
    throw repositoryLookupError(cwd, error);
  }
}

/** Directory holding the session log and the worktrees a session owns. */
export function sessionDir(root: string): string {
  return path.join(root, SESSION_DIR_NAME);
}

/** Append-only log of session events. */
export function sessionJsonlPath(root: string): string {
  return path.join(sessionDir(root), "session.jsonl");
}

/**
 * Where the log of the session `sessionId` closed on is kept.
 *
 * A finalized session is moved aside rather than deleted, so the history that
 * earned a squash commit is still on disk when someone asks what produced it.
 * The id is in the name so a repository can hold every session it ever ran.
 */
export function archivedSessionPath(root: string, sessionId: string): string {
  return path.join(sessionDir(root), `session-${sessionId}.jsonl`);
}

/** Parent of the per-side worktree directories. */
export function worktreesDir(root: string): string {
  return path.join(sessionDir(root), "worktrees");
}

/** Worktree holding the revision under test. */
export function experimentWorktreeDir(root: string): string {
  return path.join(worktreesDir(root), "experiment");
}

/** Worktree holding the revision being compared against. */
export function baselineWorktreeDir(root: string): string {
  return path.join(worktreesDir(root), "baseline");
}

/**
 * Lockfile guarding concurrent sessions over the repository at `root`.
 *
 * The name is a digest of the root rather than the root itself so it survives
 * path separators and length limits, and it lives in the system temp dir rather
 * than under `.gymrat` so a repository nobody has started a session in never
 * gains a directory.
 */
export function lockfilePath(root: string): string {
  const digest = crypto
    .createHash("sha256")
    .update(root)
    .digest("hex")
    .slice(0, LOCK_DIGEST_LENGTH);
  return path.join(os.tmpdir(), `gymrat-lock-${digest}.json`);
}
