import { execFileSync, type StdioOptions } from "node:child_process";
import crypto from "node:crypto";
import os from "node:os";
import path from "node:path";

import { GymratError } from "../errors.js";

const GIT_STDIO: StdioOptions = ["pipe", "pipe", "pipe"];

/** Directory under the repo root holding session state and its worktrees. */
export const SESSION_DIR_NAME = ".gymrat";

/** Number of hex digits of the repo-root digest that name a lockfile. */
const LOCK_DIGEST_LENGTH = 12;

/**
 * Locate the top level of the git repository containing `cwd`.
 *
 * @throws GymratError when `cwd` is not inside a git repository.
 */
export function repoRoot(cwd: string = process.cwd()): string {
  try {
    return execFileSync("git", ["rev-parse", "--show-toplevel"], {
      cwd,
      encoding: "utf-8",
      stdio: GIT_STDIO,
    }).trim();
  } catch (error) {
    throw new GymratError(
      `Not a git repository: ${cwd}`,
      "Run gymrat from inside a git repository.",
      { cause: error },
    );
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
