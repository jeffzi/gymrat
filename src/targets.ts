import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { GymratError, stderrTextOf } from "./errors.js";
import { runGit } from "./git.js";

/** A directory benchmarked where it sits, with no worktree of its own. */
export interface InPlaceTarget {
  kind: "in-place";
  dir: string;
}

/** A git ref benchmarked in a worktree, pinned to the SHA the ref had at resolution time. */
export interface RefTarget {
  kind: "ref";
  ref: string;
  resolvedSha: string;
}

/** One side of a comparison, discriminated on `kind`. */
export type Target = InPlaceTarget | RefTarget;

/**
 * A worktree directory this process claimed for a ref, pinned to a SHA.
 *
 * The directory need not exist: `planWorktree` reserves the path before any git
 * runs, and `materializeWorktree` can fail after creating it. Cleanup treats an
 * absent directory as nothing to do rather than as an error.
 */
export interface WorktreeInfo {
  dir: string;
  sha: string;
  /**
   * Whether `git worktree add` ever put this directory on disk.
   *
   * `planWorktree` starts it at `false` and `materializeWorktree` raises it once
   * the add leaves something behind, which is what lets cleanup tell a worktree
   * that was never created from one that was created and has since vanished.
   * Only the latter can leave a registry entry needing a prune.
   */
  created: boolean;
}

/**
 * A worktree directory that survived cleanup, with git's reason for refusing.
 */
export interface WorktreeRemovalFailure {
  dir: string;
  error: string;
}

/**
 * Outcome of a worktree cleanup sweep: what was removed, what survived, and
 * whether the repo-wide prune step failed.
 */
export interface CleanupResult {
  /** Worktrees this call handed to `git worktree remove` successfully. */
  removed: number;
  /** Worktrees left on disk, one entry each. */
  failures: readonly WorktreeRemovalFailure[];
  /**
   * Why the repo-wide `git worktree prune` sweep failed, or `undefined` if it
   * succeeded. Prune runs once per call, not once per worktree, so it gets its
   * own slot rather than a synthetic entry in `failures`.
   */
  pruneError: string | undefined;
}

/** Attempt directory resolution; returns `undefined` to fall through to ref resolution. */
function tryResolveDirectory(absolutePath: string): InPlaceTarget | undefined {
  const stats = fs.statSync(absolutePath, { throwIfNoEntry: false });
  if (stats?.isDirectory()) {
    return {
      kind: "in-place",
      dir: fs.realpathSync(absolutePath),
    };
  }
  return undefined;
}

/**
 * Interpret a user-supplied target as either a directory or a git ref.
 *
 * An existing directory wins over a git ref of the same name, so a branch named
 * after a sibling directory resolves to the directory. Directories are reduced
 * through `realpathSync` so a symlinked target and its destination compare as
 * the same place.
 *
 * @throws when the input is neither an existing directory nor a ref git can verify.
 */
export function resolveTarget(input: string, repoDir: string): Target {
  const directory = tryResolveDirectory(path.resolve(input));
  if (directory !== undefined) {
    return directory;
  }

  try {
    // `^{commit}` peels the ref, so a tag resolves to the commit it points at and
    // a tree or blob sha fails instead of yielding a sha no worktree can check out.
    const resolvedSha = runGit(["rev-parse", "--verify", `${input}^{commit}`], repoDir).trim();
    return {
      kind: "ref",
      ref: input,
      resolvedSha,
    };
  } catch (error) {
    throw new GymratError(
      `Cannot resolve target '${input}': ${stderrTextOf(error)}`,
      "Pass an existing directory, or a git ref that resolves to a commit.",
      { cause: error },
    );
  }
}

/**
 * Choose where a ref target's worktree will live, without touching disk.
 *
 * Deciding the path up front is what lets a caller register the directory for
 * cleanup before git can create it: `git worktree add` can be killed once the
 * worktree is on disk but before it returns, and cleanup only sweeps paths
 * something already names.
 */
export function planWorktree(ref: RefTarget): WorktreeInfo {
  return {
    dir: path.join(fs.realpathSync.native(os.tmpdir()), `gymrat-wt-${crypto.randomUUID()}`),
    sha: ref.resolvedSha,
    created: false,
  };
}

/**
 * Check a planned worktree out into its directory, detached at its pinned SHA.
 *
 * Records on `worktree` whether anything reached disk. Callers register the
 * planned worktree before this runs, so marking the object they already hold is
 * what carries the fact through to cleanup.
 *
 * @throws when `git worktree add` fails — the planned directory may exist anyway,
 * so callers must hand it to `cleanupWorktrees` regardless of the outcome.
 */
export function materializeWorktree(worktree: WorktreeInfo, repoDir: string): void {
  try {
    runGit(["worktree", "add", "--detach", worktree.dir, worktree.sha], repoDir);
  } finally {
    // git registers the worktree before the command returns, and the add can be
    // killed in between, so what landed on disk — not whether git exited zero —
    // is what says a registry entry may exist.
    worktree.created = fs.existsSync(worktree.dir);
  }
}

/**
 * Run a git command, reporting success or failure instead of throwing.
 *
 * @returns `undefined` on success, or git's own stderr text on failure.
 */
function tryGitCommand(args: readonly string[], repoDir: string): string | undefined {
  try {
    runGit(args, repoDir);
    return undefined;
  } catch (error) {
    return stderrTextOf(error);
  }
}

/** Remove a single worktree if it exists on disk; returns the outcome or a failure record. */
function removeWorktreeIfExists(
  worktree: WorktreeInfo,
  repoDir: string,
): "removed" | "absent" | WorktreeRemovalFailure {
  if (!fs.existsSync(worktree.dir)) {
    return "absent";
  }

  const error = tryGitCommand(["worktree", "remove", "--force", worktree.dir], repoDir);
  if (error === undefined) {
    return "removed";
  }
  return { dir: worktree.dir, error };
}

/**
 * Remove any of `worktrees` that reached disk, then prune. Safe to call repeatedly.
 *
 * Never throws: `compare()` calls this while already handling a failed run, so a
 * throw here would replace the error the user actually needs to see. Everything
 * that went wrong lands in the returned result instead, letting callers report
 * it on their own terms.
 */
export function cleanupWorktrees(
  worktrees: readonly WorktreeInfo[],
  repoDir: string,
): CleanupResult {
  const failures: WorktreeRemovalFailure[] = [];
  let removed = 0;
  // A worktree git may still list without a directory backing it: gone before the
  // sweep reached it, or still there because its removal failed.
  let mayHaveStaleEntry = false;

  for (const worktree of worktrees) {
    const outcome = removeWorktreeIfExists(worktree, repoDir);
    if (outcome === "removed") {
      removed += 1;
    } else if (outcome === "absent") {
      // A worktree git never created has no entry to collect, so its absence is
      // no reason to sweep the whole repo and deregister worktrees of the user's
      // own that are only temporarily gone.
      mayHaveStaleEntry ||= worktree.created;
    } else {
      failures.push(outcome);
      mayHaveStaleEntry = true;
    }
  }

  // Prune only when the sweep could have left a registry entry without a
  // directory. `git worktree remove` clears the entry it removes, so a sweep
  // whose removals all succeeded — or that had nothing to remove, in a
  // `repoDir` that may not even be a git repo — has no stale entry to collect.
  // Pruning anyway would reach past this run's worktrees and deregister
  // worktrees of the user's own that are only temporarily absent: an unmounted
  // volume, a directory moved aside.
  const pruneError = mayHaveStaleEntry ? tryGitCommand(["worktree", "prune"], repoDir) : undefined;
  return { removed, failures, pruneError };
}
