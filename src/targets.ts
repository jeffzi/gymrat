import { execFileSync, type StdioOptions } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { messageOf } from "./errors.js";

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

const GIT_STDIO: StdioOptions = ["pipe", "pipe", "pipe"];

/**
 * Run a git command with suppressed output.
 *
 * Arguments are passed as an array rather than a shell string so that refs and
 * paths containing shell metacharacters are treated as literal git arguments.
 */
function runGitCommand(args: readonly string[], repoDir: string): string {
  return execFileSync("git", args, {
    cwd: repoDir,
    encoding: "utf-8",
    stdio: GIT_STDIO,
  });
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
    const resolvedSha = runGitCommand(["rev-parse", "--verify", input], repoDir).trim();
    return {
      kind: "ref",
      ref: input,
      resolvedSha,
    };
  } catch (error) {
    throw new Error(
      `Cannot resolve target '${input}': not an existing directory or valid git ref`,
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
  };
}

/**
 * Check a planned worktree out into its directory, detached at its pinned SHA.
 *
 * @throws when `git worktree add` fails — the planned directory may exist anyway,
 * so callers must hand it to `cleanupWorktrees` regardless of the outcome.
 */
export function materializeWorktree(worktree: WorktreeInfo, repoDir: string): void {
  runGitCommand(["worktree", "add", "--detach", worktree.dir, worktree.sha], repoDir);
}

/**
 * Git's own diagnostics for a failed `runGitCommand` call.
 *
 * `execFileSync` attaches the child's piped stderr to the thrown error,
 * separate from `message`, which it prefixes with `Command failed: git ...`
 * noise. Reporting stderr matches how `compare.ts` builds user-facing text.
 */
function hasStderr(error: unknown): error is Error & { stderr: string } {
  return error instanceof Error && "stderr" in error && typeof error.stderr === "string";
}

function gitErrorText(error: unknown): string {
  if (hasStderr(error)) {
    return error.stderr.trim();
  }

  /* v8 ignore next 2 -- spawn-level failures (git missing from PATH, repoDir
     deleted) throw before the child runs, leaving stderr null; the test harness
     cannot reproduce those without breaking the environment it runs in. */
  return messageOf(error);
}

/**
 * Run a git command, reporting success or failure instead of throwing.
 *
 * @returns `undefined` on success, or git's own stderr text on failure.
 */
function tryGitCommand(args: readonly string[], repoDir: string): string | undefined {
  try {
    runGitCommand(args, repoDir);
    return undefined;
  } catch (error) {
    return gitErrorText(error);
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
 * Prune only when the sweep could have left a registry entry without a directory.
 *
 * `git worktree remove` clears the entry it removes, so a sweep whose removals all
 * succeeded — or that had nothing to remove, in a `repoDir` that may not even be a
 * git repo — has no stale entry to collect. Pruning anyway would reach past this
 * run's worktrees and deregister worktrees of the user's own that are only
 * temporarily absent: an unmounted volume, a directory moved aside.
 */
function pruneIfNeeded(mayHaveStaleEntry: boolean, repoDir: string): string | undefined {
  if (!mayHaveStaleEntry) return undefined;
  return tryGitCommand(["worktree", "prune"], repoDir);
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
      mayHaveStaleEntry = true;
    } else {
      failures.push(outcome);
      mayHaveStaleEntry = true;
    }
  }

  return { removed, failures, pruneError: pruneIfNeeded(mayHaveStaleEntry, repoDir) };
}
