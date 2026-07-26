import { execFileSync, type StdioOptions } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

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
  ref: string;
  sha: string;
}

/**
 * A worktree directory that survived cleanup, with git's reason for refusing.
 */
export interface WorktreeRemovalFailure {
  dir: string;
  error: string;
}

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
  // Try to resolve as an existing directory first
  const absolutePath = path.resolve(input);
  try {
    const stats = fs.statSync(absolutePath);
    if (stats.isDirectory()) {
      return {
        kind: "in-place",
        dir: fs.realpathSync(absolutePath),
      };
    }
  } catch (error) {
    // Only catch fs errors (like ENOENT); rethrow other errors
    const isFsError = error instanceof Error && "code" in error;
    /* v8 ignore if -- non-fs errors from statSync are not reproducible in tests */
    if (!isFsError) {
      throw error;
    }
  }

  // Try to resolve as a git ref
  try {
    const resolvedSha = runGitCommand(["rev-parse", "--verify", input], repoDir).trim();

    return {
      kind: "ref",
      ref: input,
      resolvedSha,
    };
  } catch {
    throw new Error(`Cannot resolve target '${input}': not an existing directory or valid git ref`);
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
    dir: path.join(os.tmpdir(), `gymrat-wt-${crypto.randomUUID()}`),
    ref: ref.ref,
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
function gitErrorText(error: unknown): string {
  if (error instanceof Error && "stderr" in error && typeof error.stderr === "string") {
    return error.stderr.trim();
  }

  /* v8 ignore next 2 -- spawn-level failures (git missing from PATH, repoDir
     deleted) throw before the child runs, leaving stderr null; the test harness
     cannot reproduce those without breaking the environment it runs in. */
  return error instanceof Error ? error.message : String(error);
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

  for (const worktree of worktrees) {
    // An already-gone directory was cleaned up by an earlier pass: it is
    // neither a removal this call performed nor a leftover to report.
    if (!fs.existsSync(worktree.dir)) {
      continue;
    }

    // Keep going — one stuck worktree must not strand the rest.
    const error = tryGitCommand(["worktree", "remove", "--force", worktree.dir], repoDir);
    if (error === undefined) {
      removed += 1;
    } else {
      failures.push({ dir: worktree.dir, error });
    }
  }

  // An empty list means no ref target was ever attempted, so there is nothing to
  // prune — and no reason to assume `repoDir` is a git repository at all. Both
  // targets can be plain directories, in which case sweeping would fail with
  // "not a git repository" and report cleanup trouble for work never done. A
  // planned worktree counts even when `git worktree add` never produced one:
  // that is exactly the run whose $GIT_DIR/worktrees/ metadata needs sweeping.
  const pruneError =
    worktrees.length > 0 ? tryGitCommand(["worktree", "prune"], repoDir) : undefined;

  return { removed, failures, pruneError };
}
