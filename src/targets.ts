import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { GymratError, hasErrorCode, stderrTextOf } from "./errors.js";
import { runGit, tryGit } from "./git.js";

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
   * Only the latter can leave a registry entry behind to clear.
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
  /** Worktrees this call took off disk. */
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

/**
 * Probe failures that mean "there is no directory here", not "the probe broke".
 *
 * `throwIfNoEntry: false` already absorbs ENOENT, but a ref name carrying a
 * slash resolves to a path underneath one of its own components — `fix/typo`
 * with a file named `fix` in the working directory — and stat reports that as
 * ENOTDIR. Both leave ref resolution as the input's only remaining reading.
 */
const ABSENT_PATH_CODES = ["ENOENT", "ENOTDIR"] as const;

function isAbsentPathError(error: unknown): boolean {
  return ABSENT_PATH_CODES.some((code) => hasErrorCode(error, code));
}

/** The failure a target the tool cannot make sense of is reported as. */
function unresolvableTarget(input: string, cause: unknown): GymratError {
  return new GymratError(
    `Cannot resolve target '${input}': ${stderrTextOf(cause)}`,
    "Pass an existing directory, or a git ref that resolves to a commit.",
    { cause },
  );
}

/**
 * Attempt directory resolution; returns `undefined` to fall through to ref resolution.
 *
 * @throws when the probe fails for a reason other than the path being absent —
 * a symlink loop or an unsearchable parent says nothing about whether the input
 * is a ref, so it is reported rather than silently retried as one.
 */
function tryResolveDirectory(input: string): InPlaceTarget | undefined {
  const absolutePath = path.resolve(input);

  try {
    const stats = fs.statSync(absolutePath, { throwIfNoEntry: false });
    if (stats?.isDirectory()) {
      return {
        kind: "in-place",
        dir: fs.realpathSync(absolutePath),
      };
    }
    return undefined;
  } catch (error) {
    if (isAbsentPathError(error)) {
      return undefined;
    }
    throw unresolvableTarget(input, error);
  }
}

/**
 * Interpret a user-supplied target as either a directory or a git ref.
 *
 * An existing directory wins over a git ref of the same name, so a branch named
 * after a sibling directory resolves to the directory. Directories are reduced
 * through `realpathSync` so a symlinked target and its destination compare as
 * the same place.
 *
 * @throws when the input is neither an existing directory nor a ref git can
 * verify, and when the directory probe itself fails.
 */
export function resolveTarget(input: string, repoDir: string): Target {
  const directory = tryResolveDirectory(input);
  if (directory !== undefined) {
    return directory;
  }

  try {
    // `--end-of-options` prevents a leading-dash input from being parsed as a
    // git option. `^{commit}` peels the ref, so a tag resolves to the commit it
    // points at and a tree or blob sha fails instead of yielding a sha no
    // worktree can check out.
    const resolvedSha = runGit(
      ["rev-parse", "--verify", "--end-of-options", `${input}^{commit}`],
      repoDir,
    ).trim();
    return {
      kind: "ref",
      ref: input,
      resolvedSha,
    };
  } catch (error) {
    throw unresolvableTarget(input, error);
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
 * What handing one worktree to git accomplished.
 *
 * `deregistered` and `stale` both describe a directory that vanished behind git's
 * back: the entry git still listed either went with the targeted removal or may
 * have survived it, and only the latter leaves anything for a prune to collect.
 * `untouched` is a worktree `git worktree add` never put on disk, which has
 * neither a directory nor an entry.
 */
type RemovalOutcome = "removed" | "deregistered" | "stale" | "untouched" | WorktreeRemovalFailure;

/**
 * Take one worktree off disk, or clear the entry left behind if it is already gone.
 *
 * The removal names the worktree's own path instead of sweeping the repository,
 * because git clears the entry of a directory that vanished behind its back when
 * it is asked for that path — which leaves a worktree of the user's own that is
 * only temporarily absent registered.
 */
function removeWorktree(worktree: WorktreeInfo, repoDir: string): RemovalOutcome {
  const onDisk = fs.existsSync(worktree.dir);
  if (!onDisk && !worktree.created) {
    return "untouched";
  }

  const error = tryGit(["worktree", "remove", "--force", worktree.dir], repoDir);
  if (error !== undefined) {
    // Nothing is standing for the user to clear by hand when the directory is
    // already gone, so a refusal there is a reason to sweep, not to report.
    return onDisk ? { dir: worktree.dir, error } : "stale";
  }
  // The entry is gone — mark the worktree so a subsequent sweep treats it as
  // untouched rather than reclassifying it as stale.
  worktree.created = false;
  return onDisk ? "removed" : "deregistered";
}

/**
 * Remove each of `worktrees` git has anything for. Safe to call repeatedly.
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
    const outcome = removeWorktree(worktree, repoDir);
    switch (outcome) {
      case "removed":
        removed += 1;
        break;
      case "stale":
        mayHaveStaleEntry = true;
        break;
      case "deregistered":
      case "untouched":
        break;
      default:
        failures.push(outcome);
    }
  }

  // Prune only when a targeted removal failed and may have left an entry with no
  // directory behind it. Naming each worktree already clears its own entry, so a
  // sweep whose removals all succeeded — or that had nothing to ask git for, in a
  // `repoDir` that may not even be a git repo — has nothing left to collect.
  // Pruning anyway would reach past this run's worktrees and deregister
  // worktrees of the user's own that are only temporarily absent: an unmounted
  // volume, a directory moved aside.
  const pruneError = mayHaveStaleEntry ? tryGit(["worktree", "prune"], repoDir) : undefined;
  return { removed, failures, pruneError };
}
