import { execSync, type StdioOptions } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export interface InPlaceTarget {
  kind: "in-place";
  dir: string;
}

export interface RefTarget {
  kind: "ref";
  ref: string;
  resolvedSha: string;
}

export type Target = InPlaceTarget | RefTarget;

export interface WorktreeInfo {
  dir: string;
  ref: string;
}

const GIT_STDIO: StdioOptions = ["pipe", "pipe", "pipe"];

/**
 * Run a git command with suppressed output.
 */
function runGitCommand(command: string, repoDir: string): string {
  return execSync(command, {
    cwd: repoDir,
    encoding: "utf-8",
    stdio: GIT_STDIO,
  });
}

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
    /* v8 ignore next 3 -- non-fs errors from statSync are not reproducible in tests */
    if (!isFsError) {
      throw error;
    }
  }

  // Try to resolve as a git ref
  try {
    const resolvedSha = runGitCommand(`git rev-parse --verify "${input}"`, repoDir).trim();

    return {
      kind: "ref",
      ref: input,
      resolvedSha,
    };
  } catch {
    // Neither a directory nor a valid git ref
    throw new Error(`Cannot resolve target '${input}': not an existing directory or valid git ref`);
  }
}

/**
 * Create a detached worktree for a ref target under os.tmpdir().
 */
export function createWorktree(ref: RefTarget, repoDir: string): WorktreeInfo {
  const worktreePath = path.join(os.tmpdir(), `gymrat-wt-${crypto.randomUUID()}`);

  runGitCommand(`git worktree add --detach "${worktreePath}" "${ref.resolvedSha}"`, repoDir);

  return {
    dir: worktreePath,
    ref: ref.ref,
  };
}

/**
 * Remove created worktrees and prune. Safe to call multiple times.
 */
export function cleanupWorktrees(worktrees: WorktreeInfo[], repoDir: string): void {
  for (const worktree of worktrees) {
    try {
      if (fs.existsSync(worktree.dir)) {
        runGitCommand(`git worktree remove --force "${worktree.dir}"`, repoDir);
      }
    } catch {
      // Silently ignore if worktree removal fails (already removed or doesn't exist)
    }
  }

  // Prune dangling worktree metadata
  try {
    runGitCommand("git worktree prune", repoDir);
  } catch {
    // Silently ignore if prune fails
  }
}
