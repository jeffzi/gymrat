import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/** Throwaway git repository in the system temp dir. Call `cleanup` to remove it. */
export interface ScratchRepo {
  dir: string;
  cleanup: () => void;
}

/**
 * Create a temporary git repository on `main` with one committed file,
 * user identity configured, and GPG signing disabled.
 *
 * The caller must invoke `.cleanup()` when done.
 */
export function createScratchRepo(): ScratchRepo {
  // realpathSync.native resolves Windows 8.3 short names to their long form
  // so the path matches what git reports in `worktree list`.
  const dir = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-test-")));

  try {
    // -b main pins the initial branch: without it the name comes from the
    // developer's init.defaultBranch, and tests that check out from "main" break.
    execFileSync("git", ["init", "-b", "main"], { cwd: dir, stdio: "pipe" });

    const configs: [string, string][] = [
      ["user.name", "Test User"],
      ["user.email", "test@example.com"],
      ["commit.gpgsign", "false"],
      ["core.autocrlf", "false"],
    ];
    for (const [key, value] of configs) {
      execFileSync("git", ["config", key, value], { cwd: dir, stdio: "pipe" });
    }

    fs.writeFileSync(path.join(dir, "README.md"), "# Test Repo\n");
    execFileSync("git", ["add", "README.md"], { cwd: dir, stdio: "pipe" });
    execFileSync("git", ["commit", "-m", "Initial commit"], { cwd: dir, stdio: "pipe" });
  } catch (error) {
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    throw error;
  }

  return {
    dir,
    cleanup: () => {
      // Windows cannot delete a directory that is a process's CWD.
      if (process.cwd().startsWith(dir)) {
        process.chdir(os.tmpdir());
      }
      fs.rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    },
  };
}

/**
 * Make every later `git worktree add` in `repoDir` die once the worktree is on disk.
 *
 * Reproduces the one state a run can strand: a worktree on disk whose
 * `git worktree add` never returned success, so a caller that waits for the
 * command before recording the path is left with nothing to clean up.
 *
 * `post-checkout` fires after git has laid the worktree down and registered it,
 * so what survives the kill is a *complete* worktree — `.git` file present, admin
 * entry unlocked, `git worktree remove --force` able to take it — not a
 * half-written one. Git removes its own junk on every failure it survives, so a
 * genuinely partial worktree is not observable from the outside; an interrupted
 * one is.
 *
 * Branch checkouts fire the hook too — install it after the repo's branches exist.
 */
export function killGitDuringWorktreeAdd(repoDir: string): void {
  const hookPath = path.join(repoDir, ".git", "hooks", "post-checkout");
  fs.mkdirSync(path.dirname(hookPath), { recursive: true });
  // The redirect keeps the hook from holding git's stdio open past the kill.
  fs.writeFileSync(hookPath, '#!/bin/sh\nexec >/dev/null 2>&1\nkill -9 "$PPID"\nsleep 1\n');
  fs.chmodSync(hookPath, 0o755);
}

/**
 * Directories git currently lists as worktrees of `repoDir`, main worktree
 * included by default.
 *
 * `git worktree remove` clears a worktree's registry entry itself, so this
 * only says something about pruning when the directory vanished behind git's
 * back — that is the one case where a stale entry survives until `prune`
 * runs.
 *
 * Pass `{ includeMain: false }` to exclude the main worktree's own
 * directory. git prints resolved paths, so on macOS — where the scratch repo
 * lives under `/var/...` but git reports `/private/var/...` — the main
 * directory is matched through `realpath`.
 */
export function listWorktreeDirs(repoDir: string, options?: { includeMain?: boolean }): string[] {
  const dirs = execFileSync("git", ["worktree", "list", "--porcelain"], {
    cwd: repoDir,
    stdio: "pipe",
    encoding: "utf-8",
  })
    .split("\n")
    .filter((line) => line.startsWith("worktree "))
    .map((line) => path.normalize(line.slice("worktree ".length)));

  if (options?.includeMain === false) {
    const mainDir = fs.realpathSync(repoDir);
    return dirs.filter((dir) => dir !== mainDir);
  }
  return dirs;
}
