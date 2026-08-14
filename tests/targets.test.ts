import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterEach } from "vitest";

import { GymratError, messageOf } from "../src/errors.js";
import type { CleanupResult, InPlaceTarget, RefTarget, WorktreeInfo } from "../src/targets.js";
import {
  resolveTarget,
  planWorktree,
  materializeWorktree,
  cleanupWorktrees,
} from "../src/targets.js";
import { captureGymratError, captureThrown } from "./fixtures/errors.js";
import {
  createScratchRepo,
  killGitDuringWorktreeAdd,
  listWorktreeDirs,
  registerAbsentWorktree,
} from "./fixtures/scratch-repo.js";

/** A sha no repository holds, so `git worktree add` rejects it outright. */
const UNKNOWN_SHA = "0".repeat(40);

/** Hint gymrat attaches to every unresolvable target, duplicated here so the test asserts against the same string production emits. */
const RESOLVE_TARGET_HINT = "Pass an existing directory, or a git ref that resolves to a commit.";

function getHeadSha(repoDir: string): string {
  return execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoDir,
    encoding: "utf-8",
  }).trim();
}

function createRefTarget(ref: string, resolvedSha: string): RefTarget {
  return { kind: "ref", ref, resolvedSha };
}

function createHeadWorktree(repoDir: string): WorktreeInfo {
  const sha = getHeadSha(repoDir);
  const worktree = planWorktree(createRefTarget(sha, sha));
  materializeWorktree(worktree, repoDir);
  return worktree;
}

/**
 * Plan a worktree for `target` and attempt to materialize it, reporting whether
 * `git worktree add` failed instead of throwing.
 */
function planAndAttemptMaterialize(
  target: RefTarget,
  repoDir: string,
): { worktree: WorktreeInfo; failed: boolean } {
  const worktree = planWorktree(target);
  try {
    materializeWorktree(worktree, repoDir);
    return { worktree, failed: false };
  } catch {
    return { worktree, failed: true };
  }
}

/**
 * Leave behind the worktree a killed `git worktree add` never returned success for.
 *
 * Throws rather than returning a half-arranged fixture, so a git version that
 * cleaned up despite the kill fails the test that asked for this state instead
 * of quietly asserting nothing. The directory is removed on that path so a
 * failed arrangement cannot strand one in the system temp dir.
 */
function leaveInterruptedWorktree(repoDir: string): WorktreeInfo {
  killGitDuringWorktreeAdd(repoDir);
  const sha = getHeadSha(repoDir);
  const { worktree, failed } = planAndAttemptMaterialize(createRefTarget(sha, sha), repoDir);

  if (failed && fs.existsSync(worktree.dir)) {
    return worktree;
  }
  fs.rmSync(worktree.dir, { recursive: true, force: true, maxRetries: 3 });
  throw new Error(`expected an interrupted worktree left at ${worktree.dir}`);
}

/**
 * Plan a worktree whose `git worktree add` fails before creating anything.
 */
function planRejectedWorktree(repoDir: string): WorktreeInfo {
  const { worktree, failed } = planAndAttemptMaterialize(
    createRefTarget("missing", UNKNOWN_SHA),
    repoDir,
  );

  if (failed && !fs.existsSync(worktree.dir)) {
    return worktree;
  }
  throw new Error(`expected 'git worktree add' to create nothing at ${worktree.dir}`);
}

describe("resolveTarget", () => {
  describe("when input is an existing directory", () => {
    it("returns InPlaceTarget with dir set to absolute path", () => {
      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-test-"));

      try {
        const result = resolveTarget(tempDir, os.tmpdir());

        expect(result).toStrictEqual({
          kind: "in-place",
          dir: fs.realpathSync(tempDir),
        } satisfies InPlaceTarget);
      } finally {
        fs.rmSync(tempDir, { recursive: true, force: true, maxRetries: 3 });
      }
    });

    it("resolves relative directory paths to absolute", () => {
      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-test-"));
      const cwd = process.cwd();

      try {
        process.chdir(path.dirname(tempDir));
        const relativePath = path.basename(tempDir);

        const result = resolveTarget(relativePath, os.tmpdir());

        expect(result).toStrictEqual({
          kind: "in-place",
          dir: fs.realpathSync(tempDir),
        } satisfies InPlaceTarget);
      } finally {
        process.chdir(cwd);
        fs.rmSync(tempDir, { recursive: true, force: true, maxRetries: 3 });
      }
    });
  });

  describe("when input is a valid git ref", () => {
    let repo: ReturnType<typeof createScratchRepo> | undefined;

    afterEach(() => {
      repo?.cleanup();
    });

    it("resolves commit SHA to RefTarget", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);

      const result = resolveTarget(sha, repo.dir);

      expect(result).toStrictEqual(createRefTarget(sha, sha) satisfies RefTarget);
    });

    it("resolves branch name to RefTarget", () => {
      repo = createScratchRepo();
      const mainSha = getHeadSha(repo.dir);

      const result = resolveTarget("HEAD", repo.dir);

      expect(result).toStrictEqual(createRefTarget("HEAD", mainSha) satisfies RefTarget);
    });

    it("resolves tag to RefTarget", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);

      execFileSync("git", ["tag", "v1.0.0"], { cwd: repo.dir, stdio: "pipe" });

      const result = resolveTarget("v1.0.0", repo.dir);

      expect(result).toStrictEqual(createRefTarget("v1.0.0", sha) satisfies RefTarget);
    });
  });

  describe("when input is neither a directory nor a valid git ref", () => {
    let repo: ReturnType<typeof createScratchRepo> | undefined;

    afterEach(() => {
      repo?.cleanup();
    });

    it("throws error with message naming the input", () => {
      repo = createScratchRepo();

      expect(() => resolveTarget("nonexistent-ref-xyz", repo!.dir)).toThrow(
        /Cannot resolve target 'nonexistent-ref-xyz'/,
      );
    });

    it("carries git's own error text so the reason is not lost", () => {
      repo = createScratchRepo();

      const error = captureThrown(() => resolveTarget("definitely-not-a-ref", repo!.dir));

      expect.soft(error).toBeInstanceOf(GymratError);
      // git words the failure differently depending on the flags rev-parse is
      // given, so match only the `fatal:` prefix every wording carries.
      expect(messageOf(error)).toMatch(/fatal:/);
    });
  });

  describe("when input is the sha of a non-commit object", () => {
    let repo: ReturnType<typeof createScratchRepo> | undefined;

    afterEach(() => {
      repo?.cleanup();
    });

    it.each([
      { objectKind: "tree", rev: "HEAD^{tree}" },
      { objectKind: "blob", rev: "HEAD:README.md" },
    ])("rejects a $objectKind sha", ({ rev }) => {
      repo = createScratchRepo();
      const sha = execFileSync("git", ["rev-parse", rev], {
        cwd: repo.dir,
        encoding: "utf-8",
      }).trim();

      expect(() => resolveTarget(sha, repo!.dir)).toThrow(/Cannot resolve target/);
    });
  });

  // Windows cannot create the symlink without elevation, and its permission
  // model does not produce EACCES from chmod; root bypasses the mode bits
  // entirely, so the unreadable directory would resolve instead of failing.
  describe.skipIf(process.platform === "win32" || process.getuid?.() === 0)(
    "when the directory probe fails for a reason other than the path being absent",
    () => {
      let repo: ReturnType<typeof createScratchRepo> | undefined;
      const cleanups: (() => void)[] = [];

      afterEach(() => {
        while (cleanups.length > 0) {
          cleanups.pop()?.();
        }
        repo?.cleanup();
      });

      /** A symlink pointing at itself: stat on it fails with ELOOP. */
      function createSymlinkLoop(): string {
        const base = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-loop-"));
        cleanups.push(() => {
          fs.rmSync(base, { recursive: true, force: true, maxRetries: 3 });
        });

        const loop = path.join(base, "loop");
        fs.symlinkSync(loop, loop);
        return loop;
      }

      /** A directory whose parent denies traversal: stat on it fails with EACCES. */
      function createUnsearchableParent(): string {
        const base = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-eacces-"));
        const parent = path.join(base, "parent");
        const target = path.join(parent, "target");
        fs.mkdirSync(target, { recursive: true });
        fs.chmodSync(parent, 0o000);
        cleanups.push(() => {
          fs.chmodSync(parent, 0o700);
          fs.rmSync(base, { recursive: true, force: true, maxRetries: 3 });
        });

        return target;
      }

      it.each([
        { probeFailure: "a symlink loop", code: "ELOOP", arrange: createSymlinkLoop },
        {
          probeFailure: "an unsearchable parent directory",
          code: "EACCES",
          arrange: createUnsearchableParent,
        },
      ])("reports $probeFailure as the documented resolve failure", ({ code, arrange }) => {
        repo = createScratchRepo();
        const input = arrange();

        const error = captureGymratError(() => resolveTarget(input, repo!.dir));

        expect.soft(error.message).toContain(`Cannot resolve target '${input}'`);
        expect.soft(error.message).toContain(code);
        expect(error.hint).toBe(RESOLVE_TARGET_HINT);
      });
    },
  );
});

describe("planWorktree", () => {
  describe("when given a RefTarget", () => {
    it("names an absolute directory under os.tmpdir() without creating it", () => {
      const refTarget = createRefTarget("my-tag", UNKNOWN_SHA);

      const worktree = planWorktree(refTarget);

      expect(worktree.sha).toBe(UNKNOWN_SHA);
      expect(path.dirname(worktree.dir)).toBe(fs.realpathSync.native(os.tmpdir()));
      expect(fs.existsSync(worktree.dir)).toBe(false);
    });
  });
});

describe("materializeWorktree", () => {
  let repo: ReturnType<typeof createScratchRepo> | undefined;
  const createdWorktrees: WorktreeInfo[] = [];

  afterEach(() => {
    if (createdWorktrees.length > 0 && repo) {
      cleanupWorktrees(createdWorktrees, repo.dir);
      createdWorktrees.length = 0;
    }
    repo?.cleanup();
  });

  describe("when given a planned worktree", () => {
    it("checks out the ref's files into the planned directory", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const worktree = planWorktree(createRefTarget(sha, sha));
      createdWorktrees.push(worktree);

      materializeWorktree(worktree, repo.dir);

      expect(fs.readFileSync(path.join(worktree.dir, "README.md"), "utf-8")).toBe("# Test Repo\n");
    });
  });
});

describe("cleanupWorktrees", () => {
  let repo: ReturnType<typeof createScratchRepo> | undefined;

  afterEach(() => {
    repo?.cleanup();
  });

  describe("when given non-empty worktree list", () => {
    it("removes each directory and reports it as removed", () => {
      repo = createScratchRepo();
      const worktree = createHeadWorktree(repo.dir);

      const result = cleanupWorktrees([worktree], repo.dir);

      expect(result).toStrictEqual({
        removed: 1,
        failures: [],
        pruneError: undefined,
      } satisfies CleanupResult);
      expect(fs.existsSync(worktree.dir)).toBe(false);
    });
  });

  describe("when given empty worktree list", () => {
    it("leaves the git registry untouched", () => {
      repo = createScratchRepo();

      cleanupWorktrees([], repo.dir);

      expect(listWorktreeDirs(repo.dir)).toHaveLength(1);
    });
  });

  describe("when given empty worktree list outside a git repository", () => {
    let nonRepoDir: string | undefined;

    it("skips the prune sweep and reports no error", () => {
      nonRepoDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-not-a-repo-"));

      const result = cleanupWorktrees([], nonRepoDir);

      expect(result).toStrictEqual({
        removed: 0,
        failures: [],
        pruneError: undefined,
      } satisfies CleanupResult);
    });
  });

  // killGitDuringWorktreeAdd sends POSIX signal 9 via a post-checkout hook;
  // Windows cannot deliver that signal to the git parent process.
  describe.skipIf(process.platform === "win32")(
    "when git worktree add was killed after creating the worktree",
    () => {
      it("removes it like a worktree whose add returned normally", () => {
        repo = createScratchRepo();
        const worktree = leaveInterruptedWorktree(repo.dir);

        try {
          const result = cleanupWorktrees([worktree], repo.dir);

          expect(result.removed).toBe(1);
          expect(fs.existsSync(worktree.dir)).toBe(false);
        } finally {
          fs.rmSync(worktree.dir, { recursive: true, force: true, maxRetries: 3 });
        }
      });
    },
  );

  describe("when git worktree add left nothing on disk", () => {
    it("counts the planned worktree as neither removed nor left behind", () => {
      repo = createScratchRepo();
      const worktree = planRejectedWorktree(repo.dir);

      const result = cleanupWorktrees([worktree], repo.dir);

      expect(result).toStrictEqual({
        removed: 0,
        failures: [],
        pruneError: undefined,
      } satisfies CleanupResult);
    });

    it("leaves the registry entries of worktrees that vanished behind git's back", () => {
      repo = createScratchRepo();
      const absent = registerAbsentWorktree(repo.dir);
      const worktree = planRejectedWorktree(repo.dir);

      const result = cleanupWorktrees([worktree], repo.dir);

      expect.soft(result.removed).toBe(0);
      expect(listWorktreeDirs(repo.dir)).toContain(absent);
    });
  });

  describe("when every worktree removal succeeds", () => {
    it("leaves the registry entries of worktrees it was not asked about", () => {
      repo = createScratchRepo();
      const absent = registerAbsentWorktree(repo.dir);
      const worktree = createHeadWorktree(repo.dir);

      const result = cleanupWorktrees([worktree], repo.dir);

      expect.soft(result.removed).toBe(1);
      expect(listWorktreeDirs(repo.dir)).toContain(absent);
    });
  });

  describe("when a worktree directory no longer exists", () => {
    it("deregisters that worktree by name and no other", () => {
      repo = createScratchRepo();
      const absent = registerAbsentWorktree(repo.dir);
      const worktree = createHeadWorktree(repo.dir);
      fs.rmSync(worktree.dir, { recursive: true, force: true, maxRetries: 3 });

      cleanupWorktrees([worktree], repo.dir);

      const listed = listWorktreeDirs(repo.dir);
      expect.soft(listed).not.toContain(worktree.dir);
      expect(listed).toContain(absent);
    });
  });

  describe("when git cannot remove a worktree", () => {
    let strayDir: string | undefined;

    function createStrayWorktree(): WorktreeInfo {
      strayDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-stray-"));
      return { dir: strayDir, sha: UNKNOWN_SHA, created: true };
    }

    it("reports the failing directory with git's own error text", () => {
      repo = createScratchRepo();
      const stray = createStrayWorktree();

      const result = cleanupWorktrees([stray], repo.dir);

      expect(result.failures.map((failure) => failure.dir)).toStrictEqual([stray.dir]);
      expect(result.failures[0]?.error).toContain("is not a working tree");
      expect(result.failures[0]?.error).not.toContain("Command failed");
    });

    it("prunes the registry entries of worktrees whose directories have vanished", () => {
      repo = createScratchRepo();
      const absent = registerAbsentWorktree(repo.dir);
      const stray = createStrayWorktree();

      const result = cleanupWorktrees([stray], repo.dir);

      expect.soft(result.failures).toHaveLength(1);
      expect(listWorktreeDirs(repo.dir)).not.toContain(absent);
    });

    it("still removes the worktrees listed after the failing one", () => {
      repo = createScratchRepo();
      const stray = createStrayWorktree();
      const worktree = createHeadWorktree(repo.dir);

      const result = cleanupWorktrees([stray, worktree], repo.dir);

      expect(result.removed).toBe(1);
      expect(fs.existsSync(worktree.dir)).toBe(false);
    });
  });

  describe("when the pruning sweep fails", () => {
    let nonRepoDir: string | undefined;

    afterEach(() => {
      if (nonRepoDir !== undefined) {
        fs.rmSync(nonRepoDir, { recursive: true, force: true, maxRetries: 3 });
        nonRepoDir = undefined;
      }
    });

    it("reports the prune error instead of throwing", () => {
      nonRepoDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-not-a-repo-"));
      // A vanished directory is skipped before removal, so the sweep is the only
      // git call this list reaches — and it runs against a non-repo.
      const vanished: WorktreeInfo = {
        dir: path.join(nonRepoDir, "gone"),
        sha: UNKNOWN_SHA,
        created: true,
      };

      const result = cleanupWorktrees([vanished], nonRepoDir);

      expect(result.removed).toBe(0);
      expect(result.failures).toStrictEqual([]);
      expect(result.pruneError).toContain("not a git repository");
    });
  });

  describe("idempotency", () => {
    it("reports removed: 0 and no failures on a second sweep of the same worktrees", () => {
      repo = createScratchRepo();
      const worktree = createHeadWorktree(repo.dir);

      cleanupWorktrees([worktree], repo.dir);
      const second = cleanupWorktrees([worktree], repo.dir);

      expect.soft(fs.existsSync(worktree.dir)).toBe(false);
      expect.soft(listWorktreeDirs(repo.dir)).toHaveLength(1);
      expect.soft(second.removed).toBe(0);
      expect(second.failures).toStrictEqual([]);
    });
  });
});
