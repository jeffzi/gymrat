import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterEach } from "vitest";

import type { CleanupResult, InPlaceTarget, RefTarget, WorktreeInfo } from "../src/targets.js";
import { resolveTarget, createWorktree, cleanupWorktrees } from "../src/targets.js";
import { createScratchRepo } from "./fixtures/scratch-repo.js";

/**
 * Get the commit SHA of the working tree HEAD.
 */
function getHeadSha(repoDir: string): string {
  return execSync("git rev-parse HEAD", {
    cwd: repoDir,
    encoding: "utf-8",
  }).trim();
}

/**
 * Create a RefTarget with the given ref and resolved SHA.
 */
function createRefTarget(ref: string, resolvedSha: string): RefTarget {
  return { kind: "ref", ref, resolvedSha };
}

/**
 * Create a worktree checked out at the repo's current HEAD.
 */
function createHeadWorktree(repoDir: string): WorktreeInfo {
  const sha = getHeadSha(repoDir);
  return createWorktree(createRefTarget(sha, sha), repoDir);
}

/**
 * Directories git still lists as worktrees of the repo, main worktree included.
 *
 * `git worktree remove` clears a worktree's registry entry itself, so this only
 * says something about pruning when the directory vanished behind git's back —
 * that is the one case where a stale entry survives until `prune` runs.
 */
function listWorktreeDirs(repoDir: string): string[] {
  return execSync("git worktree list --porcelain", {
    cwd: repoDir,
    encoding: "utf-8",
  })
    .split("\n")
    .filter((line) => line.startsWith("worktree "))
    .map((line) => line.slice("worktree ".length));
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
        fs.rmSync(tempDir, { recursive: true, force: true });
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
        fs.rmSync(tempDir, { recursive: true, force: true });
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

      execSync("git tag v1.0.0", { cwd: repo.dir, stdio: "pipe" });

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
  });
});

describe("createWorktree", () => {
  let repo: ReturnType<typeof createScratchRepo> | undefined;
  const createdWorktrees: WorktreeInfo[] = [];

  afterEach(() => {
    if (createdWorktrees.length > 0 && repo) {
      cleanupWorktrees(createdWorktrees, repo.dir);
      createdWorktrees.length = 0;
    }
    repo?.cleanup();
  });

  describe("when given a RefTarget", () => {
    it("returns a WorktreeInfo naming an absolute directory under os.tmpdir()", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget("my-tag", sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

      expect(worktree.ref).toBe("my-tag");
      expect(path.isAbsolute(worktree.dir)).toBe(true);
      expect(worktree.dir).toContain(os.tmpdir());
    });

    it("checks out the ref's files into the worktree directory", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

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
    it("removes worktree directories", () => {
      repo = createScratchRepo();
      const worktree = createHeadWorktree(repo.dir);

      cleanupWorktrees([worktree], repo.dir);

      expect(fs.existsSync(worktree.dir)).toBe(false);
    });

    it("reports every worktree as removed with no failures", () => {
      repo = createScratchRepo();
      const worktree = createHeadWorktree(repo.dir);

      const result = cleanupWorktrees([worktree], repo.dir);

      expect(result).toStrictEqual({
        removed: 1,
        failures: [],
        pruneError: undefined,
      } satisfies CleanupResult);
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

    afterEach(() => {
      if (nonRepoDir !== undefined) {
        fs.rmSync(nonRepoDir, { recursive: true, force: true });
        nonRepoDir = undefined;
      }
    });

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

  describe("when a worktree directory no longer exists", () => {
    it("counts it as neither removed nor left behind", () => {
      repo = createScratchRepo();
      const worktree = createHeadWorktree(repo.dir);
      cleanupWorktrees([worktree], repo.dir);

      const result = cleanupWorktrees([worktree], repo.dir);

      expect(result).toStrictEqual({
        removed: 0,
        failures: [],
        pruneError: undefined,
      } satisfies CleanupResult);
    });

    it("prunes the registry entry git still lists for it", () => {
      repo = createScratchRepo();
      const worktree = createHeadWorktree(repo.dir);
      fs.rmSync(worktree.dir, { recursive: true, force: true });

      cleanupWorktrees([worktree], repo.dir);

      expect(listWorktreeDirs(repo.dir)).toStrictEqual([fs.realpathSync(repo.dir)]);
    });
  });

  describe("when git cannot remove a worktree", () => {
    let strayDir: string | undefined;

    afterEach(() => {
      if (strayDir !== undefined) {
        fs.rmSync(strayDir, { recursive: true, force: true });
        strayDir = undefined;
      }
    });

    function createStrayWorktree(): WorktreeInfo {
      strayDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-stray-"));
      return { dir: strayDir, ref: "stray" };
    }

    it("reports the failing directory with git's own error text", () => {
      repo = createScratchRepo();
      const stray = createStrayWorktree();

      const result = cleanupWorktrees([stray], repo.dir);

      expect(result.failures.map((failure) => failure.dir)).toStrictEqual([stray.dir]);
      expect(result.failures[0]?.error).toContain("is not a working tree");
      expect(result.failures[0]?.error).not.toContain("Command failed");
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
        fs.rmSync(nonRepoDir, { recursive: true, force: true });
        nonRepoDir = undefined;
      }
    });

    it("reports the prune error instead of throwing", () => {
      nonRepoDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-not-a-repo-"));
      // A vanished directory is skipped before removal, so the sweep is the only
      // git call this list reaches — and it runs against a non-repo.
      const vanished: WorktreeInfo = { dir: path.join(nonRepoDir, "gone"), ref: "gone" };

      const result = cleanupWorktrees([vanished], nonRepoDir);

      expect(result.removed).toBe(0);
      expect(result.failures).toStrictEqual([]);
      expect(result.pruneError).toContain("not a git repository");
    });
  });

  describe("idempotency", () => {
    it("is safe to call multiple times on the same worktrees", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);

      cleanupWorktrees([worktree], repo.dir);

      cleanupWorktrees([worktree], repo.dir);

      expect(fs.existsSync(worktree.dir)).toBe(false);
      expect(listWorktreeDirs(repo.dir)).toHaveLength(1);
    });
  });
});
