import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterEach } from "vitest";

import type { InPlaceTarget, RefTarget, WorktreeInfo } from "../src/targets.js";
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
    it("creates a detached worktree under os.tmpdir()", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget("HEAD", sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

      expect(worktree.dir).toContain(os.tmpdir());
    });

    it("creates a directory that exists and is accessible", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

      expect(fs.existsSync(worktree.dir)).toBe(true);
    });

    it("creates a worktree containing repo files from the checked-out ref", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

      const readmeFile = path.join(worktree.dir, "README.md");
      expect(fs.existsSync(readmeFile)).toBe(true);
      expect(fs.readFileSync(readmeFile, "utf-8")).toBe("# Test Repo\n");
    });

    it("returns a WorktreeInfo with the original ref", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget("my-tag", sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

      expect(worktree.ref).toBe("my-tag");
    });

    it("returns a WorktreeInfo with an absolute directory path", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget("my-tag", sha);

      const worktree = createWorktree(refTarget, repo.dir);
      createdWorktrees.push(worktree);

      expect(path.isAbsolute(worktree.dir)).toBe(true);
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
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);
      expect(fs.existsSync(worktree.dir)).toBe(true);

      cleanupWorktrees([worktree], repo.dir);

      expect(fs.existsSync(worktree.dir)).toBe(false);
    });

    it("runs git worktree prune after removing worktrees", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);

      expect(() => {
        cleanupWorktrees([worktree], repo!.dir);
      }).not.toThrow();
    });
  });

  describe("when given empty worktree list", () => {
    it("handles empty array without error", () => {
      repo = createScratchRepo();

      expect(() => {
        cleanupWorktrees([], repo!.dir);
      }).not.toThrow();
    });
  });

  describe("idempotency", () => {
    it("is safe to call multiple times on the same worktrees", () => {
      repo = createScratchRepo();
      const sha = getHeadSha(repo.dir);
      const refTarget = createRefTarget(sha, sha);

      const worktree = createWorktree(refTarget, repo.dir);

      cleanupWorktrees([worktree], repo.dir);
      expect(fs.existsSync(worktree.dir)).toBe(false);

      expect(() => {
        cleanupWorktrees([worktree], repo!.dir);
      }).not.toThrow();
    });
  });
});
