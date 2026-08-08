import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterEach, vi } from "vitest";

import { captureStderr } from "../fixtures/console.js";
import { createScratchRepo } from "../fixtures/scratch-repo.js";
import type { ScratchRepo } from "../fixtures/scratch-repo.js";
import { removeTempRoot } from "./temp-root.js";

/**
 * Break a worktree so `git worktree remove` refuses to take it, leaving plain
 * filesystem removal as the only way out.
 */
type Sabotage = (worktreeDir: string, repo: ScratchRepo) => void;

const STUBBORN_WORKTREES: { label: string; sabotage: Sabotage }[] = [
  {
    label: "a deleted .git link",
    sabotage: (worktreeDir) => {
      fs.rmSync(path.join(worktreeDir, ".git"));
    },
  },
  {
    label: "a .git link pointing at a repository that no longer exists",
    sabotage: (_worktreeDir, repo) => {
      repo.cleanup();
    },
  },
];

describe("per-file temp root", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("while a test file runs", () => {
    it("points os.tmpdir() at a resolved directory this suite created", () => {
      const root = os.tmpdir();

      expect(path.basename(root)).toMatch(/^gymrat-vitest-\w+$/);
      expect(fs.statSync(root).isDirectory()).toBe(true);
      expect(root).toBe(fs.realpathSync.native(root));
    });

    it("keeps a path built from os.tmpdir() inside that root", () => {
      const root = os.tmpdir();

      const child = fs.mkdtempSync(path.join(os.tmpdir(), "child-"));

      expect(path.dirname(child)).toBe(root);
    });
  });

  describe("removeTempRoot", () => {
    it.each(STUBBORN_WORKTREES)("deletes a root holding a worktree with $label", ({ sabotage }) => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "stubborn-"));
      const repo = createScratchRepo();
      try {
        const worktreeDir = path.join(root, "wt");
        execFileSync("git", ["worktree", "add", "--detach", worktreeDir, "HEAD"], {
          cwd: repo.dir,
          stdio: "pipe",
        });
        sabotage(worktreeDir, repo);

        removeTempRoot(root);

        expect(fs.existsSync(root)).toBe(false);
      } finally {
        repo.cleanup();
      }
    });

    it("reports a root it cannot delete instead of throwing", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "locked-"));
      vi.spyOn(fs, "rmSync").mockImplementation(() => {
        throw new Error("EBUSY: resource busy or locked");
      });

      const warned = captureStderr(() => {
        removeTempRoot(root);
      });

      expect(warned).toContain(root);
      expect(warned).toContain("EBUSY");
    });
  });
});
