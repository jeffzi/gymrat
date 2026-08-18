import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { GymratError, messageOf } from "../../src/errors.js";
import {
  archivedSessionPath,
  baselineWorktreeDir,
  experimentWorktreeDir,
  lockfilePath,
  repoRoot,
  sessionDir,
  sessionJsonlPath,
  worktreesDir,
} from "../../src/session/paths.js";
import { SESSION_ID } from "../fixtures/constants.js";
import { captureThrown } from "../fixtures/errors.js";
import { createScratchRepo } from "../fixtures/scratch-repo.js";

/** An arbitrary absolute root: the derivation helpers never touch the filesystem. */
const ROOT = path.join(os.tmpdir(), "repo-root");

/**
 * Repo roots paired with the lockfile name gymrat has always given them.
 *
 * The names are golden values, not values recomputed from the hash the
 * implementation happens to use: two runs of the CLI over the same checkout must
 * land on the same lockfile, so the mapping is a contract rather than a detail.
 */
const LOCKFILE_NAMES = [
  { root: "/srv/projects/demo", name: "gymrat-lock-9fe2fb7fa4f9.json" },
  { root: "/srv/projects/other", name: "gymrat-lock-4ff7d20c47bc.json" },
];

describe("repoRoot", () => {
  describe("when the directory is inside a git repository", () => {
    it("returns the top level of the repository, not the directory itself", () => {
      const repo = createScratchRepo();
      try {
        const nested = path.join(repo.dir, "packages", "core");
        fs.mkdirSync(nested, { recursive: true });

        const root = repoRoot(nested);

        expect(path.normalize(root)).toBe(path.normalize(repo.dir));
      } finally {
        repo.cleanup();
      }
    });

    it("falls back to the process working directory when no directory is given", () => {
      const expected = execFileSync("git", ["rev-parse", "--show-toplevel"], {
        encoding: "utf-8",
      }).trim();

      const root = repoRoot();

      expect(path.normalize(root)).toBe(path.normalize(expected));
    });
  });

  describe("when the directory is not inside a git repository", () => {
    it("throws a GymratError naming the missing repository", () => {
      const outside = fs.mkdtempSync(path.join(os.tmpdir(), "not-a-repo-"));

      const error = captureThrown(() => repoRoot(outside));

      expect.soft(error).toBeInstanceOf(GymratError);
      expect.soft(messageOf(error)).toMatch(/git repository/i);
    });
  });
});

describe("session layout", () => {
  it.each([
    { label: "sessionDir", derive: sessionDir, relative: [".gymrat"] },
    { label: "sessionJsonlPath", derive: sessionJsonlPath, relative: [".gymrat", "session.jsonl"] },
    { label: "worktreesDir", derive: worktreesDir, relative: [".gymrat", "worktrees"] },
    {
      label: "experimentWorktreeDir",
      derive: experimentWorktreeDir,
      relative: [".gymrat", "worktrees", "experiment"],
    },
    {
      label: "baselineWorktreeDir",
      derive: baselineWorktreeDir,
      relative: [".gymrat", "worktrees", "baseline"],
    },
    {
      label: "archivedSessionPath",
      derive: (root: string) => archivedSessionPath(root, SESSION_ID),
      relative: [".gymrat", `session-${SESSION_ID}.jsonl`],
    },
  ])("$label places $relative under the repo root", ({ derive, relative }) => {
    const result = derive(ROOT);

    expect(result).toBe(path.join(ROOT, ...relative));
  });
});

describe("lockfilePath", () => {
  it.each(LOCKFILE_NAMES)("maps $root to $name in the system temp dir", ({ root, name }) => {
    const result = lockfilePath(root);

    expect(result).toBe(path.join(os.tmpdir(), name));
  });
});
