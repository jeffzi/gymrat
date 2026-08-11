import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { GymratError } from "../../src/errors.js";
import { baselineWorktreeDir, experimentWorktreeDir } from "../../src/session/paths.js";
import {
  createWorkspace,
  detectWorkspace,
  ensureGitExclude,
  isWorktreeDirty,
  recreateWorkspace,
  removeWorktrees,
} from "../../src/session/workspace.js";
import { SESSION_ID } from "../fixtures/constants.js";
import { captureThrown } from "../fixtures/errors.js";
import {
  createScratchRepo,
  git,
  listWorktreeDirs,
  type ScratchRepo,
} from "../fixtures/scratch-repo.js";

const BRANCH = `gymrat/${SESSION_ID}`;
const BASELINE_REF = "main";

/** The ref a worktree has checked out: a branch name, or `HEAD` when detached. */
function checkedOutRef(worktree: string): string {
  return git(["rev-parse", "--abbrev-ref", "HEAD"], worktree);
}

function excludePath(root: string): string {
  return path.join(root, ".git", "info", "exclude");
}

/** Narrow a thrown value to `GymratError` without an `as` cast. */
function asGymratError(error: unknown): GymratError {
  if (!(error instanceof GymratError)) {
    throw new Error(`expected a GymratError, got ${String(error)}`);
  }
  return error;
}

let repo: ScratchRepo;
let baselineSha: string;

beforeEach(() => {
  repo = createScratchRepo();
  baselineSha = git(["rev-parse", "HEAD"], repo.dir);
});

afterEach(() => {
  repo.cleanup();
});

describe("createWorkspace", () => {
  describe("when the repository has no session workspace", () => {
    it("creates the session branch at the baseline SHA", () => {
      // Act
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      // Assert
      expect(git(["rev-parse", BRANCH], repo.dir)).toBe(baselineSha);
    });

    it("checks the experiment worktree out on the session branch", () => {
      // Act
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(fs.existsSync(worktree)).toBe(true);
      expect.soft(checkedOutRef(worktree)).toBe(BRANCH);
    });

    it("pins the baseline worktree detached at the baseline SHA", () => {
      // Act
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      // Assert
      const worktree = baselineWorktreeDir(repo.dir);
      expect.soft(git(["rev-parse", "HEAD"], worktree)).toBe(baselineSha);
      expect.soft(checkedOutRef(worktree)).toBe("HEAD");
    });

    it("returns the branch, both worktree paths, and the baseline ref and SHA", () => {
      // Act
      const result = createWorkspace(repo.dir, SESSION_ID, {
        ref: BASELINE_REF,
        sha: baselineSha,
      });

      // Assert
      expect(result).toStrictEqual({
        branch: BRANCH,
        worktrees: {
          experiment: experimentWorktreeDir(repo.dir),
          baseline: baselineWorktreeDir(repo.dir),
        },
        baseline: { ref: BASELINE_REF, sha: baselineSha },
      });
    });

    it("excludes .gymrat/ from the repository's git status", () => {
      // Act
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      // Assert
      const lines = fs.readFileSync(excludePath(repo.dir), "utf-8").split("\n");
      expect(lines).toContain(".gymrat/");
    });
  });

  describe("when the session branch already exists from an earlier run", () => {
    it("throws a GymratError naming the branch and hinting at a way out", () => {
      // Arrange
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      // Act
      const error = asGymratError(
        captureThrown(() =>
          createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha }),
        ),
      );

      // Assert
      expect.soft(error.message).toContain(BRANCH);
      expect.soft(error.hint).toMatch(/git branch -D/i);
    });
  });

  describe("when the directory is not inside a git repository", () => {
    it("throws a GymratError naming the missing repository", () => {
      // Arrange
      const outside = fs.mkdtempSync(path.join(os.tmpdir(), "ws-not-a-repo-"));

      try {
        // Act
        const error = asGymratError(
          captureThrown(() =>
            createWorkspace(outside, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha }),
          ),
        );

        // Assert
        expect.soft(error.message).toMatch(/not a git repository/i);
        expect.soft(error.hint).toMatch(/git repository/i);
      } finally {
        fs.rmSync(outside, { recursive: true, force: true });
      }
    });
  });
});

describe("ensureGitExclude", () => {
  describe("when the exclude file does not list .gymrat/", () => {
    it("appends the line and keeps the existing entries", () => {
      // Arrange
      fs.writeFileSync(excludePath(repo.dir), "node_modules/\n");

      // Act
      ensureGitExclude(repo.dir);

      // Assert
      const lines = fs.readFileSync(excludePath(repo.dir), "utf-8").split("\n");
      expect.soft(lines).toContain("node_modules/");
      expect.soft(lines.filter((line) => line === ".gymrat/")).toHaveLength(1);
    });
  });

  describe("when the exclude file already lists .gymrat/", () => {
    it("leaves the file byte-for-byte unchanged", () => {
      // Arrange
      const before = "node_modules/\n.gymrat/\n";
      fs.writeFileSync(excludePath(repo.dir), before);

      // Act
      ensureGitExclude(repo.dir);

      // Assert
      expect(fs.readFileSync(excludePath(repo.dir), "utf-8")).toBe(before);
    });
  });

  describe("when the exclude file does not exist", () => {
    it("creates it holding the .gymrat/ line", () => {
      // Arrange
      fs.rmSync(excludePath(repo.dir), { force: true });

      // Act
      ensureGitExclude(repo.dir);

      // Assert
      const lines = fs.readFileSync(excludePath(repo.dir), "utf-8").split("\n");
      expect(lines).toContain(".gymrat/");
    });
  });
});

describe("detectWorkspace", () => {
  it("reports absent when neither worktree directory exists", () => {
    // Act
    const status = detectWorkspace(repo.dir);

    // Assert
    expect(status).toBe("absent");
  });

  it("reports present when both worktree directories exist", () => {
    // Arrange
    createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

    // Act
    const status = detectWorkspace(repo.dir);

    // Assert
    expect(status).toBe("present");
  });

  it.each([
    { missing: "experiment", locate: experimentWorktreeDir },
    { missing: "baseline", locate: baselineWorktreeDir },
  ])("reports partial when only the $missing worktree is gone", ({ locate }) => {
    // Arrange
    createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
    fs.rmSync(locate(repo.dir), { recursive: true, force: true });

    // Act
    const status = detectWorkspace(repo.dir);

    // Assert
    expect(status).toBe("partial");
  });
});

describe("removeWorktrees", () => {
  /** The two worktree paths `createWorkspace` laid down in the scratch repo. */
  function worktrees(): { experiment: string; baseline: string } {
    return {
      experiment: experimentWorktreeDir(repo.dir),
      baseline: baselineWorktreeDir(repo.dir),
    };
  }

  beforeEach(() => {
    createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
  });

  describe("when both worktrees are on disk", () => {
    it("takes them off disk and out of git's bookkeeping, warning about nothing", () => {
      // Act
      const warnings = removeWorktrees(repo.dir, worktrees());

      // Assert
      expect.soft(warnings).toStrictEqual([]);
      expect.soft(fs.existsSync(experimentWorktreeDir(repo.dir))).toBe(false);
      expect.soft(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(false);
      expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
    });
  });

  describe("when one worktree directory is already gone", () => {
    it("removes the other and still warns about nothing", () => {
      // Arrange
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      // Act
      const warnings = removeWorktrees(repo.dir, worktrees());

      // Assert
      expect.soft(warnings).toStrictEqual([]);
      expect(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(false);
    });
  });

  describe("when git refuses to remove a worktree", () => {
    it("warns naming the worktree it left behind and removes the other", () => {
      // Arrange - git declines a locked worktree unless --force is passed twice.
      git(["worktree", "lock", experimentWorktreeDir(repo.dir)], repo.dir);

      // Act
      const warnings = removeWorktrees(repo.dir, worktrees());

      // Assert
      expect.soft(warnings).toHaveLength(1);
      expect.soft(warnings[0]).toContain(experimentWorktreeDir(repo.dir));
      expect(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(false);
    });
  });
});

describe("isWorktreeDirty", () => {
  it.each([
    { description: "nothing was touched", edit: undefined, expected: false },
    { description: "a tracked file was edited", edit: "README.md", expected: true },
    { description: "an untracked file was added", edit: "scratch.txt", expected: true },
  ] satisfies { description: string; edit: string | undefined; expected: boolean }[])(
    "reports $expected when $description",
    ({ edit, expected }) => {
      // Arrange
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      const worktree = experimentWorktreeDir(repo.dir);
      if (edit !== undefined) {
        fs.writeFileSync(path.join(worktree, edit), "# edited by the agent\n");
      }

      // Act
      const dirty = isWorktreeDirty(worktree);

      // Assert
      expect(dirty).toBe(expected);
    },
  );

  describe("when the directory does not exist", () => {
    it("reports clean rather than failing on the missing worktree", () => {
      // Act
      const dirty = isWorktreeDirty(experimentWorktreeDir(repo.dir));

      // Assert
      expect(dirty).toBe(false);
    });
  });
});

describe("recreateWorkspace", () => {
  describe("when the experiment worktree is gone", () => {
    it("puts it back on the session branch", () => {
      // Arrange
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      // Act
      recreateWorkspace(repo.dir, BRANCH, baselineSha);

      // Assert
      expect.soft(checkedOutRef(experimentWorktreeDir(repo.dir))).toBe(BRANCH);
      expect.soft(detectWorkspace(repo.dir)).toBe("present");
    });
  });

  describe("when the baseline worktree is gone", () => {
    it("puts it back detached at the pinned SHA", () => {
      // Arrange
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      fs.rmSync(baselineWorktreeDir(repo.dir), { recursive: true, force: true });

      // Act
      recreateWorkspace(repo.dir, BRANCH, baselineSha);

      // Assert
      const worktree = baselineWorktreeDir(repo.dir);
      expect.soft(git(["rev-parse", "HEAD"], worktree)).toBe(baselineSha);
      expect.soft(checkedOutRef(worktree)).toBe("HEAD");
    });
  });

  describe("when both worktrees are on disk", () => {
    it("leaves the experiment worktree's uncommitted work alone", () => {
      // Arrange
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      const edited = path.join(experimentWorktreeDir(repo.dir), "README.md");
      fs.writeFileSync(edited, "# edited by the agent\n");

      // Act
      recreateWorkspace(repo.dir, BRANCH, baselineSha);

      // Assert
      expect.soft(fs.readFileSync(edited, "utf-8")).toBe("# edited by the agent\n");
      expect.soft(detectWorkspace(repo.dir)).toBe("present");
    });
  });
});
