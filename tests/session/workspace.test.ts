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
  killGitDuringWorktreeAdd,
  listWorktreeDirs,
  registerAbsentWorktree,
  type ScratchRepo,
} from "../fixtures/scratch-repo.js";

const BRANCH = `gymrat/${SESSION_ID}`;
const BASELINE_REF = "main";

/** The id the session after {@link SESSION_ID} opens on, for a workspace built over an earlier one's leftovers. */
const NEXT_SESSION_ID = "20260808-152045-b7c1";
const NEXT_BRANCH = `gymrat/${NEXT_SESSION_ID}`;

/** The ref a worktree has checked out: a branch name, or `HEAD` when detached. */
function checkedOutRef(worktree: string): string {
  return git(["rev-parse", "--abbrev-ref", "HEAD"], worktree);
}

function excludePath(root: string): string {
  return path.join(root, ".git", "info", "exclude");
}

/** The session branches `root` still holds, one per `gymrat/…` ref. */
function sessionBranches(root: string): string[] {
  return git(["for-each-ref", "--format=%(refname:short)", "refs/heads/gymrat"], root)
    .split("\n")
    .filter((line) => line !== "");
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
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      expect(git(["rev-parse", BRANCH], repo.dir)).toBe(baselineSha);
    });

    it("checks the experiment worktree out on the session branch", () => {
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(fs.existsSync(worktree)).toBe(true);
      expect.soft(checkedOutRef(worktree)).toBe(BRANCH);
    });

    it("pins the baseline worktree detached at the baseline SHA", () => {
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      const worktree = baselineWorktreeDir(repo.dir);
      expect.soft(git(["rev-parse", "HEAD"], worktree)).toBe(baselineSha);
      expect.soft(checkedOutRef(worktree)).toBe("HEAD");
    });

    it("returns the branch, both worktree paths, and the baseline ref and SHA", () => {
      const result = createWorkspace(repo.dir, SESSION_ID, {
        ref: BASELINE_REF,
        sha: baselineSha,
      });

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
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      const lines = fs.readFileSync(excludePath(repo.dir), "utf-8").split("\n");
      expect(lines).toContain(".gymrat/");
    });
  });

  describe("when the session branch already exists from an earlier run", () => {
    it("throws a GymratError naming the branch and hinting at a way out", () => {
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      const error = asGymratError(
        captureThrown(() =>
          createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha }),
        ),
      );

      expect.soft(error.message).toContain(BRANCH);
      expect.soft(error.hint).toMatch(/git branch -D/i);
    });
  });

  // killGitDuringWorktreeAdd sends POSIX signal 9 via a post-checkout hook;
  // Windows cannot deliver that signal to the git parent process.
  describe.skipIf(process.platform === "win32")(
    "when a worktree add fails after the session branch was created",
    () => {
      it("unwinds what it made, failing with the git step that broke", () => {
        // Installed after the scratch repo's own commit so only the
        // worktree checkouts under test die.
        killGitDuringWorktreeAdd(repo.dir);

        const error = asGymratError(
          captureThrown(() =>
            createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha }),
          ),
        );

        // The unwind's own git steps never speak for it.
        expect.soft(error.message).toMatch(/cannot create the experiment worktree/i);
        expect.soft(sessionBranches(repo.dir)).toStrictEqual([]);
        expect.soft(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
        expect(fs.existsSync(experimentWorktreeDir(repo.dir))).toBe(false);
      });
    },
  );

  describe("when git still holds an entry for a worktree whose directory is gone", () => {
    beforeEach(() => {
      // A session whose worktree directories vanished behind git's back: what
      // `git worktree remove` leaves when the directory is already off disk.
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });
      fs.rmSync(baselineWorktreeDir(repo.dir), { recursive: true, force: true });
    });

    it("checks the next session's worktrees out over the stale entries", () => {
      const result = createWorkspace(repo.dir, NEXT_SESSION_ID, {
        ref: BASELINE_REF,
        sha: baselineSha,
      });

      expect.soft(checkedOutRef(result.worktrees.experiment)).toBe(NEXT_BRANCH);
      expect(git(["rev-parse", "HEAD"], result.worktrees.baseline)).toBe(baselineSha);
    });

    it("leaves a worktree that is still on disk registered", () => {
      const live = path.join(repo.dir, "live-worktree");
      git(["worktree", "add", "--detach", live, baselineSha], repo.dir);

      createWorkspace(repo.dir, NEXT_SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

      expect.soft(fs.existsSync(live)).toBe(true);
      expect(listWorktreeDirs(repo.dir, { includeMain: false })).toContain(live);
    });
  });

  describe("when a worktree directory from an earlier session is still on disk", () => {
    it("leaves its uncommitted work standing and names the path it left", () => {
      // The earlier session's log is gone, so nothing told this run the
      // workspace was already there; its worktree still holds uncommitted work.
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      const stranded = path.join(experimentWorktreeDir(repo.dir), "README.md");
      fs.writeFileSync(stranded, "# work from the earlier session\n");

      const error = asGymratError(
        captureThrown(() =>
          createWorkspace(repo.dir, NEXT_SESSION_ID, { ref: BASELINE_REF, sha: baselineSha }),
        ),
      );

      // Only this attempt's own branch is unwound.
      expect.soft(fs.readFileSync(stranded, "utf-8")).toBe("# work from the earlier session\n");
      expect.soft(sessionBranches(repo.dir)).toStrictEqual([BRANCH]);
      expect(error.message).toContain(experimentWorktreeDir(repo.dir));
    });
  });

  describe("when the directory is not inside a git repository", () => {
    it("throws a GymratError naming the missing repository", () => {
      const outside = fs.mkdtempSync(path.join(os.tmpdir(), "ws-not-a-repo-"));

      try {
        const error = asGymratError(
          captureThrown(() =>
            createWorkspace(outside, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha }),
          ),
        );

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
      fs.writeFileSync(excludePath(repo.dir), "node_modules/\n");

      ensureGitExclude(repo.dir);

      const lines = fs.readFileSync(excludePath(repo.dir), "utf-8").split("\n");
      expect.soft(lines).toContain("node_modules/");
      expect.soft(lines.filter((line) => line === ".gymrat/")).toHaveLength(1);
    });
  });

  describe("when the exclude file already lists .gymrat/", () => {
    it("leaves the file byte-for-byte unchanged", () => {
      const before = "node_modules/\n.gymrat/\n";
      fs.writeFileSync(excludePath(repo.dir), before);

      ensureGitExclude(repo.dir);

      expect(fs.readFileSync(excludePath(repo.dir), "utf-8")).toBe(before);
    });
  });

  describe("when the exclude file does not exist", () => {
    it("creates it holding the .gymrat/ line", () => {
      fs.rmSync(excludePath(repo.dir), { force: true });

      ensureGitExclude(repo.dir);

      const lines = fs.readFileSync(excludePath(repo.dir), "utf-8").split("\n");
      expect(lines).toContain(".gymrat/");
    });
  });
});

describe("detectWorkspace", () => {
  it("returns false when neither worktree directory exists", () => {
    expect(detectWorkspace(repo.dir)).toBe(false);
  });

  it("returns true when both worktree directories exist", () => {
    createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });

    expect(detectWorkspace(repo.dir)).toBe(true);
  });

  it.each([
    { missing: "experiment", locate: experimentWorktreeDir },
    { missing: "baseline", locate: baselineWorktreeDir },
  ])("returns false when only the $missing worktree is gone", ({ locate }) => {
    createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
    fs.rmSync(locate(repo.dir), { recursive: true, force: true });

    expect(detectWorkspace(repo.dir)).toBe(false);
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
      const warnings = removeWorktrees(repo.dir, worktrees());

      expect.soft(warnings).toStrictEqual([]);
      expect.soft(fs.existsSync(experimentWorktreeDir(repo.dir))).toBe(false);
      expect.soft(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(false);
      expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
    });
  });

  describe("when one worktree directory is already gone", () => {
    it("removes the other and still warns about nothing", () => {
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      const warnings = removeWorktrees(repo.dir, worktrees());

      expect.soft(warnings).toStrictEqual([]);
      expect(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(false);
    });

    it("deregisters that worktree by name and no other", () => {
      // The user's own worktree, absent only for the moment.
      const absent = registerAbsentWorktree(repo.dir);
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      removeWorktrees(repo.dir, worktrees());

      const listed = listWorktreeDirs(repo.dir);
      expect.soft(listed).not.toContain(experimentWorktreeDir(repo.dir));
      expect(listed).toContain(absent);
    });
  });

  describe("when git refuses to remove a worktree", () => {
    it("warns naming the worktree it left behind and removes the other", () => {
      // git declines a locked worktree unless --force is passed twice.
      git(["worktree", "lock", experimentWorktreeDir(repo.dir)], repo.dir);

      const warnings = removeWorktrees(repo.dir, worktrees());

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
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      const worktree = experimentWorktreeDir(repo.dir);
      if (edit !== undefined) {
        fs.writeFileSync(path.join(worktree, edit), "# edited by the agent\n");
      }

      const dirty = isWorktreeDirty(worktree);

      expect(dirty).toBe(expected);
    },
  );

  describe("when the directory does not exist", () => {
    it("reports clean rather than failing on the missing worktree", () => {
      const dirty = isWorktreeDirty(experimentWorktreeDir(repo.dir));

      expect(dirty).toBe(false);
    });
  });
});

describe("recreateWorkspace", () => {
  describe("when the experiment worktree is gone", () => {
    it("puts it back on the session branch", () => {
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      recreateWorkspace(repo.dir, BRANCH, baselineSha);

      expect.soft(checkedOutRef(experimentWorktreeDir(repo.dir))).toBe(BRANCH);
      expect.soft(detectWorkspace(repo.dir)).toBe(true);
    });
  });

  describe("when the baseline worktree is gone", () => {
    it("puts it back detached at the pinned SHA", () => {
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      fs.rmSync(baselineWorktreeDir(repo.dir), { recursive: true, force: true });

      recreateWorkspace(repo.dir, BRANCH, baselineSha);

      const worktree = baselineWorktreeDir(repo.dir);
      expect.soft(git(["rev-parse", "HEAD"], worktree)).toBe(baselineSha);
      expect.soft(checkedOutRef(worktree)).toBe("HEAD");
    });
  });

  describe("when both worktrees are on disk", () => {
    it("leaves the experiment worktree's uncommitted work alone", () => {
      createWorkspace(repo.dir, SESSION_ID, { ref: BASELINE_REF, sha: baselineSha });
      const edited = path.join(experimentWorktreeDir(repo.dir), "README.md");
      fs.writeFileSync(edited, "# edited by the agent\n");

      recreateWorkspace(repo.dir, BRANCH, baselineSha);

      expect.soft(fs.readFileSync(edited, "utf-8")).toBe("# edited by the agent\n");
      expect.soft(detectWorkspace(repo.dir)).toBe(true);
    });
  });
});
