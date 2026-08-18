import fs from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { finalizeSession } from "../../src/loop/finalize.js";
import { startSession } from "../../src/loop/start.js";
import {
  baselineWorktreeDir,
  experimentWorktreeDir,
  sessionJsonlPath,
} from "../../src/session/paths.js";
import type { SessionLogRecord, SessionRecord } from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import { ISO_PATTERN } from "../fixtures/constants.js";
import { captureGymratError } from "../fixtures/errors.js";
import {
  createScratchRepo,
  git,
  listWorktreeDirs,
  type ScratchRepo,
} from "../fixtures/scratch-repo.js";
import { committedKeep, iterationRecord, resolvedConfig } from "../fixtures/session-records.js";

let repo: ScratchRepo;
/** The commit `main` sits on before the session opens — the baseline every squash hangs from. */
let baselineSha: string;

function jsonlPath(): string {
  return sessionJsonlPath(repo.dir);
}

function records(): SessionLogRecord[] {
  return readRecords(jsonlPath());
}

/** The header the open session wrote, failing the test when the log holds none. */
function sessionHeader(): SessionRecord {
  const first = records()[0];
  if (first?.type !== "session") {
    throw new Error(`expected a session header in ${jsonlPath()}`);
  }
  return first;
}

/** The record the log ends on, failing the test when the log is empty. */
function lastRecord(): SessionLogRecord {
  const last = records().at(-1);
  if (last === undefined) {
    throw new Error(`expected a record in ${jsonlPath()}`);
  }
  return last;
}

/**
 * Commit one edit in the experiment worktree and log the iteration behind it.
 *
 * The experiment worktree is checked out on the session branch, so each call
 * moves that branch forward exactly as a real `gymrat keep` would. The keep
 * record is left to the caller, which is what lets a test log one whose fields
 * gymrat itself would never omit.
 */
function commitIteration(seq: number, message: string): string {
  const worktree = experimentWorktreeDir(repo.dir);
  fs.writeFileSync(path.join(worktree, `step-${seq}.txt`), `${message}\n`);
  git(["add", "-A"], worktree);
  git(["commit", "-m", message], worktree);
  const commit = git(["rev-parse", "HEAD"], worktree);
  appendRecord(jsonlPath(), iterationRecord({ seq }));
  return commit;
}

/** Commit one edit and log the iteration and the committed keep that settled it. */
function keepIteration(seq: number, message: string): string {
  const commit = commitIteration(seq, message);
  appendRecord(jsonlPath(), committedKeep(seq, { commit, message }));
  return commit;
}

beforeEach(() => {
  repo = createScratchRepo();
  baselineSha = git(["rev-parse", "HEAD"], repo.dir);
  startSession(repo.dir, "main", resolvedConfig());
});

afterEach(() => {
  repo.cleanup();
});

describe("finalizeSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", () => {
      const empty = createScratchRepo();

      try {
        const error = captureGymratError(() => finalizeSession(empty.dir));

        expect(error.hint).toContain("gymrat start");
      } finally {
        empty.cleanup();
      }
    });
  });

  describe("when the session was already finalized", () => {
    it("refuses naming the closed session and pointing at a fresh start", () => {
      keepIteration(1, "cache the regex");
      finalizeSession(repo.dir);

      const error = captureGymratError(() => finalizeSession(repo.dir));

      expect.soft(error.message).toContain(sessionHeader().sessionId);
      expect(error.hint).toContain("gymrat start");
    });
  });

  describe("when nothing has been kept", () => {
    it("refuses with a hint to keep some work first, creating no branch and no record", () => {
      const before = records().length;

      const error = captureGymratError(() => finalizeSession(repo.dir));

      expect.soft(error.hint).toMatch(/keep/i);
      expect.soft(git(["branch", "--list", "*-final"], repo.dir)).toBe("");
      expect(records()).toHaveLength(before);
    });
  });

  describe("when the last iteration is neither kept nor discarded", () => {
    it("refuses with a hint to settle it first, writing no record", () => {
      keepIteration(1, "cache the regex");
      appendRecord(jsonlPath(), iterationRecord({ seq: 2 }));
      const before = records().length;

      const error = captureGymratError(() => finalizeSession(repo.dir));

      expect.soft(error.hint).toMatch(/keep/i);
      expect.soft(error.hint).toMatch(/discard/i);
      expect(records()).toHaveLength(before);
    });
  });

  describe("when the experiment worktree carries uncommitted work", () => {
    it("refuses with a hint to settle it first, writing no record", () => {
      keepIteration(1, "cache the regex");
      fs.writeFileSync(path.join(experimentWorktreeDir(repo.dir), "scratch.txt"), "notes\n");
      const before = records().length;

      const error = captureGymratError(() => finalizeSession(repo.dir));

      expect.soft(error.hint).toMatch(/keep/i);
      expect.soft(error.hint).toMatch(/discard/i);
      expect(records()).toHaveLength(before);
    });
  });

  describe("when the experiment worktree is already gone from disk", () => {
    it("finalizes anyway rather than asking about work it cannot see", () => {
      keepIteration(1, "cache the regex");
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      const result = finalizeSession(repo.dir);

      expect(lastRecord()).toStrictEqual(result.record);
    });
  });

  describe("when a committed keep carries no message", () => {
    it("stands the keep's short commit in, leaving one body line per kept iteration", () => {
      keepIteration(1, "cache the regex");
      const commit = commitIteration(2, "hoist the loop");
      appendRecord(jsonlPath(), committedKeep(2, { commit, message: undefined }));

      const result = finalizeSession(repo.dir);

      const subject = git(["log", "-1", "--format=%s", result.record.branch], repo.dir);
      const body = git(["log", "-1", "--format=%b", result.record.branch], repo.dir);
      expect.soft(subject).toContain("2 kept iterations");
      expect(body.split("\n")).toStrictEqual(["cache the regex", commit.slice(0, 7)]);
    });

    it("stands a placeholder in when the keep names no commit either", () => {
      commitIteration(1, "cache the regex");
      appendRecord(jsonlPath(), committedKeep(1, { commit: undefined, message: undefined }));

      const result = finalizeSession(repo.dir);

      const body = git(["log", "-1", "--format=%b", result.record.branch], repo.dir);
      expect(body.split("\n")).toStrictEqual(["(no message)"]);
    });
  });

  describe("when the session has committed keeps", () => {
    const MESSAGES = ["cache the regex", "hoist the loop"];
    /** The branch finalize names when the caller does not. */
    let finalBranch: string;

    beforeEach(() => {
      for (const [index, message] of MESSAGES.entries()) {
        keepIteration(index + 1, message);
      }
      finalBranch = `${sessionHeader().branch}-final`;
    });

    it("builds one commit carrying the session branch's tree on the pinned baseline", () => {
      const sessionTree = git(["rev-parse", `${sessionHeader().branch}^{tree}`], repo.dir);

      const result = finalizeSession(repo.dir);

      expect.soft(git(["rev-parse", `${finalBranch}^{tree}`], repo.dir)).toBe(sessionTree);
      expect.soft(git(["rev-parse", `${finalBranch}^`], repo.dir)).toBe(baselineSha);
      expect(git(["rev-parse", finalBranch], repo.dir)).toBe(result.record.commit);
    });

    it("moves neither the repository's checkout nor the session branch", () => {
      const sessionHead = git(["rev-parse", sessionHeader().branch], repo.dir);

      finalizeSession(repo.dir);

      expect.soft(git(["rev-parse", "HEAD"], repo.dir)).toBe(baselineSha);
      expect.soft(git(["rev-parse", "--abbrev-ref", "HEAD"], repo.dir)).toBe("main");
      expect(git(["rev-parse", sessionHeader().branch], repo.dir)).toBe(sessionHead);
    });

    it("appends a finalize record naming the branch and the squash commit", () => {
      const result = finalizeSession(repo.dir);

      expect.soft(result.record).toStrictEqual({
        type: "finalize",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        branch: finalBranch,
        commit: git(["rev-parse", finalBranch], repo.dir),
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        message: expect.any(String),
      });
      expect(lastRecord()).toStrictEqual(result.record);
    });

    it("takes both worktrees off disk and out of git's bookkeeping", () => {
      finalizeSession(repo.dir);

      expect.soft(fs.existsSync(experimentWorktreeDir(repo.dir))).toBe(false);
      expect.soft(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(false);
      expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
    });

    it("reports the branch, the short commit, the kept count, and the closed session", () => {
      const result = finalizeSession(repo.dir);

      expect.soft(result.report).toContain(finalBranch);
      expect.soft(result.report).toContain(result.record.commit.slice(0, 7));
      expect.soft(result.report).toContain("2 kept");
      expect(result.report).toMatch(/closed/i);
    });

    it("generates a message naming the kept count over the kept commit messages", () => {
      const result = finalizeSession(repo.dir);

      const subject = git(["log", "-1", "--format=%s", finalBranch], repo.dir);
      const body = git(["log", "-1", "--format=%b", finalBranch], repo.dir);
      expect.soft(subject).toContain("2 kept iterations");
      expect.soft(body.split("\n")).toStrictEqual(MESSAGES);
      expect(result.record.message).toBe(`${subject}\n\n${body}`);
    });

    it("commits the caller's message verbatim when given one", () => {
      const result = finalizeSession(repo.dir, { message: "squash the tuning session" });

      expect
        .soft(git(["log", "-1", "--format=%B", finalBranch], repo.dir))
        .toBe("squash the tuning session");
      expect(result.record.message).toBe("squash the tuning session");
    });

    it("points the caller's branch name at the squash commit when given one", () => {
      const result = finalizeSession(repo.dir, { branch: "perf/regex-cache" });

      expect.soft(result.record.branch).toBe("perf/regex-cache");
      expect(git(["rev-parse", "perf/regex-cache"], repo.dir)).toBe(result.record.commit);
    });

    it("refuses when the branch it would create already exists, creating nothing", () => {
      git(["branch", finalBranch, baselineSha], repo.dir);
      const before = records().length;

      const error = captureGymratError(() => finalizeSession(repo.dir));

      expect.soft(error.message).toContain(finalBranch);
      expect.soft(git(["rev-parse", finalBranch], repo.dir)).toBe(baselineSha);
      expect(records()).toHaveLength(before);
    });

    it("closes the session anyway when git refuses to remove a worktree", () => {
      // A locked worktree is the one git declines to take with a
      // single --force, standing in for any removal the filesystem blocks.
      const experiment = experimentWorktreeDir(repo.dir);
      git(["worktree", "lock", experiment], repo.dir);

      const result = finalizeSession(repo.dir);

      expect.soft(result.report).toContain(experiment);
      expect.soft(result.report).toMatch(/git worktree remove/i);
      expect(lastRecord()).toStrictEqual(result.record);
    });
  });
});
