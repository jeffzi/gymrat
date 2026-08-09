import { execFileSync } from "node:child_process";
import fs from "node:fs";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createProgram } from "../../src/cli.js";
import type { ResolvedConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import { startSession } from "../../src/loop/start.js";
import {
  baselineWorktreeDir,
  experimentWorktreeDir,
  sessionJsonlPath,
} from "../../src/session/paths.js";
import type { IterationRecord, SessionRecord } from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import { detectWorkspace } from "../../src/session/workspace.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";
import { committedKeep, iterationRecord } from "../fixtures/session-records.js";

const SESSION_ID_PATTERN = /^\d{8}-\d{6}-[0-9a-f]{4}$/;
const ISO_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
/**
 * A settled run configuration carrying both the keys the session header snapshots
 * and keys it must leave out (`unstableNoisePct`, `stop`).
 */
const CONFIG: ResolvedConfig = {
  bench: "npm run bench",
  prepare: "npm run build",
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
  unstableNoisePct: 200,
  primary: "geomean",
  filter: "npm run bench -- {names}",
  hooks: "gymrat.hooks",
  stop: { maxIterations: 20 },
};

/** The subset of `CONFIG` the session record's schema keeps as provenance. */
const CONFIG_SNAPSHOT = {
  bench: "npm run bench",
  prepare: "npm run build",
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
  primary: "geomean",
  filter: "npm run bench -- {names}",
  hooks: "gymrat.hooks",
};

/** Run git in `cwd` and return its trimmed stdout. */
function git(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, stdio: "pipe", encoding: "utf-8" }).trim();
}

/** Run `act` and hand back the GymratError it threw, failing the test if it threw none. */
function captureGymratError(act: () => unknown): GymratError {
  try {
    act();
  } catch (error) {
    if (error instanceof GymratError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected the call to throw a GymratError");
}

/** The session header `root`'s log opens with, failing the test when there is none. */
function sessionHeaderOf(root: string): SessionRecord {
  const [first] = readRecords(sessionJsonlPath(root));
  if (first?.type !== "session") {
    throw new Error(`expected a session header in ${sessionJsonlPath(root)}`);
  }
  return first;
}

/** A measured iteration numbered `seq`, settled by nobody. */
function iteration(seq: number): IterationRecord {
  return iterationRecord({ seq });
}

let repo: ScratchRepo;
let headSha: string;
let originalCwd: string;

beforeEach(() => {
  originalCwd = process.cwd();
  repo = createScratchRepo();
  headSha = git(["rev-parse", "HEAD"], repo.dir);
});

afterEach(() => {
  vi.restoreAllMocks();
  process.chdir(originalCwd);
  repo.cleanup();
});

describe("startSession", () => {
  describe("when the repository holds no session yet", () => {
    it("writes a header naming the baseline, branch, worktrees, and config snapshot", () => {
      // Act
      startSession(repo.dir, "main", CONFIG);

      // Assert
      expect(readRecords(sessionJsonlPath(repo.dir))).toStrictEqual([
        {
          type: "session",
          schemaVersion: 1,
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          sessionId: expect.stringMatching(SESSION_ID_PATTERN),
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          createdAt: expect.stringMatching(ISO_PATTERN),
          baseline: { ref: "main", sha: headSha },
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          branch: expect.stringMatching(/^gymrat\/\d{8}-\d{6}-[0-9a-f]{4}$/),
          worktrees: {
            experiment: experimentWorktreeDir(repo.dir),
            baseline: baselineWorktreeDir(repo.dir),
          },
          config: CONFIG_SNAPSHOT,
        },
      ]);
    });

    it("names the session branch after the session id", () => {
      // Act
      const result = startSession(repo.dir, "main", CONFIG);

      // Assert
      expect(result.session.branch).toBe(`gymrat/${result.session.sessionId}`);
    });

    it("checks out the experiment and baseline worktrees", () => {
      // Act
      startSession(repo.dir, "main", CONFIG);

      // Assert
      expect(detectWorkspace(repo.dir)).toBe("present");
    });

    it("returns the recorded session with no history behind it", () => {
      // Act
      const result = startSession(repo.dir, "main", CONFIG);

      // Assert
      expect(result).toStrictEqual({
        session: sessionHeaderOf(repo.dir),
        state: {
          session: sessionHeaderOf(repo.dir),
          iterationCount: 0,
          lastIteration: undefined,
          unsettled: false,
          keepCount: 0,
          discardCount: 0,
          targetReachedAndKept: false,
          lastSeq: 0,
        },
        resumed: false,
      });
    });

    it("pins the baseline at HEAD when no ref is given", () => {
      // Act
      const result = startSession(repo.dir, undefined, CONFIG);

      // Assert
      expect(result.session.baseline).toStrictEqual({ ref: "HEAD", sha: headSha });
    });
  });

  describe("when a session is already on disk", () => {
    it("returns the recorded session and its counts without appending a record", () => {
      // Arrange
      const created = startSession(repo.dir, "main", CONFIG).session;
      appendRecord(sessionJsonlPath(repo.dir), iteration(1));
      appendRecord(sessionJsonlPath(repo.dir), committedKeep(1));

      // Act
      const result = startSession(repo.dir, "main", CONFIG);

      // Assert
      expect.soft(result.session).toStrictEqual(created);
      expect.soft(result.resumed).toBe(true);
      expect.soft(result.state.iterationCount).toBe(1);
      expect.soft(result.state.keepCount).toBe(1);
      expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(3);
    });

    it("puts back a worktree that went missing", () => {
      // Arrange
      startSession(repo.dir, "main", CONFIG);
      fs.rmSync(experimentWorktreeDir(repo.dir), { recursive: true, force: true });

      // Act
      startSession(repo.dir, "main", CONFIG);

      // Assert
      expect(detectWorkspace(repo.dir)).toBe("present");
    });
  });

  describe("when the baseline ref does not resolve", () => {
    it("throws a GymratError naming the ref and leaves no session behind", () => {
      // Act
      const error = captureGymratError(() => startSession(repo.dir, "no-such-ref", CONFIG));

      // Assert
      expect.soft(error.message).toContain("no-such-ref");
      expect(fs.existsSync(sessionJsonlPath(repo.dir))).toBe(false);
    });
  });
});

describe("the start command", () => {
  it("creates a session in the repository it runs in and reports it on stdout", async () => {
    // Arrange
    process.chdir(repo.dir);
    const program = createProgram();
    program.exitOverride();
    let stdout = "";
    vi.spyOn(process.stdout, "write").mockImplementation((chunk) => {
      stdout += String(chunk);
      return true;
    });

    // Act
    await program.parseAsync(["node", "cli.js", "start", "main", "--bench", "npm run bench"]);

    // Assert
    const session = sessionHeaderOf(repo.dir);
    expect.soft(session.baseline).toStrictEqual({ ref: "main", sha: headSha });
    expect(stdout).toContain(session.branch);
  });
});
