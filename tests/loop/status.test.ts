import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createProgram } from "../../src/cli.js";
import type { ResolvedConfig } from "../../src/config.js";
import { GymratError, messageOf } from "../../src/errors.js";
import { statusSession } from "../../src/loop/status.js";
import { sessionJsonlPath } from "../../src/session/paths.js";
import type {
  BaselineRecord,
  DiscardRecord,
  HookRecord,
  IterationRecord,
  KeepRecord,
  SessionLogRecord,
  SessionRecord,
} from "../../src/session/records.js";
import { appendRecord } from "../../src/session/store.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";

const AT = "2026-08-08T14:15:30.000Z";
const SESSION_ID = "20260808-141530-a3f2";
/** A 40-hex baseline sha whose first seven characters are recognizable on their own. */
const BASELINE_SHA = `a1b2c3d${"e".repeat(33)}`;
/** A 40-hex commit sha whose first seven characters are recognizable on their own. */
const KEEP_COMMIT = `b1b2b3b${"c".repeat(33)}`;
const ANSI_RE = /\x1b\[/;

/** A fresh repository root with no session on it yet. */
function freshRoot(): string {
  return fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-status-")));
}

/** The worktree paths a session under `root` records. */
function worktrees(root: string): SessionRecord["worktrees"] {
  return {
    experiment: path.join(root, ".gymrat", "worktrees", "experiment"),
    baseline: path.join(root, ".gymrat", "worktrees", "baseline"),
  };
}

/** The session header `start` writes for `root`. */
function sessionRecord(root: string): SessionRecord {
  return {
    type: "session",
    schemaVersion: 1,
    sessionId: SESSION_ID,
    createdAt: AT,
    baseline: { ref: "main", sha: BASELINE_SHA },
    branch: `gymrat/${SESSION_ID}`,
    worktrees: worktrees(root),
    config: {
      bench: "npm run bench",
      adapter: "metric-lines",
      samples: 10,
      timeoutSeconds: 1800,
      primary: "geomean",
      hooks: "gymrat.hooks",
    },
  };
}

/** A settled run configuration, geomean-led unless a test names its own primary. */
function config(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return {
    bench: "npm run bench",
    adapter: "metric-lines",
    samples: 10,
    timeoutSeconds: 1800,
    unstableNoisePct: 200,
    primary: "geomean",
    hooks: "gymrat.hooks",
    ...overrides,
  };
}

/** A measured iteration numbered `seq`, reading as `outcome` on a `deltaPct` primary. */
function iteration(
  seq: number,
  deltaPct: number,
  outcome: IterationRecord["outcome"],
  targetReached = false,
): IterationRecord {
  return {
    type: "iteration",
    seq,
    at: AT,
    samples: {
      experiment: [{ total_ms: 14100 }, { total_ms: 14088 }],
      baseline: [{ total_ms: 15200 }, { total_ms: 15190 }],
    },
    metrics: {
      total_ms: {
        deltaPct,
        verdict: outcome,
        method: "signed-rank",
        p: 0.002,
        noisePct: 1.4,
        gating: true,
        confirmed: outcome === "regressed",
      },
    },
    primary: { kind: "metric", name: "total_ms", deltaPct },
    outcome,
    targetReached,
  };
}

/** A keep that committed the iteration numbered `seq`. */
function committedKeep(seq: number): KeepRecord {
  return {
    type: "keep",
    seq,
    at: AT,
    status: "committed",
    commit: KEEP_COMMIT,
    message: "cache the regex",
    checks: { configured: true, passed: true },
  };
}

/** A keep the checks gate refused, leaving the iteration numbered `seq` uncommitted. */
function blockedKeep(seq: number): KeepRecord {
  return {
    type: "keep",
    seq,
    at: AT,
    status: "blocked",
    reason: "checks-failed",
    checks: { configured: true, passed: false },
  };
}

/** A discard of the iteration numbered `seq`. */
function discard(seq: number): DiscardRecord {
  return { type: "discard", seq, at: AT };
}

/** A recorded baseline measurement of `main`. */
const BASELINE: BaselineRecord = {
  type: "baseline",
  at: AT,
  label: "main",
  samples: [{ total_ms: 15200 }, { total_ms: 15184 }],
};

/** A hook run around the first iteration — history `status` has no line for. */
const HOOK: HookRecord = {
  type: "hook",
  stage: "before",
  seq: 1,
  exitCode: 0,
  durationMs: 120,
  stdoutBytes: 80,
  timedOut: false,
};

/** Write a session log for `root` opening on its header and holding `history` after it. */
function writeSessionLog(root: string, history: SessionLogRecord[] = []): void {
  appendRecord(sessionJsonlPath(root), sessionRecord(root));
  for (const record of history) {
    appendRecord(sessionJsonlPath(root), record);
  }
}

/**
 * A session that measured four iterations: one kept, one discarded, one whose
 * keep the checks blocked, and one still waiting to be settled.
 */
function fourIterations(): SessionLogRecord[] {
  return [
    BASELINE,
    HOOK,
    iteration(1, -7.2, "improved"),
    committedKeep(1),
    iteration(2, 9.4, "regressed"),
    discard(2),
    iteration(3, -3.1, "improved"),
    blockedKeep(3),
    iteration(4, 0.1, "no-signal"),
  ];
}

/** The report's lines, stripped of color, for asserting on what it says. */
function reportLines(report: string): string[] {
  return stripAnsi(report).split("\n");
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
  throw new Error("expected the call to fail with a GymratError");
}

describe("statusSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", () => {
      // Act
      const error = captureGymratError(() => statusSession(freshRoot(), config()));

      // Assert
      expect(error.hint).toContain("gymrat start");
    });
  });

  describe("when a line of the session log is not valid JSON", () => {
    it("surfaces the store's error naming the log and the line number", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root);
      fs.appendFileSync(sessionJsonlPath(root), "{not json\n");

      // Act
      const error = captureGymratError(() => statusSession(root, config()));

      // Assert
      expect(messageOf(error)).toContain(`${sessionJsonlPath(root)}:2`);
    });
  });

  describe("when the log holds a whole session's history", () => {
    it("renders the header, every measured record in file order, and the totals", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root, fourIterations());

      // Act
      const report = statusSession(root, config());

      // Assert
      expect(reportLines(report)).toStrictEqual([
        `session ${SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines`,
        `branch gymrat/${SESSION_ID}`,
        `experiment worktree ${worktrees(root).experiment}`,
        `baseline worktree ${worktrees(root).baseline}`,
        "baseline main · total_ms 15192",
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "iteration 2 · ✗ +9.4% · discarded",
        "iteration 3 · ✓ -3.1% · keep-blocked (checks-failed)",
        "iteration 4 · ~ +0.1% · unsettled",
        "4 iterations · 1 kept · 1 discarded",
      ]);
    });

    it("renders the same report every time, holding nothing between calls", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root, fourIterations());

      // Act
      const report = statusSession(root, config());

      // Assert
      expect(statusSession(root, config())).toBe(report);
    });
  });

  describe("when a stop condition is configured", () => {
    it("counts the iterations on file against the configured maximum", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root, fourIterations());

      // Act
      const report = statusSession(root, config({ stop: { maxIterations: 30 } }));

      // Assert
      expect(reportLines(report)).toContain("stop: 4 of 30 iterations");
    });

    it.each([
      { desc: "the target-reaching iteration was kept", kept: true, expected: "target reached" },
      {
        desc: "nobody kept the target-reaching iteration",
        kept: false,
        expected: "target pending",
      },
    ])("reports $expected when $desc", ({ kept, expected }) => {
      // Arrange
      const root = freshRoot();
      const settle = kept ? committedKeep(1) : discard(1);
      writeSessionLog(root, [iteration(1, -7.2, "improved", true), settle]);
      const resolved = config({ primary: "total_ms", stop: { targetValue: 95 } });

      // Act
      const report = statusSession(root, resolved);

      // Assert
      expect(reportLines(report)).toContain(`stop: ${expected}`);
    });
  });
});

describe("the status command", () => {
  let repo: ScratchRepo;
  let originalCwd: string;

  beforeEach(() => {
    originalCwd = process.cwd();
    repo = createScratchRepo();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    process.chdir(originalCwd);
    repo.cleanup();
  });

  /** A program whose subcommands throw instead of exiting, with stderr silenced. */
  function createSilentProgram(): ReturnType<typeof createProgram> {
    const program = createProgram();
    for (const command of [program, ...program.commands]) {
      command.exitOverride();
      command.configureOutput({ writeErr: () => {} });
    }
    return program;
  }

  /** Turn `process.exit` into a catchable rejection carrying the intended code. */
  function mockProcessExit(): void {
    // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- vitest mock requires cast
    vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw Object.assign(new Error(`process.exit(${code})`), { exitCode: code });
    }) as never);
  }

  /** Collect everything the program writes to stdout. */
  function captureStdout(): () => string {
    let stdout = "";
    vi.spyOn(process.stdout, "write").mockImplementation((chunk) => {
      stdout += String(chunk);
      return true;
    });
    return () => stdout;
  }

  /** Write the config file every command reads its settings from. */
  function writeConfigFile(): void {
    fs.writeFileSync(
      path.join(repo.dir, "gymrat.json"),
      JSON.stringify({ bench: "npm run bench" }),
    );
  }

  it("renders the session in the repository it runs in on stdout", async () => {
    // Arrange
    writeSessionLog(repo.dir, fourIterations());
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createSilentProgram();
    const stdout = captureStdout();

    // Act
    await program.parseAsync(["node", "cli.js", "status"]);

    // Assert
    const lines = reportLines(stdout());
    expect.soft(lines[0]).toContain(`session ${SESSION_ID}`);
    expect(lines).toContain("4 iterations · 1 kept · 1 discarded");
  });

  it("exits 2 with a start hint when the repository holds no session", async () => {
    // Arrange
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createSilentProgram();
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockProcessExit();

    // Act
    const parsing = program.parseAsync(["node", "cli.js", "status"]);

    // Assert
    await expect(parsing).rejects.toHaveProperty("exitCode", 2);
    const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
    expect(stderrText).toContain("gymrat start");
  });

  it.each([
    { desc: "styles the report when the terminal takes color", args: [], color: "1", ansi: true },
    {
      desc: "drops every style when --no-color is passed",
      args: ["--no-color"],
      color: undefined,
      ansi: false,
    },
  ])("$desc", async ({ args, color, ansi }) => {
    // Arrange
    vi.stubEnv("NO_COLOR", undefined);
    vi.stubEnv("FORCE_COLOR", color);
    writeSessionLog(repo.dir, fourIterations());
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createSilentProgram();
    const stdout = captureStdout();

    // Act
    await program.parseAsync(["node", "cli.js", "status", ...args]);

    // Assert
    expect(ANSI_RE.test(stdout())).toBe(ansi);
  });
});
