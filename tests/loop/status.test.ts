import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ResolvedConfig } from "../../src/config.js";
import { messageOf } from "../../src/errors.js";
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
import { captureStdout, createRunnableProgram, mockProcessExit } from "../fixtures/cli-harness.js";
import { ANSI_RE, SESSION_ID, reportLines } from "../fixtures/constants.js";
import { captureGymratError } from "../fixtures/errors.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";
import {
  AT,
  committedKeep,
  iterationRecord,
  resolvedConfig,
  sessionRecord as sessionRecordDefaults,
} from "../fixtures/session-records.js";

/** A 40-hex baseline sha whose first seven characters are recognizable on their own. */
const BASELINE_SHA = `a1b2c3d${"e".repeat(33)}`;
/** A 40-hex commit sha whose first seven characters are recognizable on their own. */
const KEEP_COMMIT = `b1b2b3b${"c".repeat(33)}`;

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
  return sessionRecordDefaults({
    baseline: { ref: "main", sha: BASELINE_SHA },
    worktrees: worktrees(root),
  });
}

/** A settled run configuration, geomean-led unless a test names its own primary. */
function config(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return resolvedConfig(overrides);
}

/** A measured iteration numbered `seq`, reading as `outcome` on a `deltaPct` primary. */
function iteration(
  seq: number,
  deltaPct: number,
  outcome: IterationRecord["outcome"],
  targetReached = false,
): IterationRecord {
  return iterationRecord({
    seq,
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
  });
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

/**
 * A keep refused for want of a measurement, numbered `seq`.
 *
 * `keep` writes one of these when nothing has been measured since the last
 * settle, numbering it past every iteration on file — so the number it carries
 * belongs to an iteration that does not exist yet, and may never.
 */
function nothingMeasuredKeep(seq: number): KeepRecord {
  return {
    type: "keep",
    seq,
    at: AT,
    status: "blocked",
    reason: "nothing-measured",
    checks: { configured: true },
  };
}

/** A discard of the iteration numbered `seq`. */
function discard(seq: number): DiscardRecord {
  return { type: "discard", seq, at: AT };
}

/** The four lines every report opens on: the session, its branch, and its two worktrees. */
const HEADER_LINE_COUNT = 4;

/** The report below its header: one line per record it renders, then the totals. */
function bodyLines(report: string): string[] {
  return reportLines(report).slice(HEADER_LINE_COUNT);
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
    committedKeep(1, { commit: KEEP_COMMIT }),
    iteration(2, 9.4, "regressed"),
    discard(2),
    iteration(3, -3.1, "improved"),
    blockedKeep(3),
    iteration(4, 0.1, "no-signal"),
  ];
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
  });

  describe("when a nothing-measured keep took the number a later iteration was minted with", () => {
    it("reads that iteration as unsettled, the blocked keep standing on its own", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root, [
        iteration(1, -7.2, "improved"),
        committedKeep(1, { commit: KEEP_COMMIT }),
        nothingMeasuredKeep(2),
        iteration(2, -3.1, "improved"),
      ]);

      // Act
      const report = statusSession(root, config());

      // Assert
      expect(bodyLines(report)).toStrictEqual([
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "keep-blocked (nothing-measured)",
        "iteration 2 · ✓ -3.1% · unsettled",
        "2 iterations · 1 kept · 0 discarded",
      ]);
    });
  });

  describe("when no iteration ever followed a nothing-measured keep", () => {
    it("renders the blocked keep anyway, counting it as no iteration of its own", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root, [
        iteration(1, -7.2, "improved"),
        committedKeep(1, { commit: KEEP_COMMIT }),
        nothingMeasuredKeep(2),
      ]);

      // Act
      const report = statusSession(root, config());

      // Assert
      expect(bodyLines(report)).toStrictEqual([
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "keep-blocked (nothing-measured)",
        "1 iteration · 1 kept · 0 discarded",
      ]);
    });
  });

  describe("when a stop condition is configured", () => {
    it("forwards the configured stop condition to the footer", () => {
      // Arrange
      const root = freshRoot();
      writeSessionLog(root, fourIterations());

      // Act
      const report = statusSession(root, config({ stop: { maxIterations: 30 } }));

      // Assert
      expect(reportLines(report)).toContain("stop: 4 of 30 iterations");
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
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    // Act
    await program.parseAsync(["node", "cli.js", "status"]);

    // Assert
    const lines = reportLines(stdout());
    expect.soft(lines[0]).toContain(`session ${SESSION_ID}`);
    expect(lines).toContain("4 iterations · 1 kept · 1 discarded");
  });

  /*
   * `status` never benches, so a repository with no `gymrat.json` is not a
   * misconfigured one — it is a repository the command has nothing to read a
   * bench command for and no reason to. Both rows must therefore reach the same
   * hint, the second one only because the missing config never stops it.
   */
  it.each([
    { desc: "the repository holds no session", hasConfigFile: true },
    { desc: "the repository holds neither a session nor a config file", hasConfigFile: false },
  ])("exits 2 with a start hint when $desc", async ({ hasConfigFile }) => {
    // Arrange
    if (hasConfigFile) {
      writeConfigFile();
    }
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockProcessExit();

    // Act
    const parsing = program.parseAsync(["node", "cli.js", "status"]);

    // Assert
    await expect(parsing).rejects.toHaveProperty("exitCode", 2);
    const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
    expect(stderrText).toContain("gymrat start");
  });

  /*
   * `status` styles its lines as it builds them, with no `withColor` wrapper to
   * pin the environment around the render — so `--no-color` holds here only
   * because `suppressColor` clears FORCE_COLOR as well as setting NO_COLOR.
   * Both rows therefore run with FORCE_COLOR set: the flag has to beat it.
   */
  it.each([
    { desc: "styles the report when the terminal takes color", args: [], color: "1", ansi: true },
    {
      desc: "drops every style when --no-color meets a FORCE_COLOR in the environment",
      args: ["--no-color"],
      color: "1",
      ansi: false,
    },
  ])("$desc", async ({ args, color, ansi }) => {
    // Arrange
    vi.stubEnv("NO_COLOR", undefined);
    vi.stubEnv("FORCE_COLOR", color);
    writeSessionLog(repo.dir, fourIterations());
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    // Act
    await program.parseAsync(["node", "cli.js", "status", ...args]);

    // Assert
    expect(ANSI_RE.test(stdout())).toBe(ansi);
  });
});
