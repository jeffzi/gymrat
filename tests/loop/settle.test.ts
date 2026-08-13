import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ResolvedConfig } from "../../src/config.js";
import { discardSession, keepSession } from "../../src/loop/settle.js";
import { startSession } from "../../src/loop/start.js";
import {
  baselineWorktreeDir,
  experimentWorktreeDir,
  sessionJsonlPath,
} from "../../src/session/paths.js";
import type { IterationRecord, KeepRecord, SessionLogRecord } from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import {
  captureStdout,
  createRunnableProgram,
  mockProcessExit,
  stubWrite,
} from "../fixtures/cli-harness.js";
import { ISO_PATTERN } from "../fixtures/constants.js";
import { captureRejectedGymratError } from "../fixtures/errors.js";
import { createScratchRepo, git, type ScratchRepo } from "../fixtures/scratch-repo.js";
import {
  AT,
  committedKeep,
  discardRecord,
  finalizeRecord,
  iterationRecord,
  resolvedConfig,
} from "../fixtures/session-records.js";

type Exec = typeof import("../../src/exec.js").exec;

/**
 * The one boundary this file mocks: the checks command is the consumer's own
 * test suite, which no test can run. Every git operation below is real.
 */
const execMock = vi.hoisted(() => vi.fn<Exec>());

vi.mock("../../src/exec.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/exec.js")>();
  return { ...actual, exec: execMock };
});

type MetricVerdict = NonNullable<IterationRecord["metrics"][string]>;

const CHECKS = "npm test";
/** The run timeout from `resolvedConfig().timeoutSeconds`, in milliseconds. */
const TIMEOUT_MS = 1_800_000;
const CHECKS_STDOUT = "3 tests failed";
const CHECKS_STDERR = "AssertionError: expected 2 to be 3";

/** `resolvedConfig`, defaulted to the checks command every settle test exercises. */
function checksConfig(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return resolvedConfig({ checks: CHECKS, ...overrides });
}

/**
 * 200 lines of exactly 100 bytes each, every one numbered behind `prefix`.
 *
 * The uniform line width puts the relay's byte budget on a line a test can name:
 * 81 lines are 8100 bytes and fit the 8192-byte budget the hook relay uses, an
 * 82nd would take it to 8200 and overrun it.
 */
function longOutput(prefix: string): string {
  return Array.from(
    { length: 200 },
    (_, index) => `${`${prefix}-${String(index).padStart(3, "0")}`.padEnd(99, ".")}\n`,
  ).join("");
}

const LONG_STDOUT = longOutput("out");
const LONG_STDERR = longOutput("err");

/** The commit `worktree` currently has checked out. */
function headOf(worktree: string): string {
  return git(["rev-parse", "HEAD"], worktree);
}

/** The porcelain status of `worktree` — empty when nothing is uncommitted. */
function statusOf(worktree: string): string {
  return git(["status", "--porcelain"], worktree);
}

/** A metric verdict the engine produces, improved and gating unless overridden. */
function metric(overrides: Partial<MetricVerdict> = {}): MetricVerdict {
  return {
    deltaPct: -7.2,
    verdict: "improved",
    method: "signed-rank",
    p: 0.002,
    noisePct: 1.4,
    gating: true,
    confirmed: false,
    ...overrides,
  };
}

/** A measured iteration numbered `seq`, improved unless a test says otherwise. */
function iteration(seq: number, overrides: Partial<IterationRecord> = {}): IterationRecord {
  return iterationRecord({ seq, ...overrides });
}

/** An iteration numbered `seq` whose deltas a zero baseline median left undefined. */
function undefinedDelta(seq: number): IterationRecord {
  return iteration(seq, {
    metrics: { total_ms: metric({ deltaPct: null, verdict: "no-signal" }) },
    primary: { kind: "geomean", deltaPct: null },
    outcome: "no-signal",
  });
}

/** An iteration whose gating metric came back regressed and stayed regressed on the rerun. */
function confirmedRegression(seq: number): IterationRecord {
  return iteration(seq, {
    metrics: { total_ms: metric({ deltaPct: 9.4, verdict: "regressed", confirmed: true }) },
    primary: { kind: "geomean", deltaPct: 9.4 },
    outcome: "regressed",
  });
}

/** The rerun samples a filtered bench reports when it only ever emits `total_ms`. */
const RERUN_SAMPLES = {
  experiment: [{ total_ms: 14_120 }],
  baseline: [{ total_ms: 15_170 }],
};

/**
 * An iteration whose gating `alloc_bytes` regressed on the first run and came
 * back missing from the confirmation rerun — silence, not disagreement.
 */
function unmeasuredRegression(seq: number): IterationRecord {
  return iteration(seq, {
    metrics: {
      total_ms: metric(),
      alloc_bytes: metric({ deltaPct: 9.4, verdict: "regressed" }),
    },
    primary: { kind: "geomean", deltaPct: 9.4 },
    outcome: "regressed",
    confirm: {
      ran: true,
      filtered: ["total_ms", "alloc_bytes"],
      absent: ["alloc_bytes"],
      samples: RERUN_SAMPLES,
    },
  });
}

/**
 * An iteration whose gating regression the rerun re-measured and would not repeat.
 *
 * The metric carries the verdict `iterate` writes after the demotion — `no-signal`
 * on the first run's delta — so the record is the shape a real rerun leaves behind
 * rather than one only this file produces. `confirm` is the parameter because the
 * rerun can report the metric or come from a log written before `absent` existed.
 */
function rerunConfirm(confirm: IterationRecord["confirm"]): IterationRecord {
  return iteration(1, {
    metrics: { total_ms: metric({ deltaPct: 9.4, verdict: "no-signal" }) },
    outcome: "no-signal",
    confirm,
  });
}

/** The keep a gating regression refused, numbered with the iteration it refused. */
function gatingBlock(seq: number): KeepRecord {
  return {
    type: "keep",
    seq,
    at: AT,
    status: "blocked",
    reason: "gating-regression",
    checks: { configured: true },
  };
}

let repo: ScratchRepo;

/** Open a session in the scratch repo and leave `history` behind its header. */
function startWith(history: SessionLogRecord[] = []): void {
  startSession(repo.dir, "main", checksConfig());
  for (const record of history) {
    appendRecord(sessionJsonlPath(repo.dir), record);
  }
}

/** Leave a tracked edit and an untracked file in the experiment worktree. */
function editExperiment(): void {
  const worktree = experimentWorktreeDir(repo.dir);
  fs.writeFileSync(path.join(worktree, "README.md"), "# edited by the agent\n");
  fs.writeFileSync(path.join(worktree, "scratch.txt"), "notes\n");
}

/** Answer the checks command with a clean run. */
function checksPass(): void {
  execMock.mockResolvedValue({ stdout: "10 passed", stderr: "", exitCode: 0 });
}

/** Answer the checks command with a failing run that wrote to both streams. */
function checksFail(): void {
  execMock.mockResolvedValue({ stdout: CHECKS_STDOUT, stderr: CHECKS_STDERR, exitCode: 1 });
}

/** The record `root`'s log ends on, failing the test when the log is empty. */
function lastRecordOf(root: string): SessionLogRecord {
  const last = readRecords(sessionJsonlPath(root)).at(-1);
  if (last === undefined) {
    throw new Error(`expected a record in ${sessionJsonlPath(root)}`);
  }
  return last;
}

beforeEach(() => {
  repo = createScratchRepo();
});

afterEach(() => {
  vi.restoreAllMocks();
  execMock.mockReset();
  repo.cleanup();
});

describe("keepSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", async () => {
      // Act
      const error = await captureRejectedGymratError(() => keepSession(repo.dir, checksConfig()));

      // Assert
      expect.soft(error.hint).toContain("gymrat start");
      expect(execMock).not.toHaveBeenCalled();
    });
  });

  describe("when the checks pass on an unsettled iteration", () => {
    beforeEach(() => {
      startWith([iteration(1)]);
      editExperiment();
      checksPass();
    });

    it("runs the configured checks in the experiment worktree under the run timeout", async () => {
      // Act
      await keepSession(repo.dir, checksConfig());

      // Assert
      expect(execMock).toHaveBeenCalledWith(CHECKS, {
        cwd: experimentWorktreeDir(repo.dir),
        timeoutMs: TIMEOUT_MS,
      });
    });

    it("commits every tracked and untracked change in the experiment worktree", async () => {
      // Arrange
      const before = headOf(experimentWorktreeDir(repo.dir));

      // Act
      await keepSession(repo.dir, checksConfig());

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(statusOf(worktree)).toBe("");
      expect(git(["rev-parse", "HEAD~1"], worktree)).toBe(before);
    });

    it("appends a committed keep carrying the commit and the message", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig(), {
        message: "cache the regex",
      });

      // Assert
      expect.soft(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "committed",
        commit: headOf(experimentWorktreeDir(repo.dir)),
        message: "cache the regex",
        checks: { configured: true, passed: true },
      });
      expect(lastRecordOf(repo.dir)).toStrictEqual(result.record);
    });

    it("advances the baseline worktree to the kept commit, still detached", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      const baseline = baselineWorktreeDir(repo.dir);
      expect.soft(headOf(baseline)).toBe(result.record.commit);
      expect(git(["rev-parse", "--abbrev-ref", "HEAD"], baseline)).toBe("HEAD");
    });

    it("reports the commit it made", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect(result.report).toContain(headOf(experimentWorktreeDir(repo.dir)).slice(0, 7));
    });

    it("commits a generated message naming the iteration and its primary delta", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      const subject = git(["log", "-1", "--format=%s"], experimentWorktreeDir(repo.dir));
      expect.soft(result.record.message).toContain("iteration 1");
      expect.soft(result.record.message).toContain("-7.2");
      expect(subject).toBe(result.record.message);
    });
  });

  describe("when the last iteration's primary delta was left undefined", () => {
    beforeEach(() => {
      startWith([undefinedDelta(1)]);
      editExperiment();
      checksPass();
    });

    it("commits a generated message that says so instead of trailing off", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      const subject = git(["log", "-1", "--format=%s"], experimentWorktreeDir(repo.dir));
      expect.soft(result.record.message).toBe("iteration 1: geomean delta undefined");
      expect(subject).toBe(result.record.message);
    });
  });

  describe("when no checks command is configured", () => {
    beforeEach(() => {
      startWith([iteration(1)]);
      editExperiment();
    });

    it("keeps anyway and records that the gate was off", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig({ checks: undefined }));

      // Assert
      expect.soft(execMock).not.toHaveBeenCalled();
      expect(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "committed",
        commit: headOf(experimentWorktreeDir(repo.dir)),
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        message: expect.any(String),
        checks: { configured: false },
      });
    });
  });

  describe("when the checks do not pass", () => {
    beforeEach(() => {
      startWith([iteration(1)]);
      editExperiment();
      checksFail();
    });

    it("appends a keep blocked on the failed checks", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "blocked",
        reason: "checks-failed",
        checks: {
          configured: true,
          passed: false,
          stdoutBytes: Buffer.byteLength(CHECKS_STDOUT, "utf-8"),
          stderrBytes: Buffer.byteLength(CHECKS_STDERR, "utf-8"),
        },
      });
      expect(lastRecordOf(repo.dir)).toStrictEqual(result.record);
    });

    it("reports both streams of the checks output", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(result.report).toContain(CHECKS_STDOUT);
      expect(result.report).toContain(CHECKS_STDERR);
    });

    it("leaves the experiment worktree uncommitted", async () => {
      // Arrange
      const before = headOf(experimentWorktreeDir(repo.dir));

      // Act
      await keepSession(repo.dir, checksConfig());

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(headOf(worktree)).toBe(before);
      expect(statusOf(worktree)).not.toBe("");
    });

    it("blocks the same way when the checks command times out", async () => {
      // Arrange
      execMock.mockResolvedValue({
        kind: "timeout",
        stdout: CHECKS_STDOUT,
        stderr: CHECKS_STDERR,
        timeoutMs: TIMEOUT_MS,
      });

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(result.record.status).toBe("blocked");
      expect.soft(result.record.reason).toBe("checks-failed");
      expect(result.record.checks).toStrictEqual({
        configured: true,
        passed: false,
        stdoutBytes: Buffer.byteLength(CHECKS_STDOUT, "utf-8"),
        stderrBytes: Buffer.byteLength(CHECKS_STDERR, "utf-8"),
      });
    });
  });

  describe("when the failing checks printed more than the relay allows", () => {
    beforeEach(() => {
      startWith([iteration(1)]);
      editExperiment();
      execMock.mockResolvedValue({ stdout: LONG_STDOUT, stderr: LONG_STDERR, exitCode: 1 });
    });

    it.each([
      { stream: "stdout", prefix: "out" },
      { stream: "stderr", prefix: "err" },
    ])("cuts the relayed $stream back to the last whole line in the budget", async ({ prefix }) => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert - 81 of the 100-byte lines fit the byte budget the hook relay
      // uses, an 82nd overruns it, so the cut lands between the two.
      expect.soft(result.report).toContain(`${prefix}-000`);
      expect.soft(result.report).toContain(`${prefix}-080`);
      expect(result.report).not.toContain(`${prefix}-081`);
    });

    it("records the true byte counts of the output it cut", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert - the counts are what the command printed rather than what the
      // report relayed, so a reader of the log can tell the relay was cut.
      expect.soft(result.record.checks).toStrictEqual({
        configured: true,
        passed: false,
        stdoutBytes: Buffer.byteLength(LONG_STDOUT, "utf-8"),
        stderrBytes: Buffer.byteLength(LONG_STDERR, "utf-8"),
      });
      expect(lastRecordOf(repo.dir)).toStrictEqual(result.record);
    });
  });

  describe("when the last iteration's gating regression was confirmed", () => {
    it("blocks the keep before the checks ever run", async () => {
      // Arrange
      startWith([confirmedRegression(1)]);
      editExperiment();
      checksPass();

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(execMock).not.toHaveBeenCalled();
      expect.soft(statusOf(experimentWorktreeDir(repo.dir))).not.toBe("");
      expect(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "blocked",
        reason: "gating-regression",
        checks: { configured: true },
      });
    });

    it("refuses without claiming anything went unmeasured", async () => {
      // Arrange
      startWith([confirmedRegression(1)]);
      editExperiment();
      checksPass();

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(result.report).not.toMatch(/not measured/i);
      expect(result.report).not.toMatch(/filter/i);
    });

    it("blocks a log written before the absent field on the confirmation alone", async () => {
      // Arrange
      startWith([
        iteration(1, {
          metrics: { total_ms: metric({ deltaPct: 9.4, verdict: "regressed", confirmed: true }) },
          primary: { kind: "geomean", deltaPct: 9.4 },
          outcome: "regressed",
          confirm: { ran: true, filtered: ["total_ms"], samples: RERUN_SAMPLES },
        }),
      ]);
      editExperiment();
      checksPass();

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(result.record.status).toBe("blocked");
      expect.soft(result.record.reason).toBe("gating-regression");
      expect(result.report).not.toMatch(/not measured/i);
    });

    it("keeps an iteration whose regression the rerun did not confirm", async () => {
      // Arrange
      startWith([
        iteration(1, {
          metrics: { total_ms: metric({ deltaPct: 9.4, verdict: "regressed" }) },
          outcome: "no-signal",
        }),
      ]);
      editExperiment();
      checksPass();

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect(result.record.status).toBe("committed");
    });
  });

  describe("when the last iteration's gating exact metric regressed", () => {
    it("blocks the keep even though the deterministic rerun never marked it confirmed", async () => {
      // Arrange
      startWith([
        iteration(1, {
          metrics: {
            total_ms: metric({ deltaPct: 9.4, verdict: "regressed", method: "exact" }),
          },
          primary: { kind: "geomean", deltaPct: 9.4 },
          outcome: "regressed",
        }),
      ]);
      editExperiment();
      checksPass();

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(execMock).not.toHaveBeenCalled();
      expect(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "blocked",
        reason: "gating-regression",
        checks: { configured: true },
      });
    });
  });

  describe("when the confirmation rerun never measured the gating regression", () => {
    beforeEach(() => {
      startWith([unmeasuredRegression(1)]);
      editExperiment();
      checksPass();
    });

    it("blocks the keep before the checks ever run", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(execMock).not.toHaveBeenCalled();
      expect(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "blocked",
        reason: "gating-regression",
        checks: { configured: true },
      });
    });

    it("names the unmeasured metric and points the agent at the filter", async () => {
      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert
      expect.soft(result.report).toContain("alloc_bytes");
      expect.soft(result.report).toMatch(/not measured on the confirmation rerun/i);
      expect.soft(result.report).toMatch(/filter/i);
      expect(result.report).toMatch(/discard/i);
    });
  });

  describe("when the confirmation rerun measured the gating regression away", () => {
    it.each([
      {
        description: "the rerun reported the metric",
        confirm: {
          ran: true,
          filtered: ["total_ms"],
          absent: [],
          samples: RERUN_SAMPLES,
        },
      },
      {
        description: "the log predates the absent field",
        confirm: { ran: true, filtered: ["total_ms"], samples: RERUN_SAMPLES },
      },
    ] satisfies { description: string; confirm: IterationRecord["confirm"] }[])(
      "keeps the iteration when $description",
      async ({ confirm }) => {
        // Arrange
        startWith([rerunConfirm(confirm)]);
        editExperiment();
        checksPass();

        // Act
        const result = await keepSession(repo.dir, checksConfig());

        // Assert
        expect(result.record.status).toBe("committed");
      },
    );
  });

  describe("when the baseline worktree cannot advance", () => {
    it("writes no keep record, leaving the iteration unsettled", async () => {
      // Arrange
      startWith([iteration(1)]);
      editExperiment();
      checksPass();
      // Sabotage the baseline so git commands inside it fail. On POSIX we can
      // overwrite the .git pointer directly; on Windows git holds the file
      // locked, so we force-remove the whole worktree instead.
      const baseline = baselineWorktreeDir(repo.dir);
      if (process.platform === "win32") {
        execFileSync("git", ["worktree", "remove", "--force", baseline], {
          cwd: repo.dir,
          stdio: "pipe",
        });
      } else {
        fs.writeFileSync(path.join(baseline, ".git"), "gitdir: /nonexistent\n");
      }

      // Act
      const keeping = keepSession(repo.dir, checksConfig());

      // Assert
      await expect.soft(keeping).rejects.toThrow();
      expect(lastRecordOf(repo.dir)).toMatchObject({ type: "iteration", seq: 1 });
    });
  });

  describe("when nothing has been measured since the last settle", () => {
    // The blocked keep settles nothing, so its seq must not alias an iteration
    // the log already settled — it takes the number no iteration has used yet.
    it.each([
      { description: "no iteration was ever recorded", history: [], seq: 1 },
      {
        description: "the last iteration was already kept",
        history: [iteration(1), committedKeep(1)],
        seq: 2,
      },
    ] satisfies { description: string; history: SessionLogRecord[]; seq: number }[])(
      "refuses with a nothing-measured keep when $description",
      async ({ history, seq }) => {
        // Arrange
        startWith(history);
        editExperiment();
        checksPass();

        // Act
        const result = await keepSession(repo.dir, checksConfig());

        // Assert
        expect.soft(execMock).not.toHaveBeenCalled();
        expect(result.record).toStrictEqual({
          type: "keep",
          seq,
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          at: expect.stringMatching(ISO_PATTERN),
          status: "blocked",
          reason: "nothing-measured",
          checks: { configured: true },
        });
      },
    );

    it("numbers a second refusal past the first instead of reusing its number", async () => {
      // Arrange
      startWith();
      editExperiment();
      checksPass();
      await keepSession(repo.dir, checksConfig());

      // Act
      const result = await keepSession(repo.dir, checksConfig());

      // Assert - a consumer walking the raw log sees two distinct records, not
      // one number written twice.
      const keeps = readRecords(sessionJsonlPath(repo.dir)).filter(
        (record) => record.type === "keep",
      );
      expect.soft(keeps.map((record) => record.seq)).toStrictEqual([1, 2]);
      expect(result.record.seq).toBe(2);
    });
  });
});

describe("discardSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", async () => {
      // Act
      const error = await captureRejectedGymratError(() => discardSession(repo.dir));

      // Assert
      expect(error.hint).toContain("gymrat start");
    });
  });

  describe("when the experiment worktree carries an unsettled edit", () => {
    beforeEach(() => {
      startWith([iteration(1)]);
      editExperiment();
    });

    it("throws away tracked edits and untracked files alike", () => {
      // Act
      discardSession(repo.dir);

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(fs.readFileSync(path.join(worktree, "README.md"), "utf-8")).toBe("# Test Repo\n");
      expect.soft(fs.existsSync(path.join(worktree, "scratch.txt"))).toBe(false);
      expect(statusOf(worktree)).toBe("");
    });

    it("appends a discard naming the iteration it settled", () => {
      // Act
      const result = discardSession(repo.dir);

      // Assert
      expect.soft(result.record).toStrictEqual({
        type: "discard",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
      });
      expect(lastRecordOf(repo.dir)).toStrictEqual(result.record);
    });
  });

  describe("when the last iteration's primary delta was left undefined", () => {
    it("throws it away like any other", () => {
      // Arrange
      startWith([undefinedDelta(1)]);
      editExperiment();

      // Act
      const result = discardSession(repo.dir);

      // Assert
      expect(result.record.seq).toBe(1);
    });
  });

  describe("when the experiment worktree is clean", () => {
    it("records the discard anyway", () => {
      // Arrange
      startWith([iteration(1)]);

      // Act
      const result = discardSession(repo.dir);

      // Assert
      expect(lastRecordOf(repo.dir)).toStrictEqual(result.record);
    });
  });

  describe("when the last keep was blocked for a gating regression", () => {
    beforeEach(() => {
      startWith([confirmedRegression(1), gatingBlock(1)]);
      editExperiment();
    });

    it("throws away the edit the regression blocked", () => {
      // Act
      discardSession(repo.dir);

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(fs.readFileSync(path.join(worktree, "README.md"), "utf-8")).toBe("# Test Repo\n");
      expect.soft(fs.existsSync(path.join(worktree, "scratch.txt"))).toBe(false);
      expect(statusOf(worktree)).toBe("");
    });

    it("appends a discard numbered past the block, leaving the block in history", () => {
      // Act
      const result = discardSession(repo.dir);

      // Assert - the block already settled iteration 1, so the discard takes the
      // number no iteration has used yet. Reusing 1 would overwrite the block in
      // `status`, which reports the last settling record to carry an iteration's
      // number, and the block is history the log has to keep showing.
      expect.soft(result.record).toStrictEqual({
        type: "discard",
        seq: 2,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
      });
      expect(readRecords(sessionJsonlPath(repo.dir)).slice(-2)).toStrictEqual([
        gatingBlock(1),
        result.record,
      ]);
    });
  });

  describe("when the last keep was blocked for a regression the rerun never measured", () => {
    it("throws away the edit the block refused, numbered past it", () => {
      // Arrange
      startWith([unmeasuredRegression(1), gatingBlock(1)]);
      editExperiment();

      // Act
      const result = discardSession(repo.dir);

      // Assert
      expect.soft(statusOf(experimentWorktreeDir(repo.dir))).toBe("");
      expect(result.record).toStrictEqual({
        type: "discard",
        seq: 2,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
      });
    });
  });

  describe("when a keep retried after a gating block refused for want of a measurement", () => {
    beforeEach(async () => {
      startWith([confirmedRegression(1), gatingBlock(1)]);
      editExperiment();
      checksPass();
      await keepSession(repo.dir, checksConfig());
    });

    it("throws away the edit the refusal left standing", () => {
      // Act
      discardSession(repo.dir);

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(fs.readFileSync(path.join(worktree, "README.md"), "utf-8")).toBe("# Test Repo\n");
      expect.soft(fs.existsSync(path.join(worktree, "scratch.txt"))).toBe(false);
      expect(statusOf(worktree)).toBe("");
    });

    it("appends the discard after the refusal, leaving the refusal in history", () => {
      // Act
      const result = discardSession(repo.dir);

      // Assert
      const tail = readRecords(sessionJsonlPath(repo.dir)).slice(-2);
      expect.soft(tail[0]).toMatchObject({
        type: "keep",
        status: "blocked",
        reason: "nothing-measured",
      });
      expect(tail[1]).toStrictEqual(result.record);
    });
  });

  describe("when nothing has been measured since the last settle", () => {
    it.each([
      { description: "no iteration was ever recorded", history: [] },
      {
        description: "the last iteration was already kept",
        history: [iteration(1), committedKeep(1)],
      },
      {
        description: "the gating regression it blocked was already discarded",
        history: [confirmedRegression(1), gatingBlock(1), discardRecord(2)],
      },
    ] satisfies { description: string; history: SessionLogRecord[] }[])(
      "refuses rather than settle a second time when $description",
      async ({ history }) => {
        // Arrange
        startWith(history);
        editExperiment();
        const before = readRecords(sessionJsonlPath(repo.dir)).length;

        // Act
        await captureRejectedGymratError(() => discardSession(repo.dir));

        // Assert
        expect.soft(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(before);
        expect(statusOf(experimentWorktreeDir(repo.dir))).not.toBe("");
      },
    );
  });
});

describe("when the session on disk was finalized", () => {
  it.each([
    {
      command: "keepSession",
      settle: (): unknown => keepSession(repo.dir, checksConfig()),
    },
    { command: "discardSession", settle: (): unknown => discardSession(repo.dir) },
  ])(
    "$command refuses with a hint pointing at a fresh start, writing nothing",
    async ({ settle }) => {
      // Arrange
      startWith([iteration(1), committedKeep(1), finalizeRecord()]);
      editExperiment();
      const before = readRecords(sessionJsonlPath(repo.dir)).length;

      // Act
      const error = await captureRejectedGymratError(settle);

      // Assert
      expect.soft(error.hint).toContain("gymrat start");
      expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(before);
    },
  );
});

describe("the settle commands", () => {
  /** Write the config file the settle commands read their checks gate from. */
  function writeConfigFile(): void {
    fs.writeFileSync(
      path.join(repo.dir, "gymrat.json"),
      JSON.stringify({ bench: "npm run bench", checks: CHECKS }),
    );
  }

  it("keeps the session in the repository it runs in and prints the commit", async () => {
    // Arrange
    startWith([iteration(1)]);
    editExperiment();
    checksPass();
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    // Act
    await program.parseAsync(["node", "cli.js", "keep", "-m", "cache the regex"]);

    // Assert
    const record = lastRecordOf(repo.dir);
    expect.soft(record).toMatchObject({ type: "keep", status: "committed" });
    expect(stdout()).toContain(headOf(experimentWorktreeDir(repo.dir)).slice(0, 7));
  });

  it("exits 1 when the checks block the keep", async () => {
    // Arrange
    startWith([iteration(1)]);
    editExperiment();
    checksFail();
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    captureStdout();
    mockProcessExit();

    // Act
    const parsing = program.parseAsync(["node", "cli.js", "keep"]);

    // Assert
    await expect(parsing).rejects.toHaveProperty("exitCode", 1);
    expect(lastRecordOf(repo.dir)).toMatchObject({ status: "blocked", reason: "checks-failed" });
  });

  it("discards the session in the repository it runs in", async () => {
    // Arrange
    startWith([iteration(1)]);
    editExperiment();
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    // Act
    await program.parseAsync(["node", "cli.js", "discard"]);

    // Assert
    expect.soft(statusOf(experimentWorktreeDir(repo.dir))).toBe("");
    expect.soft(lastRecordOf(repo.dir)).toMatchObject({ type: "discard" });
    expect(stdout()).toMatch(/discard/i);
  });

  it.each([{ command: "keep" }, { command: "discard" }])(
    "exits 2 with a start hint when $command runs without a session",
    async ({ command }) => {
      // Arrange
      writeConfigFile();
      process.chdir(repo.dir);
      const program = createRunnableProgram({ exitOverride: "all", silent: true });
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      // Act
      const parsing = program.parseAsync(["node", "cli.js", command]);

      // Assert
      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
      expect(stderrText).toContain("gymrat start");
    },
  );
});
