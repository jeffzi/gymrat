import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createProgram } from "../../src/cli.js";
import type { ResolvedConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import { discardSession, keepSession } from "../../src/loop/settle.js";
import { startSession } from "../../src/loop/start.js";
import {
  baselineWorktreeDir,
  experimentWorktreeDir,
  sessionJsonlPath,
} from "../../src/session/paths.js";
import type { IterationRecord, SessionLogRecord } from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import { captureStdout, mockProcessExit } from "../fixtures/cli-harness.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";
import { committedKeep, iterationRecord, resolvedConfig } from "../fixtures/session-records.js";

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

const ISO_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const CHECKS = "npm test";
/** The run timeout `config()` sets, in the milliseconds `exec` takes. */
const TIMEOUT_MS = 1_800_000;
const CHECKS_STDOUT = "3 tests failed";
const CHECKS_STDERR = "AssertionError: expected 2 to be 3";

/** Run git in `cwd` and return its trimmed stdout. */
function git(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, stdio: "pipe", encoding: "utf-8" }).trim();
}

/** The commit `worktree` currently has checked out. */
function headOf(worktree: string): string {
  return git(["rev-parse", "HEAD"], worktree);
}

/** The porcelain status of `worktree` — empty when nothing is uncommitted. */
function statusOf(worktree: string): string {
  return git(["status", "--porcelain"], worktree);
}

/** A settled run configuration whose checks gate is on. */
function config(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return resolvedConfig({ checks: CHECKS, ...overrides });
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

/** An iteration whose gating metric came back regressed and stayed regressed on the rerun. */
function confirmedRegression(seq: number): IterationRecord {
  return iteration(seq, {
    metrics: { total_ms: metric({ deltaPct: 9.4, verdict: "regressed", confirmed: true }) },
    primary: { kind: "geomean", deltaPct: 9.4 },
    outcome: "regressed",
  });
}

let repo: ScratchRepo;
let originalCwd: string;

/** Open a session in the scratch repo and leave `history` behind its header. */
function startWith(history: SessionLogRecord[] = []): void {
  startSession(repo.dir, "main", config());
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

/** Run `act` and hand back the GymratError it rejected with, failing the test if it threw none. */
async function captureGymratError(act: () => unknown): Promise<GymratError> {
  try {
    await act();
  } catch (error) {
    if (error instanceof GymratError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected the call to fail with a GymratError");
}

beforeEach(() => {
  originalCwd = process.cwd();
  repo = createScratchRepo();
});

afterEach(() => {
  vi.restoreAllMocks();
  execMock.mockReset();
  process.chdir(originalCwd);
  repo.cleanup();
});

describe("keepSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", async () => {
      // Act
      const error = await captureGymratError(() => keepSession(repo.dir, config()));

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
      await keepSession(repo.dir, config());

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
      await keepSession(repo.dir, config());

      // Assert
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(statusOf(worktree)).toBe("");
      expect(git(["rev-parse", "HEAD~1"], worktree)).toBe(before);
    });

    it("appends a committed keep carrying the commit and the message", async () => {
      // Act
      const result = await keepSession(repo.dir, config(), { message: "cache the regex" });

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
      const result = await keepSession(repo.dir, config());

      // Assert
      const baseline = baselineWorktreeDir(repo.dir);
      expect.soft(headOf(baseline)).toBe(result.record.commit);
      expect(git(["rev-parse", "--abbrev-ref", "HEAD"], baseline)).toBe("HEAD");
    });

    it("reports the commit it made", async () => {
      // Act
      const result = await keepSession(repo.dir, config());

      // Assert
      expect(result.report).toContain(headOf(experimentWorktreeDir(repo.dir)).slice(0, 7));
    });

    it("commits a generated message naming the iteration and its primary delta", async () => {
      // Act
      const result = await keepSession(repo.dir, config());

      // Assert
      const subject = git(["log", "-1", "--format=%s"], experimentWorktreeDir(repo.dir));
      expect.soft(result.record.message).toContain("iteration 1");
      expect.soft(result.record.message).toContain("-7.2");
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
      const result = await keepSession(repo.dir, config({ checks: undefined }));

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
      const result = await keepSession(repo.dir, config());

      // Assert
      expect.soft(result.record).toStrictEqual({
        type: "keep",
        seq: 1,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        status: "blocked",
        reason: "checks-failed",
        checks: { configured: true, passed: false },
      });
      expect(lastRecordOf(repo.dir)).toStrictEqual(result.record);
    });

    it("reports both streams of the checks output", async () => {
      // Act
      const result = await keepSession(repo.dir, config());

      // Assert
      expect.soft(result.report).toContain(CHECKS_STDOUT);
      expect(result.report).toContain(CHECKS_STDERR);
    });

    it("leaves the experiment worktree uncommitted", async () => {
      // Arrange
      const before = headOf(experimentWorktreeDir(repo.dir));

      // Act
      await keepSession(repo.dir, config());

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
      const result = await keepSession(repo.dir, config());

      // Assert
      expect.soft(result.record.status).toBe("blocked");
      expect.soft(result.record.reason).toBe("checks-failed");
      expect(result.record.checks).toStrictEqual({ configured: true, passed: false });
    });
  });

  describe("when the last iteration's gating regression was confirmed", () => {
    it("blocks the keep before the checks ever run", async () => {
      // Arrange
      startWith([confirmedRegression(1)]);
      editExperiment();
      checksPass();

      // Act
      const result = await keepSession(repo.dir, config());

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
      const result = await keepSession(repo.dir, config());

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
      const result = await keepSession(repo.dir, config());

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

  describe("when the baseline worktree cannot advance", () => {
    it("writes no keep record, leaving the iteration unsettled", async () => {
      // Arrange
      startWith([iteration(1)]);
      editExperiment();
      checksPass();
      // A linked worktree reaches its repository through this file, so pointing
      // it at a missing directory fails every git command run inside it.
      fs.writeFileSync(path.join(baselineWorktreeDir(repo.dir), ".git"), "gitdir: /nonexistent\n");

      // Act
      const keeping = keepSession(repo.dir, config());

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
        const result = await keepSession(repo.dir, config());

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
  });
});

describe("discardSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", async () => {
      // Act
      const error = await captureGymratError(() => discardSession(repo.dir));

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

  describe("when nothing has been measured since the last settle", () => {
    it.each([
      { description: "no iteration was ever recorded", history: [] },
      {
        description: "the last iteration was already kept",
        history: [iteration(1), committedKeep(1)],
      },
    ] satisfies { description: string; history: SessionLogRecord[] }[])(
      "refuses rather than settle a second time when $description",
      async ({ history }) => {
        // Arrange
        startWith(history);
        editExperiment();
        const before = readRecords(sessionJsonlPath(repo.dir)).length;

        // Act
        await captureGymratError(() => discardSession(repo.dir));

        // Assert
        expect.soft(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(before);
        expect(statusOf(experimentWorktreeDir(repo.dir))).not.toBe("");
      },
    );
  });
});

describe("the settle commands", () => {
  /** A program whose subcommands throw instead of exiting, with stderr silenced. */
  function createSilentProgram(): ReturnType<typeof createProgram> {
    const program = createProgram();
    for (const command of [program, ...program.commands]) {
      command.exitOverride();
      command.configureOutput({ writeErr: () => {} });
    }
    return program;
  }

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
    const program = createSilentProgram();
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
    const program = createSilentProgram();
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
    const program = createSilentProgram();
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
      const program = createSilentProgram();
      const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
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
