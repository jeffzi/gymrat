import fs from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ResolvedConfig } from "../../src/config.js";
import { discardSession, keepSession } from "../../src/loop/settle.js";
import { startSession } from "../../src/loop/start.js";
import { experimentWorktreeDir, sessionJsonlPath } from "../../src/session/paths.js";
import type { IterationRecord, SessionLogRecord } from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import {
  captureStdout,
  createRunnableProgram,
  mockProcessExit,
  stubWrite,
} from "../fixtures/cli-harness.js";
import { captureRejectedGymratError } from "../fixtures/errors.js";
import { createScratchRepo, git, type ScratchRepo } from "../fixtures/scratch-repo.js";
import {
  type LoosePartial,
  committedKeep,
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

const CHECKS = "npm test";
const CHECKS_STDOUT = "3 tests failed";
const CHECKS_STDERR = "AssertionError: expected 2 to be 3";

/** `resolvedConfig`, defaulted to the checks command every settle test exercises. */
function checksConfig(overrides: LoosePartial<ResolvedConfig> = {}): ResolvedConfig {
  return resolvedConfig({ checks: CHECKS, ...overrides });
}

/** The commit `worktree` currently has checked out. */
function headOf(worktree: string): string {
  return git(["rev-parse", "HEAD"], worktree);
}

/** The porcelain status of `worktree` — empty when nothing is uncommitted. */
function statusOf(worktree: string): string {
  return git(["status", "--porcelain"], worktree);
}

/** A measured iteration numbered `seq`, improved unless a test says otherwise. */
function iteration(seq: number, overrides: LoosePartial<IterationRecord> = {}): IterationRecord {
  return iterationRecord({ seq, ...overrides });
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
  execMock.mockResolvedValue({
    stdout: "10 passed",
    stderr: "",
    exitCode: 0,
    stdoutBytes: Buffer.byteLength("10 passed", "utf-8"),
    stderrBytes: 0,
  });
}

/** Answer the checks command with a failing run that wrote to both streams. */
function checksFail(): void {
  execMock.mockResolvedValue({
    stdout: CHECKS_STDOUT,
    stderr: CHECKS_STDERR,
    exitCode: 1,
    stdoutBytes: Buffer.byteLength(CHECKS_STDOUT, "utf-8"),
    stderrBytes: Buffer.byteLength(CHECKS_STDERR, "utf-8"),
  });
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
      startWith([iteration(1), committedKeep(1), finalizeRecord()]);
      editExperiment();
      const before = readRecords(sessionJsonlPath(repo.dir)).length;

      const error = await captureRejectedGymratError(settle);

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
    startWith([iteration(1)]);
    editExperiment();
    checksPass();
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    await program.parseAsync(["node", "cli.js", "keep", "-m", "cache the regex"]);

    const record = lastRecordOf(repo.dir);
    expect.soft(record).toMatchObject({ type: "keep", status: "committed" });
    expect(stdout()).toContain(headOf(experimentWorktreeDir(repo.dir)).slice(0, 7));
  });

  it("exits 1 when nothing to commit blocks the keep", async () => {
    startWith([iteration(1)]);
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    captureStdout();
    mockProcessExit();

    const parsing = program.parseAsync(["node", "cli.js", "keep"]);

    await expect(parsing).rejects.toHaveProperty("exitCode", 1);
    expect(lastRecordOf(repo.dir)).toMatchObject({
      status: "blocked",
      reason: "nothing-to-commit",
    });
  });

  it("exits 1 when the checks block the keep", async () => {
    startWith([iteration(1)]);
    editExperiment();
    checksFail();
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    captureStdout();
    mockProcessExit();

    const parsing = program.parseAsync(["node", "cli.js", "keep"]);

    await expect(parsing).rejects.toHaveProperty("exitCode", 1);
    expect(lastRecordOf(repo.dir)).toMatchObject({ status: "blocked", reason: "checks-failed" });
  });

  it("discards the session in the repository it runs in", async () => {
    startWith([iteration(1)]);
    editExperiment();
    writeConfigFile();
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    await program.parseAsync(["node", "cli.js", "discard"]);

    expect.soft(statusOf(experimentWorktreeDir(repo.dir))).toBe("");
    expect.soft(lastRecordOf(repo.dir)).toMatchObject({ type: "discard" });
    expect(stdout()).toMatch(/discard/i);
  });

  it.each([{ command: "keep" }, { command: "discard" }])(
    "exits 2 with a start hint when $command runs without a session",
    async ({ command }) => {
      writeConfigFile();
      process.chdir(repo.dir);
      const program = createRunnableProgram({ exitOverride: "all", silent: true });
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(["node", "cli.js", command]);

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
      expect(stderrText).toContain("gymrat start");
    },
  );
});
