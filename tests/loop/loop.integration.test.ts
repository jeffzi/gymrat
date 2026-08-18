import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { finalizeSession } from "../../src/loop/finalize.js";
import { startSession } from "../../src/loop/start.js";
import {
  archivedSessionPath,
  baselineWorktreeDir,
  experimentWorktreeDir,
  lockfilePath,
  sessionJsonlPath,
} from "../../src/session/paths.js";
import type { SessionLogRecord } from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import { BASELINE_LATENCY, commitProject, TUNING_FILE } from "../fixtures/bench-harness.js";
import {
  captureStdout,
  createRunnableProgram,
  exitCodeOf,
  mockProcessExit,
} from "../fixtures/cli-harness.js";
import { createScratchRepo, git, type ScratchRepo } from "../fixtures/scratch-repo.js";
import { committedKeep, iterationRecord, resolvedConfig } from "../fixtures/session-records.js";

/** Generous budget: every command below creates real worktrees and spawns real bench processes. */
const LONG_RUN_TIMEOUT_MS = 180_000;

/** Paired samples per iteration — enough to be a real measurement, few enough to stay quick. */
const SAMPLES = 5;

/** The latency the first edit tunes to, and the one the keep commits. */
const KEPT_LATENCY = 90;
/** The latency the second edit tunes to, and the one the discard throws away. */
const DISCARDED_LATENCY = 80;

/** Content of the throwaway file the discard must erase, distinctive enough to grep history for. */
const DISCARD_MARKER = "discarded-edit-marker";

const DISCARDED_FILE = "discarded-note.txt";

/** Tune the experiment worktree to `latency`, the edit an agent would make between iterations. */
function tuneExperiment(repo: ScratchRepo, latency: number): void {
  fs.writeFileSync(path.join(experimentWorktreeDir(repo.dir), TUNING_FILE), `${String(latency)}\n`);
}

/** Swallow both streams and hand back a reader that drains the stdout collected so far. */
function captureOutput(): () => string {
  return captureStdout({ silenceStderr: true, drainOnRead: true });
}

/**
 * Run one CLI command in a program built from scratch and answer with its exit code.
 *
 * A fresh `createProgram()` per call is what makes the sequence a restart test:
 * nothing a command computed survives into the next one, so every command has to
 * rebuild the session from the log on disk.
 */
async function runCli(argv: readonly string[]): Promise<number> {
  const program = createRunnableProgram({ exitOverride: "all", silent: true });
  try {
    await program.parseAsync(["node", "cli.js", ...argv]);
    return 0;
  } catch (error) {
    return exitCodeOf(error);
  }
}

/** The records of one `type`, narrowed to that record's own shape. */
function pick<T extends SessionLogRecord["type"]>(
  records: readonly SessionLogRecord[],
  type: T,
): Extract<SessionLogRecord, { type: T }>[] {
  return records.filter(
    (record): record is Extract<SessionLogRecord, { type: T }> => record.type === type,
  );
}

/** `count` sample maps, each reporting `latency`. */
function latencySamples(latency: number, count = SAMPLES): { latency: number }[] {
  return Array.from({ length: count }, () => ({ latency }));
}

describe("the gymrat loop – integration", () => {
  describe("when a whole session is driven command by command", () => {
    let repo: ScratchRepo;
    let exitCodes: number[];
    let records: SessionLogRecord[];
    let statusReport: string;
    let branch: string;
    let keptCommit: string;
    let mainStatus: string;
    let history: string;

    beforeAll(async () => {
      repo = createScratchRepo();
      commitProject(repo);
      process.chdir(repo.dir);

      mockProcessExit();
      const takeStdout = captureOutput();

      // Each entry is one command run from a cold start, in the order an agent
      // would drive them: open the session, pin the baseline, edit, measure,
      // keep, edit, measure, throw away.
      exitCodes = [];
      exitCodes.push(await runCli(["start", "main"]));
      exitCodes.push(await runCli(["measure", "main", "--record"]));

      tuneExperiment(repo, KEPT_LATENCY);
      exitCodes.push(await runCli(["iterate"]));
      exitCodes.push(await runCli(["keep", "-m", "tune latency to 90"]));

      takeStdout();
      await runCli(["status", "--no-color"]);
      statusReport = stripAnsi(takeStdout());

      tuneExperiment(repo, DISCARDED_LATENCY);
      fs.writeFileSync(
        path.join(experimentWorktreeDir(repo.dir), DISCARDED_FILE),
        `${DISCARD_MARKER}\n`,
      );
      exitCodes.push(await runCli(["iterate"]));
      exitCodes.push(await runCli(["discard"]));

      vi.restoreAllMocks();

      records = readRecords(sessionJsonlPath(repo.dir));
      branch = pick(records, "session")[0]?.branch ?? "";
      keptCommit = pick(records, "keep")[0]?.commit ?? "";
      mainStatus = git(["status", "--porcelain"], repo.dir);
      history = git(["log", "--all", "-p"], repo.dir);
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      repo.cleanup();
    });

    it("succeeds on every command in the sequence", () => {
      expect(exitCodes).toStrictEqual([0, 0, 0, 0, 0, 0]);
    });

    it("logs the session, the baseline, both iterations, the keep, and the discard in order", () => {
      expect(records.map((record) => record.type)).toStrictEqual([
        "session",
        "baseline",
        "iteration",
        "keep",
        "iteration",
        "discard",
      ]);
    });

    it("numbers the second iteration from the log left by the first", () => {
      // Nothing carried over in memory: seq 2 can only come from reading the log.
      expect.soft(pick(records, "iteration").map((record) => record.seq)).toStrictEqual([1, 2]);
      expect.soft(pick(records, "keep")[0]).toMatchObject({ seq: 1, status: "committed" });
      expect(pick(records, "discard")[0]?.seq).toBe(2);
    });

    it("records what the real bench printed in each worktree it ran in", () => {
      // Adapter sample maps are null-prototype (src/metric-record.ts invariant),
      // so these compare by value rather than by prototype.
      expect.soft(pick(records, "baseline")[0]).toMatchObject({
        label: "main",
        samples: latencySamples(BASELINE_LATENCY),
      });
      const [first, second] = pick(records, "iteration");
      expect.soft(first?.samples).toEqual({
        experiment: latencySamples(KEPT_LATENCY),
        baseline: latencySamples(BASELINE_LATENCY),
      });
      expect(second?.samples).toStrictEqual({
        experiment: latencySamples(DISCARDED_LATENCY),
        baseline: latencySamples(KEPT_LATENCY),
      });
    });

    it("rebuilds the whole session for status out of the log alone", () => {
      const lines = statusReport.split("\n");
      expect.soft(lines[0]).toContain(`session ${pick(records, "session")[0]?.sessionId ?? ""}`);
      expect.soft(lines).toContain(`baseline main · latency ${String(BASELINE_LATENCY)}`);
      expect(lines.join("\n")).toMatch(
        new RegExp(`^iteration 1 · .* · kept ${keptCommit.slice(0, 7)}$`, "m"),
      );
    });

    it("leaves the main working tree clean", () => {
      expect(mainStatus).toBe("");
    });

    it("puts the kept edit on the experiment branch and nothing else", () => {
      expect
        .soft(git(["log", "--format=%H", `main..${branch}`], repo.dir).split("\n"))
        .toStrictEqual([keptCommit]);
      expect(git(["show", `${branch}:${TUNING_FILE}`], repo.dir)).toBe(String(KEPT_LATENCY));
    });

    it("leaves the discarded edit nowhere on disk or in history", () => {
      const worktree = experimentWorktreeDir(repo.dir);
      expect.soft(history).not.toContain(DISCARD_MARKER);
      expect.soft(fs.existsSync(path.join(worktree, DISCARDED_FILE))).toBe(false);
      expect(fs.readFileSync(path.join(worktree, TUNING_FILE), "utf-8").trim()).toBe(
        String(KEPT_LATENCY),
      );
    });
  });

  describe("when a second iterate starts while the first still holds the repository lock", () => {
    let repo: ScratchRepo;
    let gateDir: string;
    let gateFile: string;

    beforeEach(async () => {
      gateDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-gate-"));
      gateFile = path.join(gateDir, "release");
      repo = createScratchRepo();
      commitProject(repo, { gateFile });
      process.chdir(repo.dir);

      mockProcessExit();
      captureOutput();
      await runCli(["start", "main"]);
      tuneExperiment(repo, KEPT_LATENCY);
    }, LONG_RUN_TIMEOUT_MS);

    afterEach(() => {
      vi.restoreAllMocks();
      repo.cleanup();
      fs.rmSync(gateDir, { recursive: true, force: true });
    });

    it(
      "refuses the second run with exit code 2 and lets the first one finish",
      async () => {
        // The gated bench holds the first run open until the test releases it.
        const lockPath = lockfilePath(repo.dir);
        const first = runCli(["iterate"]);
        await vi.waitFor(
          () => {
            expect(fs.existsSync(lockPath)).toBe(true);
          },
          { timeout: 10_000, interval: 25 },
        );

        const second = await runCli(["iterate"]);
        fs.writeFileSync(gateFile, "");

        expect.soft(second).toBe(2);
        expect.soft(await first).toBe(0);
        expect.soft(fs.existsSync(lockPath)).toBe(false);
        expect(pick(readRecords(sessionJsonlPath(repo.dir)), "iteration")).toHaveLength(1);
      },
      LONG_RUN_TIMEOUT_MS,
    );
  });

  describe("when a session is started again after one that finalized without its worktree on disk", () => {
    let repo: ScratchRepo;
    let closedSessionId: string;
    let closedLog: SessionLogRecord[];
    let restarted: ReturnType<typeof startSession>;

    beforeAll(() => {
      repo = createScratchRepo();

      // Drive a whole session: open it, commit the edit a keep would commit,
      // log the iteration and the keep behind it, then close it. The bench never
      // runs — the iteration record stands in for what `iterate` measured.
      const first = startSession(repo.dir, "main", resolvedConfig());
      closedSessionId = first.session.sessionId;

      const worktree = experimentWorktreeDir(repo.dir);
      fs.writeFileSync(path.join(worktree, TUNING_FILE), `${String(KEPT_LATENCY)}\n`);
      git(["add", "-A"], worktree);
      git(["commit", "-m", "tune latency to 90"], worktree);
      appendRecord(sessionJsonlPath(repo.dir), iterationRecord({ seq: 1 }));
      appendRecord(
        sessionJsonlPath(repo.dir),
        committedKeep(1, { commit: git(["rev-parse", "HEAD"], worktree) }),
      );

      // The directory goes before finalize does, so `git worktree remove` finds
      // nothing to take and git keeps its entry for the path.
      fs.rmSync(worktree, { recursive: true, force: true });
      finalizeSession(repo.dir);
      closedLog = readRecords(sessionJsonlPath(repo.dir));

      restarted = startSession(repo.dir, "main", resolvedConfig());
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      repo.cleanup();
    });

    it("opens a fresh session rather than resuming the closed one", () => {
      expect.soft(restarted.resumed).toBe(false);
      expect(restarted.session.sessionId).not.toBe(closedSessionId);
    });

    it("checks out both worktrees of the fresh session", () => {
      expect.soft(fs.existsSync(experimentWorktreeDir(repo.dir))).toBe(true);
      expect(fs.existsSync(baselineWorktreeDir(repo.dir))).toBe(true);
    });

    it("archives the closed session's log under the id it belonged to", () => {
      expect(readRecords(archivedSessionPath(repo.dir, closedSessionId))).toStrictEqual(closedLog);
    });
  });
});
