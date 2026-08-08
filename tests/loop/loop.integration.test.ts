import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { createProgram } from "../../src/cli.js";
import { experimentWorktreeDir, lockfilePath, sessionJsonlPath } from "../../src/session/paths.js";
import type { SessionLogRecord } from "../../src/session/records.js";
import { readRecords } from "../../src/session/store.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";

/** Generous budget: every command below creates real worktrees and spawns real bench processes. */
const LONG_RUN_TIMEOUT_MS = 180_000;

/** Paired samples per iteration — enough to be a real measurement, few enough to stay quick. */
const SAMPLES = 5;

/** The latency the bench reports from a checkout that carries no tuning file. */
const BASELINE_LATENCY = 100;
/** The latency the first edit tunes to, and the one the keep commits. */
const KEPT_LATENCY = 90;
/** The latency the second edit tunes to, and the one the discard throws away. */
const DISCARDED_LATENCY = 80;

/** Content of the throwaway file the discard must erase, distinctive enough to grep history for. */
const DISCARD_MARKER = "discarded-edit-marker";

const TUNING_FILE = "tuning.txt";
const DISCARDED_FILE = "discarded-note.txt";
const BENCH_FILE = "bench.js";

/**
 * A `metric-lines` bench that reports whatever `tuning.txt` holds, defaulting to
 * the untuned latency when the checkout has no tuning file.
 *
 * Written as CommonJS: the scratch repo has no `package.json`, so node reads a
 * `.js` file there as CommonJS on every platform.
 *
 * With `gateFile`, the bench blocks until that file appears — the only way to
 * hold a run open long enough for a second command to collide with it without
 * betting on a sleep being longer than the first run.
 */
function benchScript(gateFile?: string): string {
  const lines = ['const fs = require("node:fs");'];
  if (gateFile !== undefined) {
    lines.push(
      `const gate = ${JSON.stringify(gateFile)};`,
      "const idle = new Int32Array(new SharedArrayBuffer(4));",
      "const deadline = Date.now() + 60000;",
      "while (!fs.existsSync(gate) && Date.now() < deadline) { Atomics.wait(idle, 0, 0, 25); }",
    );
  }
  lines.push(
    `const tuned = fs.existsSync(${JSON.stringify(TUNING_FILE)})`,
    `  ? fs.readFileSync(${JSON.stringify(TUNING_FILE)}, "utf8").trim()`,
    `  : "${String(BASELINE_LATENCY)}";`,
    'process.stdout.write("METRIC latency=" + tuned + "\\n");',
  );
  return `${lines.join("\n")}\n`;
}

/** Run git in `cwd` and return its trimmed stdout. */
function git(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, stdio: "pipe", encoding: "utf-8" }).trim();
}

/**
 * Commit the bench, the config, and a gitignore for gymrat's own directory, so
 * every worktree the loop checks out carries a runnable bench and the main tree
 * stays clean while the session runs.
 */
function commitProject(repo: ScratchRepo, gateFile?: string): void {
  const files: Record<string, string> = {
    ".gitignore": ".gymrat/\n",
    [BENCH_FILE]: benchScript(gateFile),
    "gymrat.json": `${JSON.stringify({
      bench: `node ${BENCH_FILE}`,
      adapter: "metric-lines",
      samples: SAMPLES,
      timeoutSeconds: 120,
    })}\n`,
  };
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(repo.dir, name), content);
  }
  execFileSync("git", ["add", ...Object.keys(files)], { cwd: repo.dir, stdio: "pipe" });
  execFileSync("git", ["commit", "-m", "bench harness"], { cwd: repo.dir, stdio: "pipe" });
}

/** Tune the experiment worktree to `latency`, the edit an agent would make between iterations. */
function tuneExperiment(repo: ScratchRepo, latency: number): void {
  fs.writeFileSync(path.join(experimentWorktreeDir(repo.dir), TUNING_FILE), `${String(latency)}\n`);
}

/** The exit code carried by the error a mocked `process.exit` threw. */
function exitCodeOf(error: unknown): number {
  if (error instanceof Error && "exitCode" in error && typeof error.exitCode === "number") {
    return error.exitCode;
  }
  throw error;
}

/** Turn `process.exit` into a catchable rejection carrying the intended code. */
function mockProcessExit(): void {
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- vitest mock requires cast
  vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
    throw Object.assign(new Error(`process.exit(${String(code)})`), { exitCode: code });
  }) as never);
}

/** Swallow both streams and hand back a reader that drains the stdout collected so far. */
function captureOutput(): () => string {
  let stdout = "";
  vi.spyOn(process.stdout, "write").mockImplementation((chunk) => {
    stdout += String(chunk);
    return true;
  });
  vi.spyOn(process.stderr, "write").mockReturnValue(true);
  return () => {
    const collected = stdout;
    stdout = "";
    return collected;
  };
}

/**
 * Run one CLI command in a program built from scratch and answer with its exit code.
 *
 * A fresh `createProgram()` per call is what makes the sequence a restart test:
 * nothing a command computed survives into the next one, so every command has to
 * rebuild the session from the log on disk.
 */
async function runCli(argv: readonly string[]): Promise<number> {
  const program = createProgram();
  for (const command of [program, ...program.commands]) {
    command.exitOverride();
    command.configureOutput({ writeErr: () => {} });
  }
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
    let savedCwd: string;
    let exitCodes: number[];
    let records: SessionLogRecord[];
    let statusReport: string;
    let branch: string;
    let keptCommit: string;
    let mainStatus: string;
    let history: string;

    beforeAll(async () => {
      savedCwd = process.cwd();
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
      process.chdir(savedCwd);
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
      expect(second?.samples).toEqual({
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
    let savedCwd: string;
    let gateDir: string;
    let gateFile: string;

    beforeEach(async () => {
      savedCwd = process.cwd();
      gateDir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-gate-"));
      gateFile = path.join(gateDir, "release");
      repo = createScratchRepo();
      commitProject(repo, gateFile);
      process.chdir(repo.dir);

      mockProcessExit();
      captureOutput();
      await runCli(["start", "main"]);
      tuneExperiment(repo, KEPT_LATENCY);
    }, LONG_RUN_TIMEOUT_MS);

    afterEach(() => {
      vi.restoreAllMocks();
      process.chdir(savedCwd);
      repo.cleanup();
      fs.rmSync(gateDir, { recursive: true, force: true });
    });

    it(
      "refuses the second run with exit code 2 and lets the first one finish",
      async () => {
        // Arrange - the gated bench holds the first run open until the test releases it.
        const lockPath = lockfilePath(repo.dir);
        const first = runCli(["iterate"]);
        await vi.waitFor(
          () => {
            expect(fs.existsSync(lockPath)).toBe(true);
          },
          { timeout: 10_000, interval: 25 },
        );

        // Act
        const second = await runCli(["iterate"]);

        // Assert
        fs.writeFileSync(gateFile, "");
        expect.soft(second).toBe(2);
        expect.soft(await first).toBe(0);
        expect.soft(fs.existsSync(lockPath)).toBe(false);
        expect(pick(readRecords(sessionJsonlPath(repo.dir)), "iteration")).toHaveLength(1);
      },
      LONG_RUN_TIMEOUT_MS,
    );
  });
});
