import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterAll, afterEach, beforeAll, beforeEach, vi } from "vitest";

import { AdapterError } from "../src/adapters/index.js";
import { compare } from "../src/compare.js";
import type { CompareOptions, ProgressStep } from "../src/compare.js";
import { GymratError } from "../src/errors.js";
import { renderReport } from "../src/report/text.js";
import { CommandError } from "../src/sampling.js";
import { REF_TARGET_HINT } from "./fixtures/constants.js";
import { captureRejection } from "./fixtures/errors.js";
import { isAlive, waitForPid } from "./fixtures/process-probe.js";
import {
  createInPlaceTargetDir,
  createScratchRepo,
  killGitDuringWorktreeAdd,
  listWorktreeDirs,
  removeStrandedWorktrees,
  toShellPath,
} from "./fixtures/scratch-repo.js";
import type { ScratchRepo } from "./fixtures/scratch-repo.js";
import {
  TERMINATION_SIGNALS,
  raiseSignal,
  removeLeakedListeners,
  signalListenerCounts,
  snapshotSignalListeners,
  stubProcessExit,
  type SignalName,
} from "./fixtures/signal-probe.js";

function assertCommandError(error: Error): asserts error is CommandError {
  expect(error).toBeInstanceOf(CommandError);
}

function assertGymratError(error: Error): asserts error is GymratError {
  expect(error).toBeInstanceOf(GymratError);
}

interface BranchSetup {
  name: string;
  benchScript: string;
  prepareScript?: string;
}

function createBranch(
  repo: ReturnType<typeof createScratchRepo>,
  setup: BranchSetup,
  baseRef = "main",
) {
  execFileSync("git", ["checkout", "-b", setup.name, baseRef], {
    cwd: repo.dir,
    stdio: "pipe",
  });

  fs.writeFileSync(path.join(repo.dir, "bench.sh"), setup.benchScript);
  const scripts = ["bench.sh"];

  if (setup.prepareScript) {
    fs.writeFileSync(path.join(repo.dir, "prepare.sh"), setup.prepareScript);
    scripts.push("prepare.sh");
  }

  execFileSync("git", ["add", ...scripts], {
    cwd: repo.dir,
    stdio: "pipe",
  });

  execFileSync("git", ["commit", "-m", setup.name], {
    cwd: repo.dir,
    stdio: "pipe",
  });
}

/**
 * Create a scratch repo, change into it for the duration of `fn`, then sweep
 * any stranded worktrees and clean the repo up — even if `fn` throws.
 *
 * Centralizes the create-repo/chdir/try/cleanup scaffolding shared by every
 * self-contained integration test below.
 */
async function withScratchRepo<T>(fn: (repo: ScratchRepo) => Promise<T>): Promise<T> {
  const repo = createScratchRepo();
  try {
    process.chdir(repo.dir);
    return await fn(repo);
  } finally {
    removeStrandedWorktrees(repo);
    repo.cleanup();
  }
}

/**
 * The CompareOptions fields shared by every integration test call site, with
 * only `baseline` and `candidates` required — everything else follows the
 * standard bench/adapter/samples/timeout defaults unless overridden.
 */
function compareOptions(
  overrides: Partial<CompareOptions> & Pick<CompareOptions, "baseline" | "candidates">,
): CompareOptions {
  return {
    bench: "sh bench.sh",
    adapter: "metric-lines",
    samples: 3,
    timeoutSeconds: 10,
    ...overrides,
  };
}

/**
 * The single report line matching `predicate`, or a failure naming the whole report.
 */
function findLine(report: string, predicate: (line: string) => boolean): string {
  const line = report.split("\n").find(predicate);
  if (line === undefined) {
    throw new Error(`no matching line in report:\n${report}`);
  }
  return line;
}

/**
 * Extract the directories named by `left behind: <dir> <reason>` entries.
 *
 * Requires a non-empty reason on the same line, so a multi-line git diagnostic
 * that leaked into the text would not match.
 */
function parseLeftBehindDirs(text: string): string[] {
  return text.split("\n").flatMap((line) => /^ {2}left behind: (\S+) \S/.exec(line)?.[1] ?? []);
}

/** Assert the repo lists no worktree beyond its own main directory. */
function assertWorktreesCleanedUp(repo: ReturnType<typeof createScratchRepo>) {
  expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
}

/**
 * The gymrat worktree directories currently sitting in the system temp dir.
 *
 * Read straight off the filesystem rather than from `git worktree list`: a
 * directory git has already deregistered — pruned, or removed from under it —
 * is exactly the one that leaks, and the registry can no longer see it.
 */
function listTempWorktreeDirs(): string[] {
  return fs
    .readdirSync(os.tmpdir())
    .filter((entry) => entry.startsWith("gymrat-wt-"))
    .toSorted();
}

interface ErrorPathCase {
  oldBranch: BranchSetup;
  newBranch: BranchSetup;
  options?: Partial<Omit<CompareOptions, "baseline" | "candidates">>;
}

/**
 * Run a compare() setup expected to reject with a CommandError, once for the whole
 * describe block, and hand back accessors for the repo and the captured error.
 *
 * Centralizes the create-repo/chdir/create-branches/capture-rejection/cleanup
 * scaffolding shared by every error-path describe block below.
 */
function useErrorPathCase(setup: ErrorPathCase): {
  repo: () => ReturnType<typeof createScratchRepo>;
  error: () => CommandError;
} {
  let repo: ReturnType<typeof createScratchRepo>;
  let error: CommandError;

  beforeAll(async () => {
    repo = createScratchRepo();
    process.chdir(repo.dir);

    createBranch(repo, setup.oldBranch);
    createBranch(repo, setup.newBranch);

    const options = compareOptions({
      baseline: { target: setup.oldBranch.name },
      candidates: [{ target: setup.newBranch.name }],
      ...setup.options,
    });

    const failure = await captureRejection(compare(options));
    assertCommandError(failure);
    error = failure;
  });

  afterAll(() => {
    repo.cleanup();
  });

  return {
    repo: () => repo,
    error: () => error,
  };
}

/** This pid appears only once a whole comparison run is under way, so it waits far longer. */
const PID_WAIT_MS = 10_000;

interface InFlightRun {
  /** Settles when compare() gives up; never rejects, so the test picks when to wait. */
  settled: Promise<void>;
  /** A process the bench spawned outside gymrat's own child, still running. */
  benchGrandchildPid: number;
  /** The worktrees git listed once the bench was running. */
  worktreeDirs: string[];
}

/**
 * Start a compare() run whose old-target bench blocks until something kills it.
 *
 * Resolves once that bench is up and has spawned a grandchild of its own, so a
 * signal raised afterwards lands on a run with real work in flight.
 */
async function startInFlightRun(repo: ReturnType<typeof createScratchRepo>): Promise<InFlightRun> {
  // Written into the main repo dir so it outlives the worktree the bench runs in.
  const pidFile = path.join(repo.dir, "bench-grandchild.pid");

  createBranch(repo, {
    name: "old-signal",
    benchScript: `#!/bin/sh\nsleep 30 &\necho $! > "${toShellPath(pidFile)}"\nwait`,
  });
  createBranch(repo, {
    name: "new-signal",
    benchScript: '#!/bin/sh\necho "METRIC latency=90"',
  });

  const options = compareOptions({
    baseline: { target: "old-signal" },
    candidates: [{ target: "new-signal" }],
    timeoutSeconds: 20,
  });

  // Swallowed: once a signal has torn the run down, whether compare() reports the
  // killed bench or its vanished worktree is beside the point being tested.
  const settled = compare(options).then(
    () => undefined,
    () => undefined,
  );
  const benchGrandchildPid = await waitForPid(pidFile, PID_WAIT_MS);

  return {
    settled,
    benchGrandchildPid,
    worktreeDirs: listWorktreeDirs(repo.dir, { includeMain: false }),
  };
}

/**
 * Kill any grandchild the run's bench spawned, wait for compare() to settle,
 * then clean up worktrees and the repo.
 *
 * Safety net: a run interrupted before the signal handler killed it must not
 * leak a sleeping process into the rest of the suite.
 */
async function cleanupInFlightRun(
  repo: ReturnType<typeof createScratchRepo>,
  run: InFlightRun | undefined,
): Promise<void> {
  if (run) {
    try {
      process.kill(run.benchGrandchildPid, "SIGKILL");
    } catch {
      // Already gone — the handler got it.
    }
    await run.settled;
  }
  removeStrandedWorktrees(repo);
  repo.cleanup();
}

/** Generous timeout for tests whose run creates worktrees and spawns real bench processes. */
const LONG_RUN_TIMEOUT_MS = 60_000;

const PREEXISTING_SIGNAL_LISTENERS = snapshotSignalListeners();

describe("compare – integration", () => {
  describe("when bench command fails", () => {
    const failingBench = '#!/bin/sh\necho "stderr output" >&2\nexit 1';
    const passingBench = '#!/bin/sh\necho "METRIC latency=100"';

    describe.each([
      { side: "old" as const, oldBench: failingBench, newBench: passingBench },
      { side: "new" as const, oldBench: passingBench, newBench: failingBench },
    ])("from the $side target", ({ side, oldBench, newBench }) => {
      const { repo, error } = useErrorPathCase({
        oldBranch: { name: `old-fail-${side}`, benchScript: oldBench },
        newBranch: { name: `new-fail-${side}`, benchScript: newBench },
      });

      it("throws a CommandError with structured context", () => {
        expect.soft(error().message).toContain("bench");
        expect.soft(error().message).toContain(side);
        expect.soft(error().message).toContain(`${side}-fail-${side}`);
        expect.soft(error().message).toContain("sh bench.sh");
        expect.soft(error().message).toContain("exit code: 1");
        expect.soft(error().message).toContain("sample 1");

        expect(error().message).toContain("stderr output");
        expect(error().message).toMatch(/worktree:\s+\S/);
        expect(error().message).not.toContain("hint:");
        expect(error().hint).toBe(REF_TARGET_HINT);
        expect(error().message).not.toContain("left behind");
      });

      it("cleans up all worktrees", () => {
        assertWorktreesCleanedUp(repo());
      });
    });
  });

  describe("when the bench command exceeds the timeout", () => {
    const { repo, error } = useErrorPathCase({
      oldBranch: { name: "old-slow", benchScript: "#!/bin/sh\nsleep 10" },
      newBranch: { name: "new-slow", benchScript: '#!/bin/sh\necho "METRIC latency=90"' },
      // Fractional seconds keep the test near half a second; the sleep is
      // killed with its process group, so nothing outlives the run.
      options: { timeoutSeconds: 0.5 },
    });

    it("throws a CommandError with timeout context", () => {
      expect.soft(error().message).toContain("bench");
      expect.soft(error().message).toContain("old");
      expect.soft(error().message).toContain("old-slow");
      expect.soft(error().message).toContain("sh bench.sh");
      expect.soft(error().message).toContain("500ms");
      expect.soft(error().message).not.toContain("exit code");
      expect(error().message).toMatch(/timed out/);
      expect(error().message).not.toContain("left behind");
    });

    it("cleans up all worktrees", () => {
      assertWorktreesCleanedUp(repo());
    });
  });

  describe("when prepare command fails", () => {
    const { repo, error } = useErrorPathCase({
      oldBranch: {
        name: "old-prep-fail",
        benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        prepareScript: '#!/bin/sh\necho "prep failed" >&2\nexit 1',
      },
      newBranch: {
        name: "new-prep-fail",
        benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        prepareScript: "#!/bin/sh\nexit 0",
      },
      options: { prepare: "sh prepare.sh" },
    });

    it("throws a CommandError with prepare context", () => {
      expect.soft(error().message).toContain("prepare");
      expect.soft(error().message).toContain("old");
      expect.soft(error().message).toContain("old-prep-fail");
      expect.soft(error().message).toContain("sh prepare.sh");
      expect.soft(error().message).toContain("exit code: 1");
      expect.soft(error().message).not.toContain("sample ");

      expect(error().message).toContain("prep failed");
      expect(error().message).toMatch(/worktree:\s+\S/);
      expect(error().message).not.toContain("hint:");
      expect(error().hint).toBe(REF_TARGET_HINT);
      expect(error().message).not.toContain("left behind");
    });

    it("cleans up all worktrees", () => {
      assertWorktreesCleanedUp(repo());
    });
  });

  describe("when prepare command times out", () => {
    const { repo, error } = useErrorPathCase({
      oldBranch: {
        name: "old-prep-slow",
        benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        prepareScript: "#!/bin/sh\nsleep 10",
      },
      newBranch: {
        name: "new-prep-slow",
        benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        prepareScript: "#!/bin/sh\nexit 0",
      },
      options: { prepare: "sh prepare.sh", timeoutSeconds: 0.5 },
    });

    it("throws a CommandError with prepare timeout context", () => {
      expect.soft(error().message).toContain("prepare");
      expect.soft(error().message).toContain("old");
      expect.soft(error().message).toContain("old-prep-slow");
      expect.soft(error().message).toContain("sh prepare.sh");
      expect.soft(error().message).toContain("500ms");
      expect.soft(error().message).not.toContain("exit code");
      expect(error().message).toMatch(/timed out/);
      expect(error().message).not.toContain("left behind");
    });

    it("cleans up all worktrees", () => {
      assertWorktreesCleanedUp(repo());
    });
  });

  describe("when bench command fails on in-place targets", () => {
    it("throws a CommandError with directory context and no ref hint", async () => {
      await withScratchRepo(async (repo) => {
        for (const [name, script] of [
          ["old-dir-fail", '#!/bin/sh\necho "dir bench failed" >&2\nexit 1'],
          ["new-dir-fail", '#!/bin/sh\necho "METRIC latency=90"'],
        ] as const) {
          createInPlaceTargetDir(repo, name, script);
        }

        const failure = await captureRejection(
          compare(
            compareOptions({
              baseline: { target: "old-dir-fail" },
              candidates: [{ target: "new-dir-fail" }],
            }),
          ),
        );

        assertCommandError(failure);

        expect.soft(failure.message).toContain("bench");
        expect.soft(failure.message).toContain("old");
        expect.soft(failure.message).toContain("old-dir-fail");
        expect.soft(failure.message).toContain("sh bench.sh");
        expect.soft(failure.message).toContain("exit code: 1");
        expect.soft(failure.message).toContain("sample 1");

        expect(failure.message).toContain("dir bench failed");
        expect(failure.message).toMatch(/dir:\s+\S/);
        expect(failure.hint).toBeUndefined();
        expect(failure.message).not.toContain("worktree:");
      });
    });
  });

  describe("when git cannot remove a worktree the run created", () => {
    it("reports the cleanup outcome the run actually produced", async () => {
      await withScratchRepo(async (repo) => {
        // The directory git refuses to remove outlives the suite unless the
        // test takes it down itself, as both sibling cases do.
        const leftBehindDirs: string[] = [];

        try {
          // Deleting the worktree's .git link makes `git worktree remove` refuse the
          // directory at any force level, so one worktree survives cleanup.
          createBranch(repo, {
            name: "old-unremovable",
            benchScript: '#!/bin/sh\nrm -f .git\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: "new-unremovable",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });

          const report = renderReport(
            await compare(
              compareOptions({
                baseline: { target: "old-unremovable" },
                candidates: [{ target: "new-unremovable" }],
              }),
            ),
          );

          leftBehindDirs.push(...parseLeftBehindDirs(report));
          expect(report).toContain("1 worktree removed · 1 left behind");
          expect(leftBehindDirs).toHaveLength(1);
          expect(leftBehindDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual(leftBehindDirs);
        } finally {
          for (const dir of leftBehindDirs) {
            fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
          }
        }
      });
    });
  });

  // killGitDuringWorktreeAdd sends POSIX signal 9 via a post-checkout hook;
  // Windows cannot deliver that signal to the git parent process.
  describe.skipIf(process.platform === "win32")(
    "when git worktree add is killed after creating the worktree",
    () => {
      it("sweeps the directory it left behind instead of stranding it", async () => {
        const strandedBefore = listTempWorktreeDirs();

        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: "old-interrupted",
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: "new-interrupted",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });

          // Installed last so the branch checkouts above survive it.
          killGitDuringWorktreeAdd(repo.dir);

          const failure = await captureRejection(
            compare(
              compareOptions({
                baseline: { target: "old-interrupted" },
                candidates: [{ target: "new-interrupted" }],
              }),
            ),
          );

          // The run fails because `git worktree add` did, not because cleanup left
          // anything behind — `withCleanupFailures` appends "left behind" only when
          // cleanup actually stranded something.
          expect(failure.message).toContain("worktree add");
          expect(failure.message).not.toContain("left behind");
          // The registry going empty says git removed its entry; only the temp dir
          // itself says the directory is gone, and that is the leak under test.
          expect(listTempWorktreeDirs()).toStrictEqual(strandedBefore);
          assertWorktreesCleanedUp(repo);
        });
      });
    },
  );

  describe("when the bench fails and git cannot remove a worktree", () => {
    it("throws an error naming the stranded worktree alongside the original failure", async () => {
      await withScratchRepo(async (repo) => {
        // Hoisted so `finally` can delete these: the run's own cleanup prunes the
        // registry entry before this test regains control, so `withScratchRepo`'s
        // stranded-worktree sweep enumerates nothing and the directory would
        // outlive the suite.
        const leftBehindDirs: string[] = [];

        try {
          // Deleting the worktree's .git link makes `git worktree remove` refuse the
          // directory, and the non-zero exit fails the run before cleanup happens.
          createBranch(repo, {
            name: "old-fail-unremovable",
            benchScript: '#!/bin/sh\nrm -f .git\necho "stderr output" >&2\nexit 1',
          });

          createBranch(repo, {
            name: "new-fail-unremovable",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });

          const failure = await captureRejection(
            compare(
              compareOptions({
                baseline: { target: "old-fail-unremovable" },
                candidates: [{ target: "new-fail-unremovable" }],
              }),
            ),
          );

          leftBehindDirs.push(...parseLeftBehindDirs(failure.message));
          expect(failure.message).toMatch(/stderr output/);
          expect(leftBehindDirs).toHaveLength(1);
          expect(leftBehindDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual(leftBehindDirs);
          expect(failure.cause).toBeInstanceOf(Error);
          // The hint on the original CommandError must survive the re-wrap so
          // formatCliError can still surface it.
          assertGymratError(failure);
          expect(failure.hint).toBe(REF_TARGET_HINT);
        } finally {
          for (const dir of leftBehindDirs) {
            fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
          }
        }
      });
    });
  });

  describe("when the adapter throws and cleanup leaves a worktree behind", () => {
    it("re-wraps the AdapterError with the cleanup failure details", async () => {
      await withScratchRepo(async (repo) => {
        const leftBehindDirs: string[] = [];

        try {
          createBranch(repo, {
            name: "old-adapter-fail",
            benchScript: '#!/bin/sh\nrm -f .git\necho "no metrics"',
          });

          createBranch(repo, {
            name: "new-adapter-fail",
            benchScript: '#!/bin/sh\necho "no metrics either"',
          });

          const failure = await captureRejection(
            compare(
              compareOptions({
                baseline: { target: "old-adapter-fail" },
                candidates: [{ target: "new-adapter-fail" }],
              }),
            ),
          );

          leftBehindDirs.push(...parseLeftBehindDirs(failure.message));
          expect(failure).toBeInstanceOf(AdapterError);
          expect(failure.message).toContain("METRIC");
          expect(failure.message).toContain("cleanup did not finish");
          expect(failure.cause).toBeInstanceOf(AdapterError);
        } finally {
          for (const dir of leftBehindDirs) {
            fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
          }
        }
      });
    });
  });

  describe("when metric values are zero", () => {
    it("renders a zero median without a spread and a 0.0% delta rather than NaN", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-zero",
          benchScript: '#!/bin/sh\necho "METRIC latency=0"',
        });

        createBranch(repo, {
          name: "new-zero",
          benchScript: '#!/bin/sh\necho "METRIC latency=0"',
        });

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-zero" },
              candidates: [{ target: "new-zero" }],
            }),
          ),
        );

        const latencyRow = findLine(report, (line) => line.startsWith("latency"));
        expect(latencyRow).toMatch(/^latency\s*│\s*0\s+│\s*0\s+│\s*~\s+0\.0%/);
      });
    });
  });

  describe("when the run took a single sample", () => {
    it("prints each median alone, claiming no spread from one observation", async () => {
      await withScratchRepo(async (repo) => {
        // In-place targets: a single sample per side is the whole run, so no
        // worktree is created and nothing has to be cleaned up afterwards.
        for (const [name, latency] of [
          ["old-single", 100],
          ["new-single", 90],
        ] as const) {
          createInPlaceTargetDir(repo, name, `#!/bin/sh\necho "METRIC latency=${latency}"`);
        }

        const result = await compare(
          compareOptions({
            baseline: { target: "old-single" },
            candidates: [{ target: "new-single" }],
            samples: 1,
          }),
        );
        const cells = findLine(renderReport(result), (line) => line.startsWith("latency")).split(
          "│",
        );

        expect.soft(result.metrics["latency"]?.baselineSpread).toBeUndefined();
        expect.soft(cells[1]?.trim()).toBe("100");
        expect(cells[2]?.trim()).toBe("90");
      });
    });
  });

  describe("when bench script produces no metrics", () => {
    it("raises an AdapterError so the CLI blames the bench script", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-empty",
          benchScript: '#!/bin/sh\necho "no metrics here"',
        });

        createBranch(repo, {
          name: "new-empty",
          benchScript: '#!/bin/sh\necho "also no metrics"',
        });

        const error = await compare(
          compareOptions({
            baseline: { target: "old-empty" },
            candidates: [{ target: "new-empty" }],
          }),
        ).catch((e: unknown) => e);
        expect.soft(error).toBeInstanceOf(AdapterError);
        expect(error).toHaveProperty("message", expect.stringMatching(/[Nn]o valid METRIC/));
      });
    });
  });

  // The bench spawns a background process via `sleep 30 &` and writes its PID;
  // Git for Windows' sh reports PIDs from its own process namespace that Node cannot signal.
  describe.skipIf(process.platform === "win32")("when a termination signal arrives mid-run", () => {
    let listenersBefore: Record<SignalName, readonly unknown[]>;

    beforeEach(() => {
      listenersBefore = PREEXISTING_SIGNAL_LISTENERS;
      stubProcessExit();
    });

    afterEach(() => {
      vi.restoreAllMocks();

      // Uninstall swaps cleanup handlers for exit-only handlers that catch
      // signals queued during a blocking call — those are expected to survive
      // settlement.  Remove them here for test-suite hygiene.
      for (const signal of TERMINATION_SIGNALS) {
        removeLeakedListeners(signal, listenersBefore[signal]);
      }
    });

    it.each<{ signal: SignalName; expectedCode: number }>([
      { signal: "SIGINT", expectedCode: 130 },
      { signal: "SIGTERM", expectedCode: 143 },
      { signal: "SIGHUP", expectedCode: 129 },
    ])(
      "exits with $expectedCode on $signal",
      async ({ signal, expectedCode }) => {
        const repo = createScratchRepo();
        let run: InFlightRun | undefined;

        try {
          process.chdir(repo.dir);
          run = await startInFlightRun(repo);

          const exitCode = raiseSignal(signal, listenersBefore[signal]);

          expect(exitCode).toBe(expectedCode);
        } finally {
          await cleanupInFlightRun(repo, run);
        }
      },
      LONG_RUN_TIMEOUT_MS,
    );

    describe("once SIGINT interrupts an in-flight run", () => {
      let repo: ReturnType<typeof createScratchRepo>;
      let run: InFlightRun;

      beforeAll(async () => {
        stubProcessExit();

        repo = createScratchRepo();
        process.chdir(repo.dir);
        run = await startInFlightRun(repo);
        // Without this guard, the post-signal assertions below would also pass for
        // a pid that was never alive — `waitForPid` only proves a number reached
        // the file. `expect()` isn't usable in `beforeAll`, so this throws instead.
        if (!isAlive(run.benchGrandchildPid)) {
          throw new Error(`bench grandchild pid ${run.benchGrandchildPid} is not alive`);
        }

        raiseSignal("SIGINT", PREEXISTING_SIGNAL_LISTENERS.SIGINT);
      }, LONG_RUN_TIMEOUT_MS);

      afterAll(async () => {
        vi.restoreAllMocks();
        await cleanupInFlightRun(repo, run);
      });

      it("removes the worktrees it created", () => {
        expect(run.worktreeDirs.length).toBeGreaterThan(0);
        expect(run.worktreeDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual([]);
        assertWorktreesCleanedUp(repo);
      });

      it("kills the bench process group, leaving nothing running in the removed worktree", async () => {
        const grandchildPid = run.benchGrandchildPid;
        await vi.waitFor(
          () => {
            expect(isAlive(grandchildPid)).toBe(false);
          },
          { timeout: 5000, interval: 25 },
        );
      });
    });
  });

  describe("when a run finishes on its own", () => {
    it.each<{ outcome: "resolved" | "rejected"; benchScript: string }>([
      { outcome: "resolved", benchScript: '#!/bin/sh\necho "METRIC latency=100"' },
      { outcome: "rejected", benchScript: "#!/bin/sh\nexit 1" },
    ])(
      "leaves the signal listener count where it found it once the run has $outcome",
      async ({ outcome, benchScript }) => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, { name: "old-settled", benchScript });
          createBranch(repo, {
            name: "new-settled",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });

          const runOnce = async (): Promise<string> =>
            compare(
              compareOptions({
                baseline: { target: "old-settled" },
                candidates: [{ target: "new-settled" }],
                timeoutSeconds: 20,
              }),
            ).then(
              () => "resolved",
              () => "rejected",
            );

          const settled = await runOnce();
          const afterFirst = signalListenerCounts();
          await runOnce();

          // What "no leak" means: the handler is attached once and reused, so a
          // second run adds nothing. Counting the first run's own install would
          // pin the arrangement instead — and a per-run install is exactly the
          // shape that accumulated listeners until it tripped MaxListeners.
          expect(settled).toBe(outcome);
          expect(signalListenerCounts()).toStrictEqual(afterFirst);
        });
      },
      LONG_RUN_TIMEOUT_MS,
    );
  });

  describe("progress callback", () => {
    /** A steps sink and the `onProgress` callback that appends to it, in call order. */
    function collectSteps(): {
      steps: ProgressStep[];
      onProgress: (step: ProgressStep) => void;
    } {
      const steps: ProgressStep[] = [];
      return { steps, onProgress: (step) => steps.push(step) };
    }

    it("receives sample steps in round-robin order when no prepare is configured", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const { steps, onProgress } = collectSteps();

        await compare(
          compareOptions({
            baseline: { target: "old-progress" },
            candidates: [{ target: "new-progress" }],
            onProgress,
          }),
        );

        expect(steps).toStrictEqual([
          { kind: "sample", index: 1, total: 3, label: "old-progress" },
          { kind: "sample", index: 1, total: 3, label: "new-progress" },
          { kind: "sample", index: 2, total: 3, label: "old-progress" },
          { kind: "sample", index: 2, total: 3, label: "new-progress" },
          { kind: "sample", index: 3, total: 3, label: "old-progress" },
          { kind: "sample", index: 3, total: 3, label: "new-progress" },
        ]);
      });
    });

    it("receives prepare steps before sample steps when prepare is configured", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-prep-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          prepareScript: "#!/bin/sh\nexit 0",
        });

        createBranch(repo, {
          name: "new-prep-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          prepareScript: "#!/bin/sh\nexit 0",
        });

        const { steps, onProgress } = collectSteps();

        await compare(
          compareOptions({
            baseline: { target: "old-prep-progress" },
            candidates: [{ target: "new-prep-progress" }],
            prepare: "sh prepare.sh",
            samples: 2,
            onProgress,
          }),
        );

        expect(steps).toStrictEqual([
          { kind: "prepare", label: "old-prep-progress" },
          { kind: "prepare", label: "new-prep-progress" },
          { kind: "sample", index: 1, total: 2, label: "old-prep-progress" },
          { kind: "sample", index: 1, total: 2, label: "new-prep-progress" },
          { kind: "sample", index: 2, total: 2, label: "old-prep-progress" },
          { kind: "sample", index: 2, total: 2, label: "new-prep-progress" },
        ]);
      });
    });

    it("uses explicit labels in progress steps when supplied", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-labelled-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-labelled-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const { steps, onProgress } = collectSteps();

        await compare(
          compareOptions({
            baseline: { target: "old-labelled-progress", label: "baseline" },
            candidates: [{ target: "new-labelled-progress", label: "candidate" }],
            samples: 2,
            onProgress,
          }),
        );

        const labels = steps.map((s) => s.label);
        expect(labels).toStrictEqual(["baseline", "candidate", "baseline", "candidate"]);
      });
    });

    it("emits sample steps for three targets in round-robin order", async () => {
      await withScratchRepo(async (repo) => {
        for (const [name, latency] of [
          ["base-progress-3", 100],
          ["cand-a-progress", 80],
          ["cand-b-progress", 120],
        ] as const) {
          createBranch(repo, {
            name,
            benchScript: `#!/bin/sh\necho "METRIC latency=${latency}"`,
          });
        }

        const { steps, onProgress } = collectSteps();

        await compare(
          compareOptions({
            baseline: { target: "base-progress-3" },
            candidates: [{ target: "cand-a-progress" }, { target: "cand-b-progress" }],
            samples: 2,
            onProgress,
          }),
        );

        expect(steps).toStrictEqual([
          { kind: "sample", index: 1, total: 2, label: "base-progress-3" },
          { kind: "sample", index: 1, total: 2, label: "cand-a-progress" },
          { kind: "sample", index: 1, total: 2, label: "cand-b-progress" },
          { kind: "sample", index: 2, total: 2, label: "base-progress-3" },
          { kind: "sample", index: 2, total: 2, label: "cand-a-progress" },
          { kind: "sample", index: 2, total: 2, label: "cand-b-progress" },
        ]);
      });
    });

    it("completes a run when onProgress is omitted", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-no-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-no-progress",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const result = await compare(
          compareOptions({
            baseline: { target: "old-no-progress" },
            candidates: [{ target: "new-no-progress" }],
            samples: 2,
          }),
        );
        expect(result.samples).toBe(2);
      });
    });
  });
});
