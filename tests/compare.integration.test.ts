import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterAll, afterEach, beforeAll, beforeEach, vi } from "vitest";

import { AdapterError } from "../src/adapters/index.js";
import { compare, CommandError } from "../src/compare.js";
import type { CompareOptions, ProgressStep } from "../src/compare.js";
import { GymratError } from "../src/errors.js";
import { renderReport } from "../src/report/text.js";
import type { ComparisonResult } from "../src/report/types.js";
import { REF_TARGET_HINT } from "./fixtures/constants.js";
import { isAlive, waitForPid } from "./fixtures/process-probe.js";
import {
  createScratchRepo,
  killGitDuringWorktreeAdd,
  listWorktreeDirs,
} from "./fixtures/scratch-repo.js";
import type { ScratchRepo } from "./fixtures/scratch-repo.js";

/** Convert a path to forward slashes so it can be embedded in a shell script on any platform. */
function toShellPath(p: string): string {
  return p.replace(/\\/g, "/");
}

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
  execSync(`git checkout -b ${setup.name} ${baseRef}`, {
    cwd: repo.dir,
    stdio: "pipe",
  });

  fs.writeFileSync(path.join(repo.dir, "bench.sh"), setup.benchScript);
  const scripts = ["bench.sh"];

  if (setup.prepareScript) {
    fs.writeFileSync(path.join(repo.dir, "prepare.sh"), setup.prepareScript);
    scripts.push("prepare.sh");
  }

  execSync(`git add ${scripts.join(" ")}`, {
    cwd: repo.dir,
    stdio: "pipe",
  });

  execSync(`git commit -m '${setup.name}'`, {
    cwd: repo.dir,
    stdio: "pipe",
  });
}

/**
 * Create an untracked subdirectory of the repo with its own executable `bench.sh`.
 *
 * `resolveTarget` sees a plain directory and returns an in-place target, so
 * benching it never creates a worktree.
 */
function createInPlaceTargetDir(
  repo: ReturnType<typeof createScratchRepo>,
  name: string,
  benchScript: string,
) {
  fs.mkdirSync(path.join(repo.dir, name));
  const script = path.join(repo.dir, name, "bench.sh");
  fs.writeFileSync(script, benchScript);
}

/**
 * Create a scratch repo, chdir into it for the duration of `fn`, then sweep
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
 * Await a promise expected to reject and hand back the Error it rejected with.
 */
async function captureRejection(promise: Promise<unknown>): Promise<Error> {
  const outcome: unknown = await promise.then(
    () => undefined,
    (error: unknown) => error,
  );
  if (!(outcome instanceof Error)) {
    throw new Error(`expected a rejection with an Error, got: ${String(outcome)}`);
  }
  return outcome;
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

/**
 * Delete every worktree directory the repo still lists, main repo dir excluded.
 *
 * Keeps a run that stranded a worktree from leaking it into the system temp dir,
 * whatever the assertions did or did not manage to read.
 */
function removeStrandedWorktrees(repo: ReturnType<typeof createScratchRepo>) {
  for (const dir of listWorktreeDirs(repo.dir, { includeMain: false })) {
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  }
}

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

const TERMINATION_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

type SignalName = (typeof TERMINATION_SIGNALS)[number];

/** Thrown by the stubbed `process.exit` so a handler unwinds where it really would. */
class ProcessExited extends Error {
  constructor(readonly code: number | string | null | undefined) {
    super(`process.exit(${String(code)})`);
    this.name = "ProcessExited";
  }
}

function isSignalListener(value: unknown): value is (signal: SignalName) => void {
  return typeof value === "function";
}

function signalListenerCounts(): Record<SignalName, number> {
  return {
    SIGINT: process.listeners("SIGINT").length,
    SIGTERM: process.listeners("SIGTERM").length,
    SIGHUP: process.listeners("SIGHUP").length,
  };
}

/**
 * Remove every listener on `signal` that was not already present `before`.
 *
 * Reports whether it found (and removed) one, so a caller can name every
 * signal that still had a handler installed after a run had settled.
 */
function removeLeakedListeners(signal: SignalName, before: readonly unknown[]): boolean {
  let leaked = false;
  for (const listener of process.listeners(signal)) {
    if (!before.includes(listener) && isSignalListener(listener)) {
      process.removeListener(signal, listener);
      leaked = true;
    }
  }
  return leaked;
}

/**
 * Run the handlers compare() installed for `signal` and report the code it exits with.
 *
 * Emitting the signal for real would also trip vitest's own handling and tear the
 * test run down, so only the listeners added since `before` are invoked. With
 * `process.exit` stubbed to throw, a handler unwinds exactly where the real one
 * would stop.
 */
function raiseSignal(
  signal: SignalName,
  before: readonly unknown[],
): number | string | null | undefined {
  const installed = process.listeners(signal).filter((listener) => !before.includes(listener));
  if (installed.length === 0) {
    throw new Error(`compare() installed no ${signal} handler`);
  }

  try {
    for (const listener of installed) {
      if (isSignalListener(listener)) {
        listener(signal);
      }
    }
  } catch (error) {
    if (error instanceof ProcessExited) {
      return error.code;
    }
    throw error;
  }
  throw new Error(`the ${signal} handler returned instead of exiting`);
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
 * Centralizes the save-cwd/create-repo/chdir/create-branches/capture-rejection/
 * restore-cwd/cleanup scaffolding shared by every error-path describe block below.
 */
function useErrorPathCase(setup: ErrorPathCase): {
  repo: () => ReturnType<typeof createScratchRepo>;
  error: () => CommandError;
} {
  let repo: ReturnType<typeof createScratchRepo>;
  let error: CommandError;
  let savedCwd: string;

  beforeAll(async () => {
    savedCwd = process.cwd();
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
    process.chdir(savedCwd);
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

describe("compare – integration", () => {
  let originalCwd: string;

  beforeEach(() => {
    originalCwd = process.cwd();
  });

  afterEach(() => {
    process.chdir(originalCwd);
  });

  describe("when prepare command is provided", () => {
    it("runs prepare once before every bench run on both targets", async () => {
      await withScratchRepo(async (repo) => {
        // The scripts run inside throwaway worktrees, so they append to a log in
        // the main repo dir that outlives cleanup.
        const oldLog = path.join(repo.dir, "old-log.txt");
        const newLog = path.join(repo.dir, "new-log.txt");

        createBranch(repo, {
          name: "old-prep",
          benchScript: `#!/bin/sh\necho bench >> "${toShellPath(oldLog)}"\necho "METRIC latency=100"`,
          prepareScript: `#!/bin/sh\necho prepare >> "${toShellPath(oldLog)}"`,
        });

        createBranch(repo, {
          name: "new-prep",
          benchScript: `#!/bin/sh\necho bench >> "${toShellPath(newLog)}"\necho "METRIC latency=90"`,
          prepareScript: `#!/bin/sh\necho prepare >> "${toShellPath(newLog)}"`,
        });

        await compare(
          compareOptions({
            baseline: { target: "old-prep" },
            candidates: [{ target: "new-prep" }],
            prepare: "sh prepare.sh",
          }),
        );

        const readLog = (file: string) => fs.readFileSync(file, "utf-8").trim().split("\n");
        expect(readLog(oldLog)).toStrictEqual(["prepare", "bench", "bench", "bench"]);
        expect(readLog(newLog)).toStrictEqual(["prepare", "bench", "bench", "bench"]);
      });
    });
  });

  describe("when one baseline is compared against two candidates", () => {
    let repo: ReturnType<typeof createScratchRepo>;
    let savedCwd: string;
    let benchOrder: string[];
    let result: ComparisonResult;

    beforeAll(async () => {
      savedCwd = process.cwd();
      repo = createScratchRepo();
      process.chdir(repo.dir);

      // The benches run inside throwaway worktrees, so they append to one log in
      // the main repo dir: a single file records the interleaving across targets.
      const log = path.join(repo.dir, "round-robin.txt");
      for (const [name, latency] of [
        ["base3", 100],
        ["candidate-a", 80],
        ["candidate-b", 120],
      ] as const) {
        createBranch(repo, {
          name,
          benchScript: `#!/bin/sh\necho ${name} >> "${toShellPath(log)}"\necho "METRIC latency=${latency}"`,
        });
      }

      result = await compare(
        compareOptions({
          baseline: { target: "base3" },
          candidates: [{ target: "candidate-a" }, { target: "candidate-b" }],
        }),
      );
      benchOrder = fs.readFileSync(log, "utf-8").trim().split("\n");
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      process.chdir(savedCwd);
      removeStrandedWorktrees(repo);
      repo.cleanup();
    });

    it("runs the bench once per target per round, baseline first, in argument order", () => {
      expect(benchOrder).toStrictEqual([
        "base3",
        "candidate-a",
        "candidate-b",
        "base3",
        "candidate-a",
        "candidate-b",
        "base3",
        "candidate-a",
        "candidate-b",
      ]);
    });

    it("carries the baseline once and every candidate in argument order", () => {
      expect.soft(result.baselineLabel).toBe("base3");
      expect
        .soft(result.candidates.map((candidate) => candidate.label))
        .toStrictEqual(["candidate-a", "candidate-b"]);
      expect(result.metrics["latency"]?.baselineMedian).toBe(100);
    });

    it("gives each candidate its own verdict and geomean against the shared baseline", () => {
      const latency = result.metrics["latency"];

      expect
        .soft(latency?.candidates.map((candidate) => candidate.median))
        .toStrictEqual([80, 120]);
      expect
        .soft(latency?.candidates.map((candidate) => candidate.verdict?.verdict))
        .toStrictEqual(["improved", "regressed"]);
      expect.soft(result.candidates[0]?.kinds[0]?.geomean.value).toBeLessThan(0);
      expect(result.candidates[1]?.kinds[0]?.geomean.value).toBeGreaterThan(0);
    });

    it("creates and removes one worktree per ref target", () => {
      expect.soft(result.worktreesRemoved).toBe(3);
      expect.soft(result.worktreesLeftBehind).toStrictEqual([]);
      assertWorktreesCleanedUp(repo);
    });
  });

  describe("when a candidate's bench fails in a three-target run", () => {
    it(
      "removes the worktree of every target the run created",
      async () => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: "base-fail3",
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });
          createBranch(repo, {
            name: "candidate-ok3",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });
          createBranch(repo, {
            name: "candidate-bad3",
            benchScript: '#!/bin/sh\necho "boom" >&2\nexit 1',
          });

          const failure = await captureRejection(
            compare(
              compareOptions({
                baseline: { target: "base-fail3" },
                candidates: [{ target: "candidate-ok3" }, { target: "candidate-bad3" }],
              }),
            ),
          );

          assertCommandError(failure);
          expect.soft(failure.label).toBe("candidate-bad3");
          expect.soft(failure.message).toContain("boom");
          expect.soft(failure.message).not.toContain("left behind");
          assertWorktreesCleanedUp(repo);
        });
      },
      LONG_RUN_TIMEOUT_MS,
    );
  });

  describe("when targets are plain directories rather than refs", () => {
    let repo: ReturnType<typeof createScratchRepo>;
    let savedCwd: string;
    let result: ComparisonResult;
    let report: string;

    beforeAll(async () => {
      savedCwd = process.cwd();
      repo = createScratchRepo();
      process.chdir(repo.dir);

      for (const [name, latency] of [
        ["old-dir", 100],
        ["new-dir", 90],
      ] as const) {
        createInPlaceTargetDir(repo, name, `#!/bin/sh\necho "METRIC latency=${latency}"`);
      }

      result = await compare(
        compareOptions({
          baseline: { target: "old-dir" },
          candidates: [{ target: "new-dir" }],
        }),
      );
      report = renderReport(result);
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      process.chdir(savedCwd);
      repo.cleanup();
    });

    it("benches in place and labels each column with its directory name", () => {
      const headerRow = findLine(report, (line) => line.startsWith("metric"));
      expect.soft(headerRow).toContain("old-dir");
      expect.soft(headerRow).toContain("new-dir");
      // No worktree was created, so cleanup has nothing to report and stays
      // silent — the prune sweep is skipped, and sweeping anyway would fail
      // on targets that need not be git repositories at all.
      expect(report).not.toContain("worktrees removed");
    });

    it("returns the comparison data rather than the rendered report", () => {
      expect.soft(result.baselineLabel).toBe("old-dir");
      expect.soft(result.candidates.map((candidate) => candidate.label)).toStrictEqual(["new-dir"]);
      expect.soft(result.samples).toBe(3);
      expect.soft(result.adapter).toBe("metric-lines");
      expect.soft(result.metrics["latency"]?.baselineMedian).toBe(100);
      expect.soft(result.metrics["latency"]?.candidates[0]?.median).toBe(90);
      expect(result.worktreesRemoved).toBe(0);
    });
  });

  describe("when explicit labels are supplied", () => {
    it("uses them in the report instead of the ref names", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-labelled",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-labelled",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-labelled", label: "baseline" },
              candidates: [{ target: "new-labelled", label: "candidate" }],
            }),
          ),
        );

        const headerRow = findLine(report, (line) => line.startsWith("metric"));
        expect.soft(headerRow).toContain("baseline");
        expect.soft(headerRow).toContain("candidate");
        expect.soft(report).not.toContain("old-labelled");
        expect(report).not.toContain("new-labelled");
      });
    });
  });

  describe("when comparing refs with metric-lines adapter and different metric sets", () => {
    it("renders union of metrics from both refs with one-sided rows", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"\necho "METRIC memory=200"',
        });

        createBranch(repo, {
          name: "new-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=80"\necho "METRIC throughput=500"',
        });

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-branch" },
              candidates: [{ target: "new-branch" }],
            }),
          ),
        );

        expect.soft(report).toContain("latency");
        expect.soft(report).toContain("throughput");
        // memory is baseline-only, so its row's candidate cell is empty — a
        // trailing separator means the candidate and verdict cells were trimmed away.
        const memoryRow = findLine(report, (line) => line.startsWith("memory"));
        expect(memoryRow.endsWith("│")).toBe(true);
        assertWorktreesCleanedUp(repo);
      });
    });
  });

  describe("when a metric is named after an Object.prototype member", () => {
    it("renders a baseline-only toString metric as a one-sided row", async () => {
      await withScratchRepo(async (repo) => {
        // In-place targets keep the run to the two benches: no worktree is created.
        createInPlaceTargetDir(
          repo,
          "old-proto",
          '#!/bin/sh\necho "METRIC toString=100"\necho "METRIC latency=100"',
        );
        createInPlaceTargetDir(repo, "new-proto", '#!/bin/sh\necho "METRIC latency=90"');

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-proto" },
              candidates: [{ target: "new-proto" }],
            }),
          ),
        );

        // The candidate bench never emitted toString, so its cell has to be empty —
        // a trailing separator means the candidate and verdict cells were trimmed
        // away. Reading the name off Object.prototype instead would fill them with
        // a value no bench produced.
        const toStringRow = findLine(report, (line) => line.startsWith("toString"));
        expect.soft(toStringRow).toContain("100");
        expect(toStringRow).toMatch(/│$/);
      });
    });
  });

  describe("when using mitata adapter with fixture replay", () => {
    let repo: ReturnType<typeof createScratchRepo>;
    let savedCwd: string;
    let result: ComparisonResult;

    beforeAll(async () => {
      savedCwd = process.cwd();
      repo = createScratchRepo();
      process.chdir(repo.dir);

      const fixturePath = path.resolve(savedCwd, "tests/fixtures/mitata.json");
      const mitataBenchScript = `#!/bin/sh\ncat "${toShellPath(fixturePath)}"`;
      createBranch(repo, { name: "mitata-branch", benchScript: mitataBenchScript });
      createBranch(repo, { name: "mitata-branch-2", benchScript: mitataBenchScript });

      result = await compare(
        compareOptions({
          baseline: { target: "mitata-branch" },
          candidates: [{ target: "mitata-branch-2" }],
          adapter: "mitata",
        }),
      );
    });

    afterAll(() => {
      process.chdir(savedCwd);
      repo.cleanup();
    });

    it("parses the mitata JSON fixture into metrics keyed by benchmark and stat", () => {
      // Both branches replay the same fixture, so baseline and candidate
      // medians must match exactly — a swapped side or a wrong median would
      // make them differ.
      expect.soft(result.metrics["decode/text=digits/time"]?.baselineMedian).toBe(4.0791015625);
      expect
        .soft(result.metrics["decode/text=digits/time"]?.candidates[0]?.median)
        .toBe(4.0791015625);
      expect.soft(result.metrics["decode/text=words/time"]?.baselineMedian).toBe(7.8125);
      expect.soft(result.metrics["decode/text=words/time"]?.candidates[0]?.median).toBe(7.8125);
      expect.soft(result.metrics["encode/time"]?.baselineMedian).toBe(42.66357421875);
      expect(result.metrics["encode/time"]?.candidates[0]?.median).toBe(42.66357421875);
    });

    it("renders the parsed benchmarks in the report", () => {
      const report = renderReport(result);

      expect.soft(report).toContain("decode");
      expect.soft(report).toContain("encode");
      expect(report).toContain("time");
    });
  });

  describe("when the sample count selects the verdict method", () => {
    it.each([
      { samples: 10, newLatency: 80, expectedFooter: "Wilcoxon signed-rank" },
      { samples: 3, newLatency: 90, expectedFooter: "noise band ±(half-range × K)" },
    ])(
      "reports $expectedFooter for $samples samples",
      async ({ samples, newLatency, expectedFooter }) => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: `old-${samples}`,
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: `new-${samples}`,
            benchScript: `#!/bin/sh\necho "METRIC latency=${newLatency}"`,
          });

          const report = renderReport(
            await compare(
              compareOptions({
                baseline: { target: `old-${samples}` },
                candidates: [{ target: `new-${samples}` }],
                samples,
              }),
            ),
            { verbose: true },
          );

          expect(report).toContain(expectedFooter);
        });
      },
    );
  });

  describe("when an unstable noise threshold is configured", () => {
    /**
     * A bench whose successive runs emit 40, 50 then 60 for `latency`.
     *
     * The counter file lives in the main repo dir so it survives the throwaway
     * worktree each run happens in. Paired against a flat 100 on the other side,
     * the spread works out to a noise band of 30%: median 50, half-range 10,
     * 1.5 × 100 × (10 / 50).
     */
    const noisyBenchScript = (counterFile: string) => {
      const p = toShellPath(counterFile);
      return [
        "#!/bin/sh",
        `echo x >> "${p}"`,
        `n=$(wc -l < "${p}" | tr -d ' ')`,
        'echo "METRIC latency=$((30 + 10 * n))"',
      ].join("\n");
    };

    it.each([
      {
        unstableNoisePct: 20,
        expectedVerdict: "≈  unstable",
        expectedGeomean: "no stable metrics",
      },
      { unstableNoisePct: 40, expectedVerdict: "✓  -50.0%", expectedGeomean: "-50.0%" },
    ])(
      "renders '$expectedVerdict' for a metric with 30% noise when unstableNoisePct is $unstableNoisePct",
      async ({ unstableNoisePct, expectedVerdict, expectedGeomean }) => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: `old-noisy-${unstableNoisePct}`,
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: `new-noisy-${unstableNoisePct}`,
            benchScript: noisyBenchScript(path.join(repo.dir, "noisy-runs.txt")),
          });

          const report = renderReport(
            await compare(
              compareOptions({
                baseline: { target: `old-noisy-${unstableNoisePct}` },
                candidates: [{ target: `new-noisy-${unstableNoisePct}` }],
                unstableNoisePct,
              }),
            ),
          );

          const latencyRow = findLine(report, (line) => line.startsWith("latency"));
          expect.soft(latencyRow).toContain(expectedVerdict);
          // The run's only gating metric is the unstable one, so excluding it leaves the
          // geomean with nothing to average; a stable one keeps its ρ and tracks the -50%.
          expect(findLine(report, (line) => line.includes("geomean"))).toContain(expectedGeomean);
        });
      },
    );
  });

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
        expect.soft(error().phase).toBe("bench");
        expect.soft(error().position).toBe(side);
        expect.soft(error().label).toBe(`${side}-fail-${side}`);
        expect.soft(error().command).toBe("sh bench.sh");
        expect.soft(error().exitCode).toBe(1);
        expect.soft(error().sample).toBe(1);
        expect(error().target).toStrictEqual(
          expect.objectContaining({ kind: "ref", ref: `${side}-fail-${side}` }),
        );

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
      expect.soft(error().phase).toBe("bench");
      expect.soft(error().position).toBe("old");
      expect.soft(error().label).toBe("old-slow");
      expect.soft(error().command).toBe("sh bench.sh");
      expect.soft(error().timeoutMs).toBe(500);
      expect.soft(error().exitCode).toBeUndefined();
      expect
        .soft(error().target)
        .toStrictEqual(expect.objectContaining({ kind: "ref", ref: "old-slow" }));
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
      expect.soft(error().phase).toBe("prepare");
      expect.soft(error().position).toBe("old");
      expect.soft(error().label).toBe("old-prep-fail");
      expect.soft(error().command).toBe("sh prepare.sh");
      expect.soft(error().exitCode).toBe(1);
      expect.soft(error().sample).toBeUndefined();
      expect(error().target).toStrictEqual(
        expect.objectContaining({ kind: "ref", ref: "old-prep-fail" }),
      );

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
      expect.soft(error().phase).toBe("prepare");
      expect.soft(error().position).toBe("old");
      expect.soft(error().label).toBe("old-prep-slow");
      expect.soft(error().command).toBe("sh prepare.sh");
      expect.soft(error().timeoutMs).toBe(500);
      expect.soft(error().exitCode).toBeUndefined();
      expect
        .soft(error().target)
        .toStrictEqual(expect.objectContaining({ kind: "ref", ref: "old-prep-slow" }));
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
        const error = failure;

        expect.soft(error.phase).toBe("bench");
        expect.soft(error.position).toBe("old");
        expect.soft(error.label).toBe("old-dir-fail");
        expect.soft(error.command).toBe("sh bench.sh");
        expect.soft(error.exitCode).toBe(1);
        expect.soft(error.sample).toBe(1);
        expect(error.target.kind).toBe("in-place");
        expect(error.dir).toContain("old-dir-fail");

        expect(error.message).toContain("dir bench failed");
        expect(error.message).toMatch(/dir:\s+\S/);
        expect(error.hint).toBeUndefined();
        expect(error.message).not.toContain("worktree:");
      });
    });
  });

  describe("when git cannot remove a worktree the run created", () => {
    it("reports the cleanup outcome the run actually produced", async () => {
      await withScratchRepo(async (repo) => {
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

        const leftBehindDirs = parseLeftBehindDirs(report);
        expect(report).toContain("1 worktree removed · 1 left behind");
        expect(leftBehindDirs).toHaveLength(1);
        expect(leftBehindDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual(leftBehindDirs);
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

  describe("when metric values are zero", () => {
    it("renders a zero median as 0 ± 0% and a 0.0% delta rather than NaN", async () => {
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
        expect(latencyRow).toMatch(/^latency\s*│\s*0 ± 0%\s*│\s*0 ± 0%\s*│\s*~\s+0\.0%/);
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
      listenersBefore = {
        SIGINT: process.listeners("SIGINT"),
        SIGTERM: process.listeners("SIGTERM"),
        SIGHUP: process.listeners("SIGHUP"),
      };
      vi.spyOn(process, "exit").mockImplementation((code) => {
        throw new ProcessExited(code);
      });
    });

    afterEach(() => {
      vi.restoreAllMocks();

      // On every passing path `compare()` has already uninstalled its own
      // handlers by the time the test's `finally` awaits the run's settlement,
      // so this normally finds nothing. It fires only when setup threw before
      // the run could settle — and it asserts rather than scrubbing silently,
      // because a handler surviving a settled run is the exact regression the
      // sibling listener-count test exists to catch.
      const leaked = TERMINATION_SIGNALS.filter((signal) =>
        removeLeakedListeners(signal, listenersBefore[signal]),
      );

      if (leaked.length > 0) {
        throw new Error(`compare() left ${leaked.join(", ")} handler(s) installed after settling`);
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
      let savedCwd: string;

      beforeAll(async () => {
        savedCwd = process.cwd();
        const signalListenersBefore = {
          SIGINT: process.listeners("SIGINT"),
          SIGTERM: process.listeners("SIGTERM"),
        };
        vi.spyOn(process, "exit").mockImplementation((code) => {
          throw new ProcessExited(code);
        });

        repo = createScratchRepo();
        process.chdir(repo.dir);
        run = await startInFlightRun(repo);
        // Without this guard, the post-signal assertions below would also pass for
        // a pid that was never alive — `waitForPid` only proves a number reached
        // the file. `expect()` isn't usable in `beforeAll`, so this throws instead.
        if (!isAlive(run.benchGrandchildPid)) {
          throw new Error(`bench grandchild pid ${run.benchGrandchildPid} is not alive`);
        }

        raiseSignal("SIGINT", signalListenersBefore.SIGINT);
      }, LONG_RUN_TIMEOUT_MS);

      afterAll(async () => {
        vi.restoreAllMocks();
        process.chdir(savedCwd);
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
      "holds one handler per signal only until the run has $outcome",
      async ({ outcome, benchScript }) => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, { name: "old-settled", benchScript });
          createBranch(repo, {
            name: "new-settled",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });

          const before = signalListenerCounts();

          const running = compare(
            compareOptions({
              baseline: { target: "old-settled" },
              candidates: [{ target: "new-settled" }],
              timeoutSeconds: 20,
            }),
          );
          const duringRun = signalListenerCounts();
          const settled = await running.then(
            () => "resolved",
            () => "rejected",
          );

          expect(settled).toBe(outcome);
          expect(duringRun).toStrictEqual({
            SIGINT: before.SIGINT + 1,
            SIGTERM: before.SIGTERM + 1,
            SIGHUP: before.SIGHUP + 1,
          });
          expect(signalListenerCounts()).toStrictEqual(before);
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
