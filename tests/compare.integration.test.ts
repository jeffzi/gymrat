import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";

import { compare } from "../src/compare.js";
import type { CompareOptions } from "../src/compare.js";
import { createScratchRepo } from "./fixtures/scratch-repo.js";

interface BranchSetup {
  name: string;
  benchScript: string;
  prepareScript?: string;
}

/**
 * Create a git branch with bench and optional prepare scripts.
 */
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

  if (setup.prepareScript) {
    fs.writeFileSync(path.join(repo.dir, "prepare.sh"), setup.prepareScript);
    execSync("chmod +x prepare.sh bench.sh && git add prepare.sh bench.sh", {
      cwd: repo.dir,
      stdio: "pipe",
    });
  } else {
    execSync("chmod +x bench.sh && git add bench.sh", {
      cwd: repo.dir,
      stdio: "pipe",
    });
  }

  execSync(`git commit -m '${setup.name}'`, {
    cwd: repo.dir,
    stdio: "pipe",
  });
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
 * Extract the directories named by `left behind: <dir> <reason>` entries.
 *
 * Requires a non-empty reason on the same line, so a multi-line git diagnostic
 * that leaked into the text would not match.
 */
function parseLeftBehindDirs(text: string): string[] {
  return text.split("\n").flatMap((line) => /^ {2}left behind: (\S+) \S/.exec(line)?.[1] ?? []);
}

/**
 * The worktree directories the repo still lists, main repo dir excluded.
 *
 * git prints resolved paths, so the main dir is matched through `realpath` — on
 * macOS the scratch repo lives under `/var/...` and git reports `/private/var/...`.
 */
function listWorktreeDirs(repo: ReturnType<typeof createScratchRepo>): string[] {
  const worktreeList = execSync("git worktree list", {
    cwd: repo.dir,
    stdio: "pipe",
    encoding: "utf-8",
  });
  const mainDir = fs.realpathSync(repo.dir);
  return worktreeList
    .split("\n")
    .flatMap((line) => /^(\S+)\s/.exec(line)?.[1] ?? [])
    .filter((dir) => dir !== mainDir);
}

/**
 * Delete every worktree directory the repo still lists, main repo dir excluded.
 *
 * Keeps a run that stranded a worktree from leaking it into the system temp dir,
 * whatever the assertions did or did not manage to read.
 */
function removeStrandedWorktrees(repo: ReturnType<typeof createScratchRepo>) {
  for (const dir of listWorktreeDirs(repo)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * Verify all worktrees except the main repo dir have been cleaned up.
 */
function assertWorktreesCleanedUp(repo: ReturnType<typeof createScratchRepo>) {
  expect(listWorktreeDirs(repo)).toStrictEqual([]);
}

type SignalName = "SIGINT" | "SIGTERM";

const TERMINATION_SIGNALS: readonly SignalName[] = ["SIGINT", "SIGTERM"];

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
  };
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

/** True while a process with `pid` exists. */
function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** Reads a pid written by `echo $! > file`; NaN while the write is absent or incomplete. */
function readPid(pidPath: string): number {
  const raw = fs.readFileSync(pidPath, "utf8");
  return raw.endsWith("\n") ? Number.parseInt(raw, 10) : Number.NaN;
}

async function waitForPid(pidPath: string): Promise<number> {
  let pid = Number.NaN;
  await vi.waitFor(
    () => {
      pid = readPid(pidPath);
      expect(pid).toBeGreaterThan(0);
    },
    { timeout: 10_000, interval: 25 },
  );
  return pid;
}

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
    benchScript: `#!/bin/sh\nsleep 30 &\necho $! > "${pidFile}"\nwait`,
  });
  createBranch(repo, {
    name: "new-signal",
    benchScript: '#!/bin/sh\necho "METRIC latency=90"',
  });

  const options: CompareOptions = {
    oldTarget: "old-signal",
    newTarget: "new-signal",
    bench: "./bench.sh",
    adapter: "metric-lines",
    samples: 3,
    timeoutSeconds: 20,
  };

  // Swallowed: once a signal has torn the run down, whether compare() reports the
  // killed bench or its vanished worktree is beside the point being tested.
  const settled = compare(options).then(
    () => undefined,
    () => undefined,
  );
  const benchGrandchildPid = await waitForPid(pidFile);

  return { settled, benchGrandchildPid, worktreeDirs: listWorktreeDirs(repo) };
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
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        // The scripts run inside throwaway worktrees, so they append to a log in
        // the main repo dir that outlives cleanup.
        const oldLog = path.join(repo.dir, "old-log.txt");
        const newLog = path.join(repo.dir, "new-log.txt");

        createBranch(repo, {
          name: "old-prep",
          benchScript: `#!/bin/sh\necho bench >> "${oldLog}"\necho "METRIC latency=100"`,
          prepareScript: `#!/bin/sh\necho prepare >> "${oldLog}"`,
        });

        createBranch(repo, {
          name: "new-prep",
          benchScript: `#!/bin/sh\necho bench >> "${newLog}"\necho "METRIC latency=90"`,
          prepareScript: `#!/bin/sh\necho prepare >> "${newLog}"`,
        });

        const options: CompareOptions = {
          oldTarget: "old-prep",
          newTarget: "new-prep",
          bench: "./bench.sh",
          prepare: "./prepare.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        await compare(options);

        const readLog = (file: string) => fs.readFileSync(file, "utf-8").trim().split("\n");
        expect(readLog(oldLog)).toStrictEqual(["prepare", "bench", "bench", "bench"]);
        expect(readLog(newLog)).toStrictEqual(["prepare", "bench", "bench", "bench"]);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when targets are plain directories rather than refs", () => {
    it("benches in place and labels each column with its directory name", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        // Untracked subdirectories of the repo: resolveTarget sees a directory
        // and returns an in-place target, so no worktree is ever created.
        for (const [name, latency] of [
          ["old-dir", 100],
          ["new-dir", 90],
        ] as const) {
          fs.mkdirSync(path.join(repo.dir, name));
          const script = path.join(repo.dir, name, "bench.sh");
          fs.writeFileSync(script, `#!/bin/sh\necho "METRIC latency=${latency}"`);
          fs.chmodSync(script, 0o755);
        }

        const options: CompareOptions = {
          oldTarget: "old-dir",
          newTarget: "new-dir",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("old (old-dir)");
        expect(report).toContain("new (new-dir)");
        // No worktree was created, so cleanup reports zero of both rather than
        // a failure — the prune sweep is skipped, and sweeping anyway would
        // fail on targets that need not be git repositories at all.
        expect(report).toContain("0 worktrees removed · 0 left behind");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when explicit labels are supplied", () => {
    it("uses them in the report instead of the ref names", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-labelled",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-labelled",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const options: CompareOptions = {
          oldTarget: "old-labelled",
          newTarget: "new-labelled",
          oldLabel: "baseline",
          newLabel: "candidate",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("old (baseline)");
        expect(report).toContain("new (candidate)");
        expect(report).not.toContain("old-labelled");
        expect(report).not.toContain("new-labelled");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when comparing refs with metric-lines adapter and different metric sets", () => {
    it("renders union of metrics from both refs with one-sided rows", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"\necho "METRIC memory=200"',
        });

        createBranch(repo, {
          name: "new-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=80"\necho "METRIC throughput=500"',
        });

        const options: CompareOptions = {
          oldTarget: "old-branch",
          newTarget: "new-branch",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("latency");
        expect(report).toContain("throughput");
        assertWorktreesCleanedUp(repo);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when using mitata adapter with fixture replay", () => {
    it("parses mitata JSON fixture and generates report", async () => {
      const repo = createScratchRepo();
      const fixturePath = path.resolve(originalCwd, "tests/fixtures/mitata.json");

      try {
        process.chdir(repo.dir);

        const mitataBenchScript = `#!/bin/sh\ncat "${fixturePath}"`;
        createBranch(repo, {
          name: "mitata-branch",
          benchScript: mitataBenchScript,
        });

        createBranch(repo, {
          name: "mitata-branch-2",
          benchScript: mitataBenchScript,
        });

        const options: CompareOptions = {
          oldTarget: "mitata-branch",
          newTarget: "mitata-branch-2",
          bench: "./bench.sh",
          adapter: "mitata",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("decode");
        expect(report).toContain("encode");
        expect(report).toContain("time");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when the sample count selects the verdict method", () => {
    it.each([
      { samples: 10, newLatency: 80, expectedFooter: "Wilcoxon signed-rank" },
      { samples: 3, newLatency: 90, expectedFooter: "noise band ±(half-range × K)" },
    ])(
      "reports $expectedFooter for $samples samples",
      async ({ samples, newLatency, expectedFooter }) => {
        const repo = createScratchRepo();

        try {
          process.chdir(repo.dir);

          createBranch(repo, {
            name: `old-${samples}`,
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: `new-${samples}`,
            benchScript: `#!/bin/sh\necho "METRIC latency=${newLatency}"`,
          });

          const options: CompareOptions = {
            oldTarget: `old-${samples}`,
            newTarget: `new-${samples}`,
            bench: "./bench.sh",
            adapter: "metric-lines",
            samples,
            timeoutSeconds: 10,
          };

          const report = await compare(options);

          expect(report).toContain(expectedFooter);
        } finally {
          repo.cleanup();
        }
      },
    );
  });

  describe("when bench command fails", () => {
    const failingBench = '#!/bin/sh\necho "stderr output" >&2\nexit 1';
    const passingBench = '#!/bin/sh\necho "METRIC latency=100"';

    it.each([
      { side: "old", oldBench: failingBench, newBench: passingBench },
      { side: "new", oldBench: passingBench, newBench: failingBench },
    ])(
      "throws an error carrying the stderr captured from the $side target",
      async ({ side, oldBench, newBench }) => {
        const repo = createScratchRepo();

        try {
          process.chdir(repo.dir);

          createBranch(repo, { name: `old-fail-${side}`, benchScript: oldBench });
          createBranch(repo, { name: `new-fail-${side}`, benchScript: newBench });

          const options: CompareOptions = {
            oldTarget: `old-fail-${side}`,
            newTarget: `new-fail-${side}`,
            bench: "./bench.sh",
            adapter: "metric-lines",
            samples: 3,
            timeoutSeconds: 10,
          };

          const failure = await captureRejection(compare(options));

          expect(failure.message).toMatch(/stderr output/);
          expect(failure.message).not.toContain("left behind");
          assertWorktreesCleanedUp(repo);
        } finally {
          repo.cleanup();
        }
      },
    );
  });

  describe("when the bench command exceeds the timeout", () => {
    it("throws naming the target and the elapsed timeout, and still cleans up", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, { name: "old-slow", benchScript: "#!/bin/sh\nsleep 10" });
        createBranch(repo, {
          name: "new-slow",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const options: CompareOptions = {
          oldTarget: "old-slow",
          newTarget: "new-slow",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          // Fractional seconds keep the test near half a second; the sleep is
          // killed with its process group, so nothing outlives the run.
          timeoutSeconds: 0.5,
        };

        const failure = await captureRejection(compare(options));

        expect(failure.message).toMatch(/Bench command on old target timed out after 500ms/);
        expect(failure.message).not.toContain("left behind");
        assertWorktreesCleanedUp(repo);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when git cannot remove a worktree the run created", () => {
    it("reports the cleanup outcome the run actually produced", async () => {
      const repo = createScratchRepo();
      const leftBehindDirs: string[] = [];

      try {
        process.chdir(repo.dir);

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

        const options: CompareOptions = {
          oldTarget: "old-unremovable",
          newTarget: "new-unremovable",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        leftBehindDirs.push(
          ...report.split("\n").flatMap((line) => /^ {2}left behind: (\S+) /.exec(line)?.[1] ?? []),
        );
        expect(report).toContain("1 worktree removed · 1 left behind");
        expect(leftBehindDirs).toHaveLength(1);
        expect(leftBehindDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual(leftBehindDirs);
      } finally {
        for (const dir of leftBehindDirs) {
          fs.rmSync(dir, { recursive: true, force: true });
        }
        repo.cleanup();
      }
    });
  });

  describe("when the bench fails and git cannot remove a worktree", () => {
    it("throws an error naming the stranded worktree alongside the original failure", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

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

        const options: CompareOptions = {
          oldTarget: "old-fail-unremovable",
          newTarget: "new-fail-unremovable",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const failure = await captureRejection(compare(options));

        const leftBehindDirs = parseLeftBehindDirs(failure.message);
        expect(failure.message).toMatch(/stderr output/);
        expect(leftBehindDirs).toHaveLength(1);
        expect(leftBehindDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual(leftBehindDirs);
        expect(failure.cause).toBeInstanceOf(Error);
      } finally {
        removeStrandedWorktrees(repo);
        repo.cleanup();
      }
    });
  });

  describe("when metric values are zero", () => {
    it("handles zero values gracefully in median and spread calculation", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-zero",
          benchScript: '#!/bin/sh\necho "METRIC latency=0"',
        });

        createBranch(repo, {
          name: "new-zero",
          benchScript: '#!/bin/sh\necho "METRIC latency=0"',
        });

        const options: CompareOptions = {
          oldTarget: "old-zero",
          newTarget: "new-zero",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        // Zero medians must render as a zero spread and a zero delta rather than
        // NaN or a division blow-up, so assert the row itself, not just the name.
        const latencyRow = report.split("\n").find((line) => line.startsWith("latency"));
        expect(latencyRow).toBeDefined();
        expect(latencyRow!).toMatch(/^latency\s*│\s*0 ± 0%\s*│\s*0 ± 0%\s*│\s*~ 0\.0%/);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when bench script produces no metrics", () => {
    it("throws error indicating no metrics found", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-empty",
          benchScript: '#!/bin/sh\necho "no metrics here"',
        });

        createBranch(repo, {
          name: "new-empty",
          benchScript: '#!/bin/sh\necho "also no metrics"',
        });

        const options: CompareOptions = {
          oldTarget: "old-empty",
          newTarget: "new-empty",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        await expect(compare(options)).rejects.toThrow(/[Nn]o valid METRIC|[Nn]o metrics/);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when a termination signal arrives mid-run", () => {
    let listenersBefore: Record<SignalName, readonly unknown[]>;

    beforeEach(() => {
      listenersBefore = {
        SIGINT: process.listeners("SIGINT"),
        SIGTERM: process.listeners("SIGTERM"),
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
      const leaked: SignalName[] = [];
      for (const signal of TERMINATION_SIGNALS) {
        for (const listener of process.listeners(signal)) {
          if (!listenersBefore[signal].includes(listener) && isSignalListener(listener)) {
            process.removeListener(signal, listener);
            leaked.push(signal);
          }
        }
      }

      if (leaked.length > 0) {
        throw new Error(`compare() left ${leaked.join(", ")} handler(s) installed after settling`);
      }
    });

    it.each<{ signal: SignalName; expectedCode: number }>([
      { signal: "SIGINT", expectedCode: 130 },
      { signal: "SIGTERM", expectedCode: 143 },
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
      60_000,
    );

    it("removes the worktrees it created", async () => {
      const repo = createScratchRepo();
      let run: InFlightRun | undefined;

      try {
        process.chdir(repo.dir);
        run = await startInFlightRun(repo);

        raiseSignal("SIGINT", listenersBefore.SIGINT);

        expect(run.worktreeDirs.length).toBeGreaterThan(0);
        expect(run.worktreeDirs.filter((dir) => fs.existsSync(dir))).toStrictEqual([]);
        assertWorktreesCleanedUp(repo);
      } finally {
        await cleanupInFlightRun(repo, run);
      }
    }, 60_000);

    it("kills the bench process group, leaving nothing running in the removed worktree", async () => {
      const repo = createScratchRepo();
      let run: InFlightRun | undefined;

      try {
        process.chdir(repo.dir);
        run = await startInFlightRun(repo);
        // Without this, the assertion below would also pass for a pid that was
        // never alive — `waitForPid` only proves a number reached the file.
        expect(isAlive(run.benchGrandchildPid)).toBe(true);

        raiseSignal("SIGINT", listenersBefore.SIGINT);

        const grandchildPid = run.benchGrandchildPid;
        await vi.waitFor(
          () => {
            expect(isAlive(grandchildPid)).toBe(false);
          },
          { timeout: 5000, interval: 25 },
        );
      } finally {
        await cleanupInFlightRun(repo, run);
      }
    }, 60_000);
  });

  describe("when a run finishes on its own", () => {
    it.each<{ outcome: "resolved" | "rejected"; benchScript: string }>([
      { outcome: "resolved", benchScript: '#!/bin/sh\necho "METRIC latency=100"' },
      { outcome: "rejected", benchScript: "#!/bin/sh\nexit 1" },
    ])(
      "holds one handler per signal only until the run has $outcome",
      async ({ outcome, benchScript }) => {
        const repo = createScratchRepo();

        try {
          process.chdir(repo.dir);

          createBranch(repo, { name: "old-settled", benchScript });
          createBranch(repo, {
            name: "new-settled",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });

          const options: CompareOptions = {
            oldTarget: "old-settled",
            newTarget: "new-settled",
            bench: "./bench.sh",
            adapter: "metric-lines",
            samples: 3,
            timeoutSeconds: 20,
          };
          const before = signalListenerCounts();

          const running = compare(options);
          const duringRun = signalListenerCounts();
          const settled = await running.then(
            () => "resolved",
            () => "rejected",
          );

          expect(settled).toBe(outcome);
          expect(duringRun).toStrictEqual({
            SIGINT: before.SIGINT + 1,
            SIGTERM: before.SIGTERM + 1,
          });
          expect(signalListenerCounts()).toStrictEqual(before);
        } finally {
          removeStrandedWorktrees(repo);
          repo.cleanup();
        }
      },
      60_000,
    );
  });
});
