/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from "vitest";

import { createProgram } from "../src/cli.js";
import type { ResolvedConfig } from "../src/config.js";
import { startSession } from "../src/loop/start.js";
import type { ComparisonResult, MeasurementResult } from "../src/report/types.js";
import {
  experimentWorktreeDir,
  lockfilePath,
  repoRoot,
  sessionJsonlPath,
} from "../src/session/paths.js";
import type { SessionLogRecord } from "../src/session/records.js";
import { appendRecord, readRecords } from "../src/session/store.js";
import {
  captureStdout,
  createRunnableProgram,
  mockProcessExit,
  stubWrite,
} from "./fixtures/cli-harness.js";
import { createComparisonResult } from "./fixtures/comparison-result.js";
import { ISO_PATTERN } from "./fixtures/constants.js";
import { createMeasurementResult } from "./fixtures/measurement-result.js";
import { createScratchRepo, git, type ScratchRepo } from "./fixtures/scratch-repo.js";
import { committedKeep, iterationRecord } from "./fixtures/session-records.js";

// `...actual` is spread so CommandError passes through unmocked and tests can construct real instances.
vi.mock("../src/compare.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/compare.js")>();
  return {
    ...actual,
    compare: vi.fn(),
  };
});

// `...actual` is spread for the same reason as compare's mock: measure re-exports
// CommandError, and tests construct real instances of it.
vi.mock("../src/measure.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/measure.js")>();
  return {
    ...actual,
    measure: vi.fn(),
  };
});

vi.mock("../src/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/config.js")>();
  return {
    ...actual,
    resolveConfig: vi.fn(),
    resolveBenchlessConfig: vi.fn(),
  };
});

vi.mock("../src/config-inspect.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/config-inspect.js")>();
  return {
    ...actual,
    inspectConfig: vi.fn(),
  };
});

const { mockConfirmAction } = vi.hoisted(() => ({
  mockConfirmAction: vi.fn<(message: string, stream: NodeJS.ReadableStream) => Promise<boolean>>(),
}));

vi.mock("../src/confirm.js", () => ({
  confirmAction: mockConfirmAction,
}));

vi.mock("../src/report/json.js", () => ({
  renderJson: vi.fn().mockReturnValue('{"report": true}'),
  renderMeasureJson: vi.fn().mockReturnValue('{"measurement": true}'),
}));

vi.mock("../src/doctor/checks.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/doctor/checks.js")>();
  return {
    ...actual,
    buildEnvironmentSection: vi.fn(),
    buildConfigSection: vi.fn(),
    buildWorkflowSection: vi.fn(),
  };
});

vi.mock("../src/doctor/bench.js", () => ({
  buildBenchSection: vi.fn(),
}));

vi.mock("../src/doctor/render.js", () => ({
  renderDoctorReport: vi.fn().mockReturnValue("doctor text report"),
  renderDoctorJson: vi.fn().mockReturnValue('{"doctor": true}'),
}));

const { mockSpinnerInstance, mockYoctoSpinner, mockEtaRecord, mockFormatEta } = vi.hoisted(() => {
  const instance = {
    start: vi.fn(),
    stop: vi.fn(),
    clear: vi.fn(),
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    text: "",
    color: "cyan" as const,
    isSpinning: false,
  };
  // Self-referencing returns for chaining
  instance.start.mockReturnValue(instance);
  instance.stop.mockReturnValue(instance);
  instance.clear.mockReturnValue(instance);

  return {
    mockSpinnerInstance: instance,
    mockYoctoSpinner: vi.fn().mockReturnValue(instance),
    mockEtaRecord: vi.fn(),
    mockFormatEta: vi.fn<(ms: number) => string>(),
  };
});

vi.mock("yocto-spinner", () => ({
  default: mockYoctoSpinner,
}));

vi.mock("../src/eta.js", () => {
  class MockEtaTracker {
    record = mockEtaRecord;
  }
  return {
    EtaTracker: MockEtaTracker,
    formatEta: mockFormatEta,
  };
});

const { mockRunWizard, mockScaffold } = vi.hoisted(() => ({
  mockRunWizard: vi.fn(),
  mockScaffold: vi.fn(),
}));

vi.mock("../src/init/wizard.js", () => ({
  runWizard: mockRunWizard,
}));

vi.mock("../src/init/scaffold.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/init/scaffold.js")>();
  return {
    ...actual,
    scaffold: mockScaffold,
  };
});

function resolvedConfigFixture(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return {
    bench: "bench.sh",
    adapter: "metric-lines",
    samples: 1,
    timeoutSeconds: 300,
    unstableNoisePct: 200,
    primary: "geomean",
    ...overrides,
  };
}

/** A mock async function's settle-branch, shared by `mockResolvedValue`/`mockRejectedValue` stubs. */
interface ResolvableMock<T> {
  mockResolvedValue(value: T): unknown;
  mockRejectedValue(reason: unknown): unknown;
}

/**
 * Settle `mock` on `outcome`: rejects with it when it's an `Error`, otherwise
 * resolves with it (or `fallback` when omitted).
 *
 * The branch `setupMocks` and `setupMeasureMocks` share, differing only in
 * which mock and fallback result they settle.
 */
function stubOutcome<T>(
  mock: ResolvableMock<T>,
  outcome: T | Error | undefined,
  fallback: T,
): void {
  if (outcome instanceof Error) {
    mock.mockRejectedValue(outcome);
  } else {
    mock.mockResolvedValue(outcome ?? fallback);
  }
}

async function setupMocks(
  compareMockReturn?: ComparisonResult | Error,
  resolveConfigMockReturn: Partial<ResolvedConfig> = {},
) {
  const { compare: compareMock } = await import("../src/compare.js");
  const { resolveConfig: resolveConfigMock } = await import("../src/config.js");

  vi.mocked(resolveConfigMock).mockReturnValue(resolvedConfigFixture(resolveConfigMockReturn));
  stubOutcome(vi.mocked(compareMock), compareMockReturn, createComparisonResult());

  return { compareMock, resolveConfigMock };
}

/** `setupMocks` for the single-target command: stubs `resolveConfig` and `measure`. */
async function setupMeasureMocks(
  measureMockReturn?: MeasurementResult | Error,
  resolveConfigMockReturn: Partial<ResolvedConfig> = {},
) {
  const { measure: measureMock } = await import("../src/measure.js");
  const { resolveConfig: resolveConfigMock } = await import("../src/config.js");

  vi.mocked(resolveConfigMock).mockReturnValue(resolvedConfigFixture(resolveConfigMockReturn));
  stubOutcome(vi.mocked(measureMock), measureMockReturn, createMeasurementResult());

  return { measureMock, resolveConfigMock };
}

/**
 * Parse `argv` for its help text and return whatever was written to stdout.
 *
 * A subcommand renders its own help, so every command needs its own output
 * config; `--help` throws rather than exiting because of `exitOverride`.
 */
async function captureHelp(argv: string[]): Promise<string> {
  const program = createProgram();
  program.exitOverride();
  let helpOutput = "";
  for (const command of [program, ...program.commands]) {
    command.configureOutput({
      writeOut: (str) => {
        helpOutput += str;
      },
    });
  }

  await expect(program.parseAsync(argv)).rejects.toThrow("outputHelp");

  return helpOutput;
}

/** Prepends the `["node", "cli.js", "compare"]` prefix Commander expects. */
function compareArgv(...args: string[]): string[] {
  return ["node", "cli.js", "compare", ...args];
}

/** Prepends the `["node", "cli.js", "measure"]` prefix Commander expects. */
function measureArgv(...args: string[]): string[] {
  return ["node", "cli.js", "measure", ...args];
}

function stderrWrites(stderrSpy: ReturnType<typeof vi.spyOn>): unknown[] {
  return stderrSpy.mock.calls.map((c: unknown[]) => c[0]);
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  mockSpinnerInstance.text = "";
  mockSpinnerInstance.isSpinning = false;
});

describe("createProgram", () => {
  describe("repository lock", () => {
    /** The lockfile both commands derive from the repository they are run in. */
    function repoLockPath(): string {
      return lockfilePath(repoRoot());
    }

    /** The holder record in the repository's lockfile, or `undefined` when no lock is held. */
    function readRepoLock(): unknown {
      const lockPath = repoLockPath();
      return existsSync(lockPath) ? JSON.parse(readFileSync(lockPath, "utf8")) : undefined;
    }

    /**
     * Take the repository lock on behalf of this very process.
     *
     * The current pid is the one holder a liveness probe can never mistake for
     * stale, so the lock reliably looks like another gymrat run still in flight.
     */
    function holdRepoLock(command: string): void {
      const lockPath = repoLockPath();
      mkdirSync(dirname(lockPath), { recursive: true });
      writeFileSync(
        lockPath,
        JSON.stringify({ pid: process.pid, command, at: "2026-01-01T00:00:00.000Z" }),
      );
    }

    /**
     * Stub the command's run function to run `duringRun` before it settles,
     * rejecting with `failure` when one is given, and hand back the stub.
     *
     * `duringRun` fires while the command is mid-flight, which is the only
     * moment the lock the command took is observable.
     */
    function arrangeRun<T>(
      setup: (failure?: Error) => Promise<(...args: never[]) => Promise<T>>,
      resultFactory: () => T,
    ): (duringRun: () => void, failure?: Error) => Promise<MockInstance> {
      return async (duringRun, failure) => {
        const mock = await setup(failure);
        return vi.mocked(mock).mockImplementation(() => {
          duringRun();
          return failure ? Promise.reject(failure) : Promise.resolve(resultFactory());
        });
      };
    }

    const LOCKING_COMMANDS = [
      {
        command: "compare",
        argv: compareArgv("main", "branch"),
        arrange: arrangeRun(
          async (failure) => (await setupMocks(failure)).compareMock,
          createComparisonResult,
        ),
      },
      {
        command: "measure",
        argv: measureArgv("main"),
        arrange: arrangeRun(
          async (failure) => (await setupMeasureMocks(failure)).measureMock,
          createMeasurementResult,
        ),
      },
    ];

    afterEach(() => {
      rmSync(repoLockPath(), { force: true });
    });

    it.each(LOCKING_COMMANDS)(
      "$command exits 2 without benchmarking while another live process holds the lock",
      async ({ argv, arrange }) => {
        const runMock = await arrange(() => {});
        holdRepoLock("measure");
        const program = createRunnableProgram({ exitOverride: "all" });
        stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        mockProcessExit();

        const parsing = program.parseAsync(argv);

        // Nothing was run, and the diagnostic names the holder
        await expect(parsing).rejects.toHaveProperty("exitCode", 2);
        const stderrText = stderrWrites(stderrSpy).map(String).join("");
        expect.soft(runMock).not.toHaveBeenCalled();
        expect.soft(stderrText).toContain(`PID ${String(process.pid)}`);
        expect(stderrText).toMatch(/another gymrat run/i);
      },
    );

    it.each(LOCKING_COMMANDS)(
      "$command holds the lock for the length of a successful run and releases it",
      async ({ command, argv, arrange }) => {
        let heldDuringRun: unknown;
        await arrange(() => {
          heldDuringRun = readRepoLock();
        });
        const program = createRunnableProgram();
        stubWrite(process.stdout);

        await program.parseAsync(argv);

        expect.soft(heldDuringRun).toStrictEqual({
          pid: process.pid,
          command,
          at: expect.stringMatching(ISO_PATTERN),
        });
        expect(readRepoLock()).toBeUndefined();
      },
    );

    it.each(LOCKING_COMMANDS)(
      "$command has released the lock by the time it writes its report",
      async ({ argv, arrange }) => {
        // Stdout belongs to whoever is reading it, and a slow reader can
        // hold the drain open indefinitely. The repository must not be held with it.
        await arrange(() => {});
        const program = createRunnableProgram();
        let heldAtReport: unknown;
        let reportWritten = false;
        stubWrite(process.stdout, () => {
          if (!reportWritten) {
            reportWritten = true;
            heldAtReport = readRepoLock();
          }
        });

        await program.parseAsync(argv);

        expect.soft(reportWritten).toBe(true);
        expect(heldAtReport).toBeUndefined();
      },
    );

    it.each(LOCKING_COMMANDS)(
      "$command releases the lock when the run fails",
      async ({ command, argv, arrange }) => {
        let heldDuringRun: unknown;
        await arrange(() => {
          heldDuringRun = readRepoLock();
        }, new Error("benchmark crashed"));
        const program = createRunnableProgram({ exitOverride: "all" });
        stubWrite(process.stdout);
        stubWrite(process.stderr);
        mockProcessExit();

        const parsing = program.parseAsync(argv);

        await expect(parsing).rejects.toHaveProperty("exitCode", 2);
        expect.soft(heldDuringRun).toStrictEqual({
          pid: process.pid,
          command,
          at: expect.stringMatching(ISO_PATTERN),
        });
        expect(readRepoLock()).toBeUndefined();
      },
    );
  });
});

describe("the discard command", () => {
  let repo: ScratchRepo;
  const originalStdinIsTTY = process.stdin.isTTY;

  beforeEach(() => {
    repo = createScratchRepo();
    process.chdir(repo.dir);
    // Start a real session so requireOpenSession finds it, and add an
    // unsettled iteration so discardSession has something to discard.
    startSession(repo.dir, "main", resolvedConfigFixture());
    appendRecord(sessionJsonlPath(repo.dir), iterationRecord({ seq: 1 }));
  });

  afterEach(() => {
    process.stdin.isTTY = originalStdinIsTTY;
    rmSync(lockfilePath(repo.dir), { force: true });
    repo.cleanup();
  });

  it("documents --force in its help text", async () => {
    vi.stubEnv("FORCE_COLOR", undefined);

    const helpOutput = await captureHelp(["node", "cli.js", "discard", "--help"]);

    expect(helpOutput).toContain("-f, --force");
  });

  describe("when stdin is a TTY and --force is not passed", () => {
    it("prompts naming the experiment worktree path and proceeds when confirmed", async () => {
      process.stdin.isTTY = true;
      mockConfirmAction.mockResolvedValue(true);
      const program = createRunnableProgram();
      const readStdout = captureStdout();

      await program.parseAsync(["node", "cli.js", "discard"]);

      // The prompt was called with the worktree path, and the discard proceeded
      expect(mockConfirmAction).toHaveBeenCalledWith(
        expect.stringContaining(experimentWorktreeDir(repo.dir)),
        process.stdin,
      );
      expect(readStdout()).toContain("Discarded");
    });

    it("cancels with exit 1 and a stderr message when the user declines", async () => {
      process.stdin.isTTY = true;
      mockConfirmAction.mockResolvedValue(false);
      const program = createRunnableProgram({ exitOverride: "all" });
      captureStdout();
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(["node", "cli.js", "discard"]);

      await expect(parsing).rejects.toHaveProperty("exitCode", 1);
      expect(stderrWrites(stderrSpy).map(String).join("")).toContain("discard cancelled");
    });
  });

  describe("when --force is passed", () => {
    it("skips the prompt and proceeds", async () => {
      process.stdin.isTTY = true;
      const program = createRunnableProgram();
      const readStdout = captureStdout();

      await program.parseAsync(["node", "cli.js", "discard", "--force"]);

      expect(mockConfirmAction).not.toHaveBeenCalled();
      expect(readStdout()).toContain("Discarded");
    });

    it("accepts -f as a short alias", async () => {
      process.stdin.isTTY = true;
      const program = createRunnableProgram();
      const readStdout = captureStdout();

      await program.parseAsync(["node", "cli.js", "discard", "-f"]);

      expect(mockConfirmAction).not.toHaveBeenCalled();
      expect(readStdout()).toContain("Discarded");
    });
  });

  describe("when stdin is not a TTY", () => {
    it("skips the prompt and proceeds without confirming", async () => {
      // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- node leaves isTTY undefined on non-TTY streams
      process.stdin.isTTY = undefined as unknown as boolean;
      const program = createRunnableProgram();
      const readStdout = captureStdout();

      await program.parseAsync(["node", "cli.js", "discard"]);

      // Confirm was not called, but discard ran and reported success
      expect(mockConfirmAction).not.toHaveBeenCalled();
      expect(readStdout()).toContain("Discarded iteration");
    });
  });
});

describe("the finalize command", () => {
  let repo: ScratchRepo;
  // The command closes the session in the repository it runs in, so the run
  // happens in a throwaway one rather than in the checkout the suite lives in.
  beforeEach(() => {
    repo = createScratchRepo();
    process.chdir(repo.dir);
  });

  afterEach(() => {
    rmSync(lockfilePath(repo.dir), { force: true });
    repo.cleanup();
  });

  /** Open a session with one kept commit on it, and hand back its branch. */
  function sessionWithOneKeep(): string {
    const { session } = startSession(repo.dir, "main", resolvedConfigFixture());
    const worktree = session.worktrees.experiment;
    writeFileSync(join(worktree, "step.txt"), "cache the regex\n");
    git(["add", "-A"], worktree);
    git(["commit", "-m", "cache the regex"], worktree);
    appendRecord(sessionJsonlPath(repo.dir), iterationRecord({ seq: 1 }));
    appendRecord(
      sessionJsonlPath(repo.dir),
      committedKeep(1, { commit: git(["rev-parse", "HEAD"], worktree) }),
    );
    return session.branch;
  }

  /** Everything the scratch repository's session log holds. */
  function sessionLog(): SessionLogRecord[] {
    return readRecords(sessionJsonlPath(repo.dir));
  }

  it("documents both its flags and the branch it defaults to", async () => {
    vi.stubEnv("FORCE_COLOR", undefined);

    const helpOutput = await captureHelp(["node", "cli.js", "finalize", "--help"]);

    expect.soft(helpOutput).toContain("Usage: gymrat finalize");
    expect.soft(helpOutput).toContain("-m, --message");
    expect.soft(helpOutput).toContain("--branch");
    expect(helpOutput).toContain("-final");
  });

  it.each([
    {
      desc: "the branch the caller named",
      args: ["--branch", "perf/regex-cache"],
      named: "perf/regex-cache",
    },
    { desc: "the session branch's -final when the caller names none", args: [], named: undefined },
  ])("records and reports $desc", async ({ args, named }) => {
    const branch = sessionWithOneKeep();
    const finalBranch = named ?? `${branch}-final`;
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stdout = captureStdout();

    await program.parseAsync(["node", "cli.js", "finalize", ...args]);

    expect.soft(sessionLog().at(-1)).toMatchObject({ type: "finalize", branch: finalBranch });
    expect(stdout()).toContain(finalBranch);
  });

  it("commits the message it was given", async () => {
    sessionWithOneKeep();
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    captureStdout();

    await program.parseAsync(["node", "cli.js", "finalize", "-m", "squash the tuning session"]);

    expect(sessionLog().at(-1)).toMatchObject({ message: "squash the tuning session" });
  });

  it("exits 2 with a start hint when the repository holds no session", async () => {
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    captureStdout();
    const stderrSpy = stubWrite(process.stderr);
    mockProcessExit();

    const parsing = program.parseAsync(["node", "cli.js", "finalize"]);

    await expect(parsing).rejects.toHaveProperty("exitCode", 2);
    expect(stderrWrites(stderrSpy).map(String).join("")).toContain("gymrat start");
  });

  it("exits 2 leaving the session open while another live process holds the lock", async () => {
    // This process's own pid is the one holder a liveness probe can never call stale
    sessionWithOneKeep();
    const openLog = sessionLog();
    mkdirSync(dirname(lockfilePath(repo.dir)), { recursive: true });
    writeFileSync(
      lockfilePath(repo.dir),
      JSON.stringify({ pid: process.pid, command: "iterate", at: "2026-01-01T00:00:00.000Z" }),
    );
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    captureStdout();
    const stderrSpy = stubWrite(process.stderr);
    mockProcessExit();

    const parsing = program.parseAsync(["node", "cli.js", "finalize"]);

    await expect(parsing).rejects.toHaveProperty("exitCode", 2);
    expect.soft(sessionLog()).toStrictEqual(openLog);
    expect(stderrWrites(stderrSpy).map(String).join("")).toMatch(/another gymrat run/i);
  });
});

describe("the start command", () => {
  let repo: ScratchRepo;

  beforeEach(() => {
    repo = createScratchRepo();
    process.chdir(repo.dir);
  });

  afterEach(() => {
    rmSync(lockfilePath(repo.dir), { force: true });
    repo.cleanup();
  });

  describe("when runbook is configured", () => {
    const RUNBOOK_PATH = ".claude/skills/ecstatic-bench/SKILL.md";

    it.each([
      { desc: "fresh session", resumed: false },
      { desc: "resumed session", resumed: true },
    ])("includes a runbook row in the summary ($desc)", async ({ resumed }) => {
      if (resumed) {
        startSession(repo.dir, "main", resolvedConfigFixture());
      }
      await setupMocks(undefined, { runbook: RUNBOOK_PATH });
      const readStdout = captureStdout();
      const program = createRunnableProgram();

      await program.parseAsync(["node", "cli.js", "start", "main"]);

      expect(readStdout()).toContain(`runbook: ${RUNBOOK_PATH} — read it before your first edit`);
    });
  });

  describe("when runbook is not configured", () => {
    it("omits the runbook row from the summary", async () => {
      await setupMocks();
      const readStdout = captureStdout();
      const program = createRunnableProgram();

      await program.parseAsync(["node", "cli.js", "start", "main"]);

      expect(readStdout()).not.toContain("runbook");
    });
  });
});

describe("the loop commands, run from a subdirectory of the repository", () => {
  /** The resolver `start` and `iterate` settle a full run configuration through. */
  async function fullConfigResolver() {
    const { resolveConfig } = await import("../src/config.js");
    return vi.mocked(resolveConfig);
  }

  /** The resolver the loop commands that never bench settle their configuration through. */
  async function benchlessConfigResolver() {
    const { resolveBenchlessConfig } = await import("../src/config.js");
    return vi.mocked(resolveBenchlessConfig);
  }

  /**
   * Every loop command that settles configuration, with the resolver it reads it
   * through.
   *
   * `discard` and `finalize` read no configuration at all, so neither has a
   * lookup to place.
   */
  const CONFIG_READING_COMMANDS = [
    { command: "start", args: ["start", "main"], resolver: fullConfigResolver },
    { command: "iterate", args: ["iterate"], resolver: fullConfigResolver },
    { command: "keep", args: ["keep"], resolver: benchlessConfigResolver },
    { command: "status", args: ["status"], resolver: benchlessConfigResolver },
  ];

  /** The subdirectory of the repository each command is run from. */
  const NESTED_DIR = join("packages", "core");

  let repo: ScratchRepo;

  beforeEach(async () => {
    repo = createScratchRepo();
    mkdirSync(join(repo.dir, NESTED_DIR), { recursive: true });
    process.chdir(join(repo.dir, NESTED_DIR));
    const { resolveConfig, resolveBenchlessConfig } = await import("../src/config.js");
    vi.mocked(resolveConfig).mockReturnValue(resolvedConfigFixture());
    vi.mocked(resolveBenchlessConfig).mockReturnValue(resolvedConfigFixture());
  });

  afterEach(() => {
    rmSync(lockfilePath(repo.dir), { force: true });
    repo.cleanup();
  });

  it.each(CONFIG_READING_COMMANDS)(
    "$command looks the implicit config up at the repository root",
    async ({ args, resolver }) => {
      const resolverMock = await resolver();
      const program = createRunnableProgram({ exitOverride: "all", silent: true });
      captureStdout({ silenceStderr: true });
      mockProcessExit();

      // Whether the command then finds the session it needs is beside the
      // point; where it looked the configuration up is what is under test.
      await program.parseAsync(["node", "cli.js", ...args]).catch(() => undefined);

      expect(resolverMock).toHaveBeenCalledWith(expect.anything(), repo.dir);
    },
  );
});
