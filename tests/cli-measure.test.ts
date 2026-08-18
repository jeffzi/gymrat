/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */
import { Command } from "commander";
import { afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from "vitest";

import { createProgram } from "../src/cli.js";
import type { ProgressStep } from "../src/compare.js";
import type { ResolvedConfig } from "../src/config.js";
import type { MeasureOptions } from "../src/measure.js";
import { renderMeasureJson } from "../src/report/json.js";
import { renderMeasureReport } from "../src/report/text.js";
import type { MeasurementResult } from "../src/report/types.js";
import { sessionJsonlPath } from "../src/session/paths.js";
import type { SessionLogRecord } from "../src/session/records.js";
import { appendRecord, readRecords } from "../src/session/store.js";
import {
  createRunnableProgram,
  mockProcessExit,
  stubWrite,
  writtenChunks,
} from "./fixtures/cli-harness.js";
import { ANSI_RE, ISO_PATTERN } from "./fixtures/constants.js";
import { createMeasurementResult, twoKindMeasurement } from "./fixtures/measurement-result.js";
import { createScratchRepo, type ScratchRepo } from "./fixtures/scratch-repo.js";
import { finalizeRecord, sessionRecord } from "./fixtures/session-records.js";

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

/** The settled run configuration both commands read, with `overrides` applied. */
const CONFIG_FLAG_TABLE = [
  { flag: "--bench", value: "my-bench", expected: { bench: "my-bench" } },
  { flag: "--prepare", value: "setup.sh", expected: { prepare: "setup.sh" } },
  { flag: "--adapter", value: "mitata", expected: { adapter: "mitata" } },
  { flag: "--samples", value: "100", expected: { samples: 100 } },
  { flag: "--timeout", value: "5000", expected: { timeout: 5000 } },
  { flag: "--config", value: "gymrat.json", expected: { config: "gymrat.json" } },
  { flag: "-b", value: "my-bench", expected: { bench: "my-bench" } },
  { flag: "-p", value: "setup.sh", expected: { prepare: "setup.sh" } },
  { flag: "-a", value: "mitata", expected: { adapter: "mitata" } },
  { flag: "-s", value: "100", expected: { samples: 100 } },
  { flag: "-t", value: "5000", expected: { timeout: 5000 } },
  { flag: "-c", value: "gymrat.json", expected: { config: "gymrat.json" } },
];

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

/** Stubs `resolveConfig` and `measure` for single-target command tests. */
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

/** Prepends the `["node", "cli.js", "measure"]` prefix Commander expects. */
function measureArgv(...args: string[]): string[] {
  return ["node", "cli.js", "measure", ...args];
}

function stderrWrites(stderrSpy: ReturnType<typeof vi.spyOn>): unknown[] {
  return stderrSpy.mock.calls.map((c: unknown[]) => c[0]);
}

/**
 * Run `measure <args>`, mocking `measure()` to resolve with `result`, and hand
 * back the stdout write spy.
 *
 * The measure counterpart of `runCompareCapturingStdout`, with the same division
 * of labour: callers own any TTY or env state, this owns program creation, mock
 * setup and the write spy.
 */
async function runMeasureCapturingStdout(
  result: MeasurementResult,
  ...args: string[]
): Promise<MockInstance<typeof process.stdout.write>> {
  const program = createRunnableProgram();
  await setupMeasureMocks(result);
  const writeSpy = stubWrite(process.stdout);
  await program.parseAsync(measureArgv(...args));
  return writeSpy;
}

/**
 * Set up a runnable `measure` program with `resolveConfig` and `measure` mocked
 * and the stdout writer stubbed out — the common preamble every measure-command
 * test starts from before parsing its own argv and asserting on the mocks.
 */
async function startMeasureRun(configOverrides: Partial<ResolvedConfig> = {}) {
  const program = createRunnableProgram();
  const { measureMock, resolveConfigMock } = await setupMeasureMocks(undefined, configOverrides);
  stubWrite(process.stdout);
  return { program, measureMock, resolveConfigMock };
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  mockSpinnerInstance.text = "";
  mockSpinnerInstance.isSpinning = false;
});

/**
 * The color rules every report-emitting command follows, checked against `run`.
 *
 * `compare` and `measure` decide color the same way — the report is styled when
 * stdout is a terminal, plain when it is redirected, and `--no-color` sets
 * `NO_COLOR` for every renderer downstream — so both drive these cases through
 * their own runner rather than each restating them.
 */
function describeColorDecision(
  run: (...flags: string[]) => Promise<MockInstance<typeof process.stdout.write>>,
): void {
  describe("when deciding whether to color the report", () => {
    const originalIsTTY = process.stdout.isTTY;

    afterEach(() => {
      process.stdout.isTTY = originalIsTTY;
    });

    it("includes ANSI escapes when stdout is a terminal", async () => {
      process.stdout.isTTY = true;
      vi.stubEnv("NO_COLOR", undefined);

      const writeSpy = await run();

      const firstCall = writeSpy.mock.calls.at(0);
      expect(firstCall?.[0]).toMatch(ANSI_RE);
    });

    it("omits ANSI escapes when stdout is redirected", async () => {
      process.stdout.isTTY = false;
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", undefined);

      const writeSpy = await run();

      const firstCall = writeSpy.mock.calls.at(0);
      expect(firstCall?.[0]).not.toMatch(ANSI_RE);
    });

    it("sets process.env.NO_COLOR when --no-color is passed", async () => {
      vi.stubEnv("NO_COLOR", undefined);

      await run("--no-color");

      expect(process.env.NO_COLOR).toBe("1");
    });
  });
}

describe("createProgram", () => {
  describe("measure command", () => {
    /** The flags both commands share; the verdict-only ones belong to compare alone. */
    const SHARED_FLAGS = [
      "--bench",
      "--prepare",
      "--adapter",
      "--samples",
      "--timeout",
      "--config",
      "--no-color",
      "--format",
    ];

    /**
     * Extract the long flags a subcommand declares, sorted, minus `--help` and
     * the flags one command carries alone. Uses Commander's programmatic API
     * instead of parsing rendered help text.
     */
    function declaredOptions(command: Command): string[] {
      const commandSpecific = new Set(["--help", "--verbose", "--fail-on", "--record"]);
      return command.options
        .map((o) => o.long)
        .filter((flag): flag is string => flag !== undefined && !commandSpecific.has(flag))
        .toSorted();
    }

    describe("on successful measurement", () => {
      it("renders the measurement measure returned and writes it to stdout", async () => {
        const result = twoKindMeasurement();

        const writeSpy = await runMeasureCapturingStdout(result, "main");

        expect(writtenChunks(writeSpy)).toStrictEqual([`${renderMeasureReport(result)}\n`]);
      });

      it("routes to renderMeasureJson for --format json", async () => {
        const result = createMeasurementResult();

        const writeSpy = await runMeasureCapturingStdout(result, "main", "--format", "json");

        expect(vi.mocked(renderMeasureJson)).toHaveBeenCalledWith(result);
        expect(writtenChunks(writeSpy)).toStrictEqual(['{"measurement": true}\n']);
      });
    });

    describe("when the target is given", () => {
      it.each([
        { form: "a bare ref", positional: "main", expected: { target: "main", label: undefined } },
        {
          form: "a label=ref pair",
          positional: "build=main",
          expected: { target: "main", label: "build" },
        },
      ])("passes $form through to measure", async ({ positional, expected }) => {
        const { program, measureMock } = await startMeasureRun();

        await program.parseAsync(measureArgv(positional));

        expect(measureMock).toHaveBeenCalledWith(expect.objectContaining({ target: expected }));
      });
    });

    describe("when the target is omitted", () => {
      it("benches the current directory in place", async () => {
        const { program, measureMock } = await startMeasureRun();

        await program.parseAsync(measureArgv());

        expect(measureMock).toHaveBeenCalledWith(
          expect.objectContaining({ target: { target: "." } }),
        );
      });
    });

    describe("when flags provided", () => {
      it.each(CONFIG_FLAG_TABLE)(
        "passes $flag through to resolveConfig",
        async ({ flag, value, expected }) => {
          const { program, resolveConfigMock } = await startMeasureRun();

          await program.parseAsync(measureArgv("main", flag, value));

          expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining(expected));
        },
      );

      it("hands the settled run configuration to measure", async () => {
        // Whatever resolveConfig settled on is what the run must use,
        // whichever of flag, config file or default supplied each value
        const configMetrics = {
          "decode/time": { direction: "higher" as const, gating: false, exact: true },
        };
        const configKinds = { memory: { gating: false } };
        const { program, measureMock } = await startMeasureRun({
          bench: "run-bench",
          prepare: "setup.sh",
          adapter: "mitata",
          samples: 7,
          timeoutSeconds: 42,
          metrics: configMetrics,
          kinds: configKinds,
        });

        await program.parseAsync(measureArgv("main"));

        expect(measureMock).toHaveBeenCalledWith(
          expect.objectContaining({
            bench: "run-bench",
            prepare: "setup.sh",
            adapter: "mitata",
            samples: 7,
            timeoutSeconds: 42,
            configMetrics,
            configKinds,
          }),
        );
      });
    });

    describe("recording", () => {
      /** The session header a log must open with for a run to have somewhere to record. */
      const SESSION_HEADER = sessionRecord({
        worktrees: { experiment: "/repo/.gymrat/experiment", baseline: "/repo/.gymrat/baseline" },
        config: {
          bench: "bench.sh",
          adapter: "metric-lines",
          samples: 1,
          timeoutSeconds: 300,
          primary: "geomean",
        },
      });

      let repo: ScratchRepo;
      // The command records into the repository it runs in, so the run happens in
      // a throwaway one rather than in the checkout the suite itself lives in.
      beforeEach(() => {
        repo = createScratchRepo();
        process.chdir(repo.dir);
      });

      afterEach(() => {
        repo.cleanup();
      });

      /** Open a session in the scratch repository. */
      function openSession(): void {
        appendRecord(sessionJsonlPath(repo.dir), SESSION_HEADER);
      }

      /** Everything the scratch repository's session log holds. */
      function sessionLog(): SessionLogRecord[] {
        return readRecords(sessionJsonlPath(repo.dir));
      }

      it.each([
        { form: "a bare ref", positional: "main", label: "main" },
        { form: "a label=ref pair", positional: "build=main", label: "build" },
      ])(
        "appends a baseline record carrying $form's label and the run's raw samples",
        async ({ positional, label }) => {
          openSession();
          const rounds = [{ latency: 41 }, { latency: 43 }];
          const program = createRunnableProgram();
          await setupMeasureMocks(createMeasurementResult({ label, rounds }));
          stubWrite(process.stdout);

          await program.parseAsync(measureArgv(positional, "--record"));

          expect(sessionLog().at(-1)).toStrictEqual({
            type: "baseline",
            at: expect.stringMatching(ISO_PATTERN),
            label,
            samples: rounds,
          });
        },
      );

      it("prints the usual report plus a note naming the session it recorded to", async () => {
        openSession();
        const result = createMeasurementResult({ rounds: [{ latency: 42 }] });
        const program = createRunnableProgram();
        await setupMeasureMocks(result);
        const writeSpy = stubWrite(process.stdout);

        await program.parseAsync(measureArgv("main", "--record"));

        const stdout = writeSpy.mock.calls.map((call) => String(call[0])).join("");
        expect.soft(stdout).toContain(renderMeasureReport(result));
        expect(stdout).toMatch(/recorded to session/i);
      });

      it("exits 2 with a start hint without measuring when the repository holds no session", async () => {
        // The check comes first: discovering there is nowhere to write
        // after ten minutes of sampling would throw the whole run away.
        const { measureMock } = await setupMeasureMocks();
        const program = createRunnableProgram({ exitOverride: "all" });
        stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        mockProcessExit();

        const parsing = program.parseAsync(measureArgv("main", "--record"));

        await expect(parsing).rejects.toHaveProperty("exitCode", 2);
        expect.soft(measureMock).not.toHaveBeenCalled();
        expect(stderrWrites(stderrSpy).map(String).join("")).toContain("gymrat start");
      });

      it("exits 2 with a start hint without measuring when the session was finalized", async () => {
        // A closed session is nowhere to record to either
        openSession();
        appendRecord(sessionJsonlPath(repo.dir), finalizeRecord());
        const { measureMock } = await setupMeasureMocks();
        const program = createRunnableProgram({ exitOverride: "all" });
        stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        mockProcessExit();

        const parsing = program.parseAsync(measureArgv("main", "--record"));

        await expect(parsing).rejects.toHaveProperty("exitCode", 2);
        expect.soft(measureMock).not.toHaveBeenCalled();
        expect(stderrWrites(stderrSpy).map(String).join("")).toContain("gymrat start");
      });

      it("leaves an open session untouched when --record is left out", async () => {
        // Recording is opt-in, so a plain run writes no history
        openSession();
        const program = createRunnableProgram();
        await setupMeasureMocks(createMeasurementResult({ rounds: [{ latency: 42 }] }));
        stubWrite(process.stdout);

        await program.parseAsync(measureArgv("main"));

        expect(sessionLog()).toStrictEqual([SESSION_HEADER]);
      });
    });

    describe("when a flag the command does not carry is passed", () => {
      it.each([
        { flag: "--fail-on", args: ["--fail-on", "regressed"] },
        { flag: "--verbose", args: ["--verbose"] },
        { flag: "--bogus", args: ["--bogus"] },
      ])("rejects $flag with a usage error naming the unknown option", async ({ flag, args }) => {
        const program = createRunnableProgram({ exitOverride: "all" });

        await expect(program.parseAsync(measureArgv("main", ...args))).rejects.toThrow(
          new RegExp(`unknown option '${flag}'`),
        );
      });

      it("exits 2 for Commander usage errors", async () => {
        // The production exitOverride (which sets exit code 2) must
        // survive here rather than being replaced by the test helper's plain one.
        // stderr is collected rather than discarded so the assertion can tell the
        // command's own usage error apart from one raised before it was reached.
        const program = createProgram();
        let usageError = "";
        for (const command of [program, ...program.commands]) {
          command.configureOutput({
            writeErr: (str) => {
              usageError += str;
            },
          });
        }
        mockProcessExit();

        // Act & Assert
        await expect(program.parseAsync(measureArgv("main", "--bogus"))).rejects.toHaveProperty(
          "exitCode",
          2,
        );
        expect(usageError).toContain("unknown option '--bogus'");
      });
    });

    describeColorDecision((...flags) =>
      runMeasureCapturingStdout(twoKindMeasurement(), "main", ...flags),
    );

    describe("progress feedback", () => {
      const originalStderrIsTTY = process.stderr.isTTY;

      afterEach(() => {
        process.stderr.isTTY = originalStderrIsTTY;
      });

      const PREPARE_STEP: ProgressStep = { kind: "prepare", label: "main" };
      const SAMPLE_STEP: ProgressStep = { kind: "sample", index: 1, total: 10, label: "main" };

      /**
       * Run `measure main` with `measure()` firing `steps` before it settles,
       * and hand back the stderr write spy.
       *
       * The steps fire inside the mock rather than after it, mirroring the real
       * `onProgress` timing: they arrive while the run is still in flight.
       */
      async function runWithProgress(
        steps: readonly ProgressStep[],
      ): Promise<ReturnType<typeof vi.spyOn>> {
        const { program, measureMock } = await startMeasureRun();
        vi.mocked(measureMock).mockImplementation((options: MeasureOptions) => {
          for (const step of steps) {
            options.onProgress?.(step);
          }
          return Promise.resolve(createMeasurementResult());
        });
        const stderrSpy = stubWrite(process.stderr);

        await program.parseAsync(measureArgv("main"));

        return stderrSpy;
      }

      it.each([
        { desc: "prepare", step: PREPARE_STEP, line: "prepare · main" },
        { desc: "sample", step: SAMPLE_STEP, line: "sample 1/10 · main" },
      ])("streams the $desc step to stderr", async ({ step, line }) => {
        process.stderr.isTTY = false;

        const stderrSpy = await runWithProgress([step]);

        expect(stderrWrites(stderrSpy)).toContainEqual(expect.stringContaining(line));
      });

      it("appends the ETA segment when the tracker yields an estimate", async () => {
        process.stderr.isTTY = false;
        mockEtaRecord.mockReturnValue(130_000);
        mockFormatEta.mockReturnValue("~2m 10s left");

        const stderrSpy = await runWithProgress([SAMPLE_STEP]);

        expect(stderrWrites(stderrSpy)).toContainEqual(expect.stringContaining("~2m 10s left"));
      });

      it("drives a stderr spinner when stderr is an interactive terminal", async () => {
        process.stderr.isTTY = true;
        vi.stubEnv("NO_COLOR", undefined);

        await runWithProgress([SAMPLE_STEP]);

        expect
          .soft(mockYoctoSpinner)
          .toHaveBeenCalledWith({ color: "yellow", stream: process.stderr });
        expect.soft(mockSpinnerInstance.start).toHaveBeenCalled();
        expect(mockSpinnerInstance.text).toContain("sample");
      });
    });

    describe("on measurement error", () => {
      it("exits 2 and writes the error to stderr", async () => {
        const program = createRunnableProgram({ exitOverride: "all" });
        await setupMeasureMocks(new Error("Measurement failed"));
        stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        mockProcessExit();

        await expect(program.parseAsync(measureArgv("main"))).rejects.toHaveProperty("exitCode", 2);

        expect(stderrWrites(stderrSpy)).toContainEqual(
          expect.stringContaining("Measurement failed"),
        );
      });
    });

    describe("when --help requested", () => {
      it("lists the command in the root help", async () => {
        vi.stubEnv("FORCE_COLOR", undefined);

        const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

        expect(helpOutput).toContain("measure");
      });

      it("documents every shared option", async () => {
        vi.stubEnv("FORCE_COLOR", undefined);

        const helpOutput = await captureHelp(measureArgv("--help"));

        expect.soft(helpOutput).toContain("Usage: gymrat measure");
        expect(SHARED_FLAGS.filter((flag) => !helpOutput.includes(flag))).toStrictEqual([]);
      });

      it("documents the recording flag under both its forms", async () => {
        const helpOutput = await captureHelp(measureArgv("--help"));

        expect(helpOutput).toContain("-r, --record");
      });

      it("offers the same shared options as compare", () => {
        const program = createProgram();
        const measure = program.commands.find((c) => c.name() === "measure");
        const compare = program.commands.find((c) => c.name() === "compare");
        if (measure === undefined || compare === undefined) {
          throw new Error("measure and compare commands must be registered");
        }

        // The parsed list is non-empty, so a broken parse can't pass vacuously
        expect.soft(declaredOptions(measure)).toStrictEqual(SHARED_FLAGS.toSorted());

        // One definition feeds both, so neither can drift from the other
        expect(declaredOptions(measure)).toStrictEqual(declaredOptions(compare));
      });

      it("ends with a measure-specific examples block", async () => {
        vi.stubEnv("FORCE_COLOR", undefined);

        const helpOutput = await captureHelp(measureArgv("--help"));

        expect.soft(helpOutput).toContain("Examples:");
        expect.soft(helpOutput).toContain("• gymrat measure --bench");
        expect.soft(helpOutput).toContain("• gymrat measure release=v2.0.0 --bench");
        expect(helpOutput).toContain('• gymrat measure --bench "npm run bench" --record');
      });
    });
  });
});
