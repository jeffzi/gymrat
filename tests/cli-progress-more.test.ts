/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */

import { afterEach, describe, expect, it, vi } from "vitest";

import type { CompareOptions, ProgressStep } from "../src/compare.js";
import type { ResolvedConfig } from "../src/config.js";
import type { ComparisonResult } from "../src/report/types.js";
import { createRunnableProgram, mockProcessExit, stubWrite } from "./fixtures/cli-harness.js";
import {
  createCandidate,
  createComparisonResult,
  metricMeta,
} from "./fixtures/comparison-result.js";

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
/** Prepends the `["node", "cli.js", "compare"]` prefix Commander expects. */
function compareArgv(...args: string[]): string[] {
  return ["node", "cli.js", "compare", ...args];
}

/** Prepends the `["node", "cli.js", "measure"]` prefix Commander expects. */
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
  describe("compare command", () => {
    describe("progress feedback", () => {
      const originalStderrIsTTY = process.stderr.isTTY;

      afterEach(() => {
        process.stderr.isTTY = originalStderrIsTTY;
      });

      const PREPARE_BASELINE_STEP: ProgressStep = { kind: "prepare", label: "baseline" };
      const PREPARE_BASELINE_LINE = "prepare · baseline";

      function findStderrWrite(stderrSpy: ReturnType<typeof vi.spyOn>, substring: string): unknown {
        return stderrWrites(stderrSpy).find((w) => typeof w === "string" && w.includes(substring));
      }

      async function setupProgressMocks(
        steps: ProgressStep[],
        mockReturn?: ComparisonResult | Error,
        duringRun?: (opts: CompareOptions) => void,
      ): Promise<{ stderrSpy: ReturnType<typeof vi.spyOn> }> {
        const { compareMock } = await setupMocks(mockReturn);

        // Progress steps fire before compare settles, mirroring the real onProgress timing.
        // `duringRun` then runs with the progress line already on screen, which is
        // where a real adapter warning would land.
        vi.mocked(compareMock).mockImplementation((opts: CompareOptions) => {
          for (const step of steps) {
            opts.onProgress?.(step);
          }
          duringRun?.(opts);
          return mockReturn instanceof Error
            ? Promise.reject(mockReturn)
            : Promise.resolve(mockReturn ?? createComparisonResult());
        });

        const stderrSpy = stubWrite(process.stderr);
        // Suppress stdout report output so tests focus on stderr
        stubWrite(process.stdout);

        return { stderrSpy };
      }

      /** Simulates an interactive stderr with color allowed (TTY, NO_COLOR unset). */
      function useColorTty(): void {
        process.stderr.isTTY = true;
        vi.stubEnv("NO_COLOR", undefined);
      }

      /** Simulates an interactive stderr with color vetoed via NO_COLOR. */
      function useNoColorTty(): void {
        process.stderr.isTTY = true;
        vi.stubEnv("FORCE_COLOR", undefined);
        vi.stubEnv("NO_COLOR", "1");
      }

      describe("when the progress line is wider than the terminal", () => {
        const NARROW_COLUMNS = 30;
        const LONG_LABEL = "baseline-with-a-very-long-target-name-that-never-fits";
        const LONG_SAMPLE_STEP: ProgressStep = {
          kind: "sample",
          index: 1,
          total: 5,
          label: LONG_LABEL,
        };
        const WARNING = "Failed to parse METRIC line: METRIC foo=bar";

        const originalColumns = Object.getOwnPropertyDescriptor(process.stderr, "columns");

        afterEach(() => {
          if (originalColumns) {
            Object.defineProperty(process.stderr, "columns", originalColumns);
          } else {
            Reflect.deleteProperty(process.stderr, "columns");
          }
        });

        /** Pins the terminal width; `process.stderr.columns` is absent on a non-TTY stream. */
        function useTerminalWidth(columns: number): void {
          Object.defineProperty(process.stderr, "columns", {
            value: columns,
            configurable: true,
            writable: true,
          });
        }

        /**
         * Drop the `\r\x1b[K` overwrite prefix so only the displayed characters
         * remain — the escape sequence occupies no columns.
         */
        function displayedLine(write: unknown): string {
          return String(write).replace(/^\r\x1b\[K/, "");
        }

        /** Step counter at the front, a middle ellipsis, then the tail of the full line. */
        const MIDDLE_ELLIPSIS_SHAPE = /^sample 1\/5 .*…/u;

        it("truncates the progress line to the terminal width", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          useTerminalWidth(NARROW_COLUMNS);
          const { stderrSpy } = await setupProgressMocks([LONG_SAMPLE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          // The line fits on one row, so \r\x1b[K erases all of it
          const displayed = displayedLine(findStderrWrite(stderrSpy, "sample 1/5"));
          expect.soft(displayed.length).toBeLessThanOrEqual(NARROW_COLUMNS);
          expect(displayed).toMatch(MIDDLE_ELLIPSIS_SHAPE);
        });

        it("truncates the progress line redrawn after a warning", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          useTerminalWidth(NARROW_COLUMNS);
          const { stderrSpy } = await setupProgressMocks([LONG_SAMPLE_STEP], undefined, (opts) => {
            opts.warn?.(WARNING);
          });

          await program.parseAsync(compareArgv("main", "branch"));

          // The redraw after the warning is truncated like the first write
          const writes = stderrWrites(stderrSpy).map((write) => String(write));
          const warnIndex = writes.findIndex((write) => write.includes(WARNING));
          const redraw = writes.slice(warnIndex + 1).find((write) => write.includes("sample 1/5"));
          const displayed = displayedLine(redraw);
          expect.soft(displayed.length).toBeLessThanOrEqual(NARROW_COLUMNS);
          expect(displayed).toMatch(MIDDLE_ELLIPSIS_SHAPE);
        });
      });

      describe("when stderr is not a TTY", () => {
        it("does not construct a spinner", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          expect(mockYoctoSpinner).not.toHaveBeenCalled();
        });

        it("writes newline-terminated lines without ANSI escapes", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Non-TTY lines end with \n and contain no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, PREPARE_BASELINE_LINE);
          expect(progressWrite).toBeDefined();
          expect(progressWrite).not.toContain("\x1b[");
          expect(progressWrite).toMatch(/\n$/);
        });

        it("includes a plain ETA segment in newline-terminated output when estimate exists", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          mockEtaRecord.mockReturnValue(130_000);
          mockFormatEta.mockReturnValue("~2m 10s left");
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Newline-terminated, plain text with ETA
          const progressWrite = findStderrWrite(stderrSpy, "~2m 10s left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · ~2m 10s left");
          expect(progressWrite).toMatch(/\n$/);
          expect(progressWrite).not.toContain("\x1b[");
        });

        it("renders a plain placeholder when ETA is not yet available for a sample step", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          mockEtaRecord.mockReturnValue(undefined);
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Placeholder as plain text, no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, "estimating time left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · estimating time left…");
          expect(progressWrite).not.toContain("\x1b[");
        });
      });

      describe("spinner cleared before output", () => {
        it("stops the spinner before the report prints to stdout", async () => {
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);
          const stdoutSpy = stubWrite(process.stdout);

          await program.parseAsync(compareArgv("main", "branch"));

          expect(mockSpinnerInstance.stop).toHaveBeenCalled();
          const stopOrder = mockSpinnerInstance.stop.mock.invocationCallOrder[0];
          const reportOrder = stdoutSpy.mock.invocationCallOrder[0];
          if (stopOrder === undefined || reportOrder === undefined) {
            throw new Error("both stop and report must have been called");
          }
          expect(stopOrder).toBeLessThan(reportOrder);
        });

        it("stops the spinner before a formatted error prints to stderr", async () => {
          const program = createRunnableProgram({ exitOverride: "all" });
          useColorTty();
          const { stderrSpy } = await setupProgressMocks(
            [PREPARE_BASELINE_STEP],
            new Error("benchmark crashed"),
          );
          mockProcessExit();

          await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
            "exitCode",
            2,
          );

          // Spinner stopped before error message written
          expect(mockSpinnerInstance.stop).toHaveBeenCalled();
          const stopOrder = mockSpinnerInstance.stop.mock.invocationCallOrder[0];
          expect(stopOrder).toBeDefined();
          const errorWrite = stderrSpy.mock.invocationCallOrder.find(
            (_order: number, i: number) => {
              const arg = stderrSpy.mock.calls[i]?.[0];
              return typeof arg === "string" && arg.includes("benchmark crashed");
            },
          );
          expect(errorWrite).toBeDefined();
          expect(stopOrder).toBeLessThan(errorWrite as number);
        });

        it("stops the spinner before the --fail-on gate-failure exit", async () => {
          const program = createRunnableProgram({ exitOverride: "all" });
          useColorTty();

          const regressedResult = createComparisonResult({
            candidates: [createCandidate()],
            metrics: {
              "decode/time": {
                baselineMedian: 100,
                baselineSpread: 1,
                candidates: [
                  {
                    median: 108,
                    spread: 1,
                    verdict: {
                      verdict: "regressed",
                      method: "signed-rank",
                      delta: 8,
                      n: 10,
                      p: 0.01,
                      noisePct: 2.5,
                      noiseAbs: 2.5,
                    },
                  },
                ],
                meta: metricMeta("decode/time"),
              },
            },
          });
          await setupProgressMocks(
            [{ kind: "sample", index: 1, total: 1, label: "baseline" }],
            regressedResult,
          );
          mockProcessExit();

          await expect(
            program.parseAsync(compareArgv("main", "branch", "--fail-on", "regressed")),
          ).rejects.toHaveProperty("exitCode", 1);

          expect(mockSpinnerInstance.stop).toHaveBeenCalled();
        });
      });

      describe("ETA countdown interpolation", () => {
        afterEach(() => {
          vi.useRealTimers();
        });

        /**
         * Set up a compare mock that fires the given progress steps immediately,
         * then returns a promise the caller controls via `resolveCompare`. This
         * lets tests advance fake timers between `emit()` and `stop()`.
         *
         * `delayedStep`, if given, fires as a second `onProgress` emit after
         * `delayMs` — used to test countdown resets mid-flight.
         */
        async function setupCountdownTest(
          steps: ProgressStep[],
          delayedStep?: { step: ProgressStep; delayMs: number },
        ): Promise<{
          stderrSpy: ReturnType<typeof vi.spyOn>;
          resolveCompare: () => void;
        }> {
          const { compareMock } = await setupMocks();
          let settle!: () => void;

          vi.mocked(compareMock).mockImplementation((opts: CompareOptions) => {
            for (const step of steps) {
              opts.onProgress?.(step);
            }
            if (delayedStep) {
              setTimeout(() => {
                opts.onProgress?.(delayedStep.step);
              }, delayedStep.delayMs);
            }
            return new Promise<ComparisonResult>((r) => {
              settle = () => {
                r(createComparisonResult());
              };
            });
          });

          const stderrSpy = stubWrite(process.stderr);
          stubWrite(process.stdout);

          // Wrapper defers the read: `settle` is assigned when `compareMock` runs
          // inside `parseAsync`, which happens after this function returns.
          return {
            stderrSpy,
            resolveCompare: () => {
              settle();
            },
          };
        }

        /** Resolves the compare promise, flushes the resulting microtask, and awaits `parseAsync`. */
        async function settleCompare(
          resolveCompare: () => void,
          parsePromise: Promise<unknown>,
        ): Promise<void> {
          resolveCompare();
          await vi.advanceTimersByTimeAsync(0);
          await parsePromise;
        }

        /** Stubs `formatEta` to render `~<seconds>s left`, floor-rounded, for countdown assertions. */
        function stubFormatEtaInSeconds(): void {
          mockFormatEta.mockImplementation((ms: number) => `~${Math.floor(ms / 1000)}s left`);
        }

        it("ticks down spinner text every ~1s using elapsed wall-clock time", async () => {
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest([
            { kind: "sample", index: 2, total: 5, label: "baseline" },
          ]);

          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          await vi.advanceTimersByTimeAsync(1000);

          // Spinner text reflects the decremented ETA
          expect(mockSpinnerInstance.text).toContain("~129s left");

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });

        it("resets the countdown to the fresh ETA when a new emit arrives", async () => {
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValueOnce(10_000).mockReturnValue(60_000);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest(
            [{ kind: "sample", index: 2, total: 5, label: "baseline" }],
            { step: { kind: "sample", index: 3, total: 5, label: "baseline" }, delayMs: 5000 },
          );

          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          // Advance past the first ETA and trigger the second emit at t=5000
          await vi.advanceTimersByTimeAsync(5000);
          mockFormatEta.mockClear();
          // Advance 1s after the second emit — countdown should use new ETA (60_000)
          await vi.advanceTimersByTimeAsync(1000);

          // The countdown is from the new ETA (60_000 - 1000 = 59_000),
          // not the old depleted one (10_000 - 6000 = clamped to 0)
          expect(mockSpinnerInstance.text).toContain("~59s left");

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });

        it("clears the countdown interval when stop is called", async () => {
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest([
            { kind: "sample", index: 2, total: 5, label: "baseline" },
          ]);

          // Start, advance to prove countdown is active, then stop
          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          await vi.advanceTimersByTimeAsync(1000);
          expect(mockSpinnerInstance.text).toContain("~129s left");

          // Resolve compare → stop() is called
          await settleCompare(resolveCompare, parsePromise);

          // Assert — after stop, further ticks no longer update the spinner text
          const textAfterStop = mockSpinnerInstance.text;
          await vi.advanceTimersByTimeAsync(5000);
          expect(mockSpinnerInstance.text).toBe(textAfterStop);
        });

        it("clears a running countdown when a subsequent emit yields no ETA", async () => {
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValueOnce(130_000).mockReturnValue(undefined);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest(
            [{ kind: "sample", index: 2, total: 5, label: "baseline" }],
            { step: { kind: "sample", index: 3, total: 5, label: "baseline" }, delayMs: 2000 },
          );

          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          // Advance 2s — the second emit fires with undefined ETA, clearing the interval
          await vi.advanceTimersByTimeAsync(2000);
          const textAfterUndefinedEta = mockSpinnerInstance.text;

          // Advance 2s more — if interval were still active, spinner text would keep changing
          await vi.advanceTimersByTimeAsync(2000);

          // No further updates after the interval was cleared
          expect(mockSpinnerInstance.text).toBe(textAfterUndefinedEta);

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });

        it("clamps the countdown ETA to zero instead of going negative", async () => {
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(1500);
          mockFormatEta.mockImplementation((ms: number) => `~${Math.ceil(ms / 1000)}s left`);

          const { resolveCompare } = await setupCountdownTest([
            { kind: "sample", index: 2, total: 5, label: "baseline" },
          ]);

          // Advance past the ETA
          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          await vi.advanceTimersByTimeAsync(5000);

          // Every formatEta call used a non-negative value
          for (const [ms] of mockFormatEta.mock.calls) {
            expect(ms).toBeGreaterThanOrEqual(0);
          }
          // The clamped value (0) was reached
          expect(mockSpinnerInstance.text).toContain("~0s left");

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });
      });
    });
  });
});
