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
import { createComparisonResult } from "./fixtures/comparison-result.js";

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

      // ANSI open/close sequences used in styled progress text assertions.
      const DIM_O = "\x1b[2m";
      const DIM_C = "\x1b[22m";
      const BOLD_O = "\x1b[1m";
      const BOLD_C = "\x1b[22m";
      const CYN_O = "\x1b[36m";
      const CYN_C = "\x1b[39m";

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

      describe("progress text format", () => {
        it("formats prepare steps as 'prepare · <label>' without trailing '· bench'", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          const writes = stderrWrites(stderrSpy);
          expect(writes).toContainEqual(expect.stringContaining(PREPARE_BASELINE_LINE));
          expect(writes).not.toContainEqual(expect.stringContaining("· bench"));
        });

        it("formats sample steps as 'sample <i>/<n> · <label>' without trailing '· bench'", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 1, total: 10, label: "baseline" },
          ]);

          await program.parseAsync(compareArgv("main", "branch"));

          const writes = stderrWrites(stderrSpy);
          expect(writes).toContainEqual(expect.stringContaining("sample 1/10 · baseline"));
          expect(writes).not.toContainEqual(expect.stringContaining("· bench"));
        });
      });

      describe("when stderr is a TTY and color is allowed", () => {
        it("constructs a yocto-spinner with yellow color and stderr stream", async () => {
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          expect(mockYoctoSpinner).toHaveBeenCalledWith({
            color: "yellow",
            stream: process.stderr,
          });
        });

        it("starts the spinner", async () => {
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([{ kind: "prepare", label: "baseline" }]);

          await program.parseAsync(compareArgv("main", "branch"));

          expect(mockSpinnerInstance.start).toHaveBeenCalled();
        });

        it("renders the prepare step word in default foreground (no dim) with cyan label", async () => {
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([{ kind: "prepare", label: "baseline" }]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Step word has no dim wrapping, label is cyan, nothing is yellow
          const text = mockSpinnerInstance.text;
          expect.soft(text).toContain("prepare");
          expect.soft(text).not.toContain(DIM_O + "prepare" + DIM_C);
          expect.soft(text).toContain(CYN_O + "baseline" + CYN_C);
          expect(text).not.toContain("\x1b[33m");
        });

        it("renders the sample step word in default foreground (no dim) with bold counter and cyan label", async () => {
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([{ kind: "sample", index: 1, total: 5, label: "baseline" }]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Step word has no dim wrapping, counter is bold, label is cyan
          const text = mockSpinnerInstance.text;
          expect.soft(text).toContain("sample");
          expect.soft(text).not.toContain(DIM_O + "sample" + DIM_C);
          expect.soft(text).toContain(BOLD_O + "1/5" + BOLD_C);
          expect(text).toContain(CYN_O + "baseline" + CYN_C);
        });

        it("appends a dim ETA segment when the tracker yields an estimate", async () => {
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          mockFormatEta.mockReturnValue("~2m 10s left");
          await setupProgressMocks([{ kind: "sample", index: 2, total: 5, label: "baseline" }]);

          await program.parseAsync(compareArgv("main", "branch"));

          // ETA segment is rendered dim
          const text = mockSpinnerInstance.text;
          expect(text).toContain(DIM_O + " · ~2m 10s left" + DIM_C);
        });

        it("omits both ETA and placeholder for prepare steps", async () => {
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(undefined);
          await setupProgressMocks([{ kind: "prepare", label: "baseline" }]);

          await program.parseAsync(compareArgv("main", "branch"));

          // No ETA segment and no placeholder
          const text = mockSpinnerInstance.text;
          expect.soft(text).not.toContain("left");
          expect(text).not.toContain("estimating");
        });

        it("renders a dim placeholder when ETA is not yet available for a sample step", async () => {
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(undefined);
          await setupProgressMocks([{ kind: "sample", index: 1, total: 5, label: "baseline" }]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Placeholder is dim-styled
          const text = mockSpinnerInstance.text;
          expect(text).toContain(DIM_O + " · estimating time left…" + DIM_C);
        });
      });

      describe("when stderr is a TTY but color is vetoed", () => {
        it("does not construct a spinner when NO_COLOR is set", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          expect(mockYoctoSpinner).not.toHaveBeenCalled();
        });

        it("does not construct a spinner when --no-color is passed", async () => {
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch", "--no-color"));

          expect(mockYoctoSpinner).not.toHaveBeenCalled();
        });

        it("falls back to \\r\\x1b[K overwrite with unstyled text", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks([PREPARE_BASELINE_STEP]);

          await program.parseAsync(compareArgv("main", "branch"));

          // TTY fallback uses \r\x1b[K prefix with plain text
          const progressWrite = findStderrWrite(stderrSpy, PREPARE_BASELINE_LINE);
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toMatch(/^\r\x1b\[K/);
        });

        it("clears the last progress line with \\r\\x1b[K before the report", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 1, total: 1, label: "baseline" },
          ]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Final stderr write clears the line
          const lastWrite = stderrWrites(stderrSpy)
            .map((w) => String(w))
            .at(-1);
          expect(lastWrite).toBe("\r\x1b[K");
        });

        it("clears progress before error text when compare throws", async () => {
          const program = createRunnableProgram({ exitOverride: "all" });
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks(
            [{ kind: "sample", index: 1, total: 5, label: "baseline" }],
            new Error("adapter parse failed"),
          );
          mockProcessExit();

          await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
            "exitCode",
            2,
          );

          // \r\x1b[K must appear before the error message in stderr writes
          const writes = stderrWrites(stderrSpy).map((w) => String(w));
          const clearIndex = writes.findIndex((w) => w === "\r\x1b[K");
          const errorIndex = writes.findIndex((w) => w.includes("adapter parse failed"));
          expect(clearIndex).toBeGreaterThanOrEqual(0);
          expect(errorIndex).toBeGreaterThanOrEqual(0);
          expect(clearIndex).toBeLessThan(errorIndex);
        });

        it("appends a plain ETA segment when the tracker yields an estimate", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          mockFormatEta.mockReturnValue("~2m 10s left");
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Plain text with ETA, no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, "~2m 10s left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · ~2m 10s left");
          expect(progressWrite).not.toContain("\x1b[2m");
        });

        it("renders a plain placeholder when ETA is not yet available for a sample step", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          mockEtaRecord.mockReturnValue(undefined);
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          await program.parseAsync(compareArgv("main", "branch"));

          // Placeholder as plain text, no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, "estimating time left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · estimating time left…");
          expect(progressWrite).not.toContain("\x1b[2m");
        });
      });

      describe("adapter warnings raised mid-run", () => {
        const WARNING = "Failed to parse METRIC line: METRIC foo=bar";
        const SAMPLE_STEP: ProgressStep = { kind: "sample", index: 1, total: 5, label: "baseline" };
        const SAMPLE_LINE = "sample 1/5 · baseline";

        /** Index of the stderr write carrying the adapter warning, or -1. */
        function warningWriteIndex(stderrSpy: ReturnType<typeof vi.spyOn>): number {
          return stderrWrites(stderrSpy).findIndex(
            (write) => typeof write === "string" && write.includes(WARNING),
          );
        }

        it("clears the overwritten progress line before the warning and redraws it after", async () => {
          const program = createRunnableProgram();
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks([SAMPLE_STEP], undefined, (opts) => {
            opts.warn?.(WARNING);
          });

          await program.parseAsync(compareArgv("main", "branch"));

          // The warning owns a clean line: cleared before, progress redrawn after
          const writes = stderrWrites(stderrSpy).map((write) => String(write));
          const warnIndex = warningWriteIndex(stderrSpy);
          expect.soft(warnIndex).toBeGreaterThan(0);
          expect.soft(writes[warnIndex - 1]).toBe("\r\x1b[K");
          expect(writes.slice(warnIndex + 1)).toContainEqual(expect.stringContaining(SAMPLE_LINE));
        });

        it("clears the spinner before the warning when the spinner owns the line", async () => {
          const program = createRunnableProgram();
          useColorTty();
          mockSpinnerInstance.isSpinning = true;
          const { stderrSpy } = await setupProgressMocks([SAMPLE_STEP], undefined, (opts) => {
            opts.warn?.(WARNING);
          });

          await program.parseAsync(compareArgv("main", "branch"));

          // spinner.clear() precedes the warning write
          const warnIndex = warningWriteIndex(stderrSpy);
          expect.soft(warnIndex).toBeGreaterThanOrEqual(0);
          const clearOrder = mockSpinnerInstance.clear.mock.invocationCallOrder.at(0) ?? Infinity;
          const warnOrder = stderrSpy.mock.invocationCallOrder[warnIndex] ?? -Infinity;
          expect(clearOrder).toBeLessThan(warnOrder);
        });

        it("prints the warning untouched when no progress line is on screen", async () => {
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([SAMPLE_STEP], undefined, (opts) => {
            opts.warn?.(WARNING);
          });

          await program.parseAsync(compareArgv("main", "branch"));

          // Nothing is cleared or redrawn around a warning with no line to protect
          const writes = stderrWrites(stderrSpy).map((write) => String(write));
          const warnIndex = warningWriteIndex(stderrSpy);
          expect.soft(warnIndex).toBeGreaterThanOrEqual(0);
          expect.soft(writes[warnIndex]).toBe(`${WARNING}\n`);
          expect(writes).not.toContainEqual("\r\x1b[K");
        });
      });
    });
  });
});
