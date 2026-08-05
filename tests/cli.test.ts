/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { Command } from "commander";
import { afterEach, describe, expect, it, type MockInstance, vi } from "vitest";

import { AdapterError } from "../src/adapters/index.js";
import { createProgram, formatCliError } from "../src/cli.js";
import {
  CommandError,
  type CommandErrorContext,
  type CompareOptions,
  type ProgressStep,
} from "../src/compare.js";
import type { ResolvedConfig } from "../src/config.js";
import type { ExecResult } from "../src/exec.js";
import { renderJson } from "../src/report/json.js";
import { renderReport } from "../src/report/text.js";
import type { ComparisonResult } from "../src/report/types.js";
import type { KindAggregate } from "../src/verdict/aggregate.js";
import type { GeomeanResult } from "../src/verdict/verdict.js";
import {
  createCandidate,
  createComparisonResult,
  geomeanOf,
  kindMetric,
  type Metrics,
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

vi.mock("../src/config.js", () => ({
  resolveConfig: vi.fn(),
}));

vi.mock("../src/report/json.js", () => ({
  renderJson: vi.fn().mockReturnValue('{"report": true}'),
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

async function setupMocks(
  compareMockReturn?: ComparisonResult | Error,
  resolveConfigMockReturn: Partial<ResolvedConfig> = {},
) {
  const { compare: compareMock } = await import("../src/compare.js");
  const { resolveConfig: resolveConfigMock } = await import("../src/config.js");

  const typedCompareMock = vi.mocked(compareMock);
  const typedConfigMock = vi.mocked(resolveConfigMock);

  const resolvedConfig: ResolvedConfig = {
    bench: "bench.sh",
    adapter: "metric-lines",
    samples: 1,
    timeoutSeconds: 300,
    unstableNoisePct: 200,
    ...resolveConfigMockReturn,
  };

  typedConfigMock.mockReturnValue(resolvedConfig);
  if (compareMockReturn instanceof Error) {
    typedCompareMock.mockRejectedValue(compareMockReturn);
  } else {
    typedCompareMock.mockResolvedValue(compareMockReturn ?? createComparisonResult());
  }

  return { compareMock, resolveConfigMock };
}

/**
 * Read the `version` field straight from the package manifest.
 *
 * The test reads the file itself rather than importing whatever the CLI uses,
 * so the assertion fails if the reported version ever stops tracking the
 * manifest — which is the whole point of the check.
 */
function readDeclaredVersion(): string {
  const manifest: unknown = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );

  if (
    typeof manifest !== "object" ||
    manifest === null ||
    !("version" in manifest) ||
    typeof manifest.version !== "string"
  ) {
    throw new Error("package.json has no string version field");
  }

  return manifest.version;
}

/**
 * Build a program whose subcommands also throw instead of exiting the process.
 *
 * `exitOverride()` applies only to the command it is called on, so a parse error
 * raised by the `compare` subcommand would otherwise reach `process.exit` and
 * surface as vitest's "process.exit unexpectedly called" rather than as the
 * `CommanderError` the test is asserting on.
 */
function createProgramWithSubcommandOverrides(): Command {
  const program = createProgram();
  for (const command of [program, ...program.commands]) {
    command.exitOverride();
  }
  return program;
}

/** Build a program whose subcommands throw instead of exiting, with stderr silenced. */
function createSilentProgram(): Command {
  const program = createProgramWithSubcommandOverrides();
  for (const command of [program, ...program.commands]) {
    command.configureOutput({ writeErr: () => {} });
  }
  return program;
}

/**
 * Build a program with stderr silenced but the production `exitOverride()` left in
 * place, so Commander's own exit code survives instead of being replaced by the
 * test helper's.
 */
function createSilentProgramWithProductionExit(): Command {
  const program = createProgram();
  for (const command of [program, ...program.commands]) {
    command.configureOutput({ writeErr: () => {} });
  }
  return program;
}

/**
 * The subcommand renders its own help, so every command needs its own output
 * config; `--help` throws rather than exiting because of `exitOverride`.
 */
async function captureCompareHelp(): Promise<string> {
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

  await expect(program.parseAsync(compareArgv("--help"))).rejects.toThrow();

  return helpOutput;
}

/**
 * Build a CommandError with minimal context for testing formatCliError.
 *
 * Only `target.kind` matters for hint assignment — ref targets get the hint,
 * in-place targets do not.
 */
function createCommandError(targetKind: "ref" | "in-place"): CommandError {
  const context: CommandErrorContext = {
    phase: "bench",
    position: "old",
    label: "baseline",
    command: "bench.sh",
    target:
      targetKind === "ref"
        ? { kind: "ref", ref: "main", resolvedSha: "abc123" }
        : { kind: "in-place", dir: "/tmp/test" },
    dir: "/tmp/test",
  };
  const failure: ExecResult = { exitCode: 1, stderr: "failed", stdout: "" };
  return new CommandError(context, failure);
}

/** Build a fresh program that throws instead of exiting, ready for a single successful parse. */
function createRunnableProgram(): Command {
  const program = createProgram();
  program.exitOverride();
  return program;
}

/** Prepends the `["node", "cli.js", "compare"]` prefix Commander expects. */
function compareArgv(...args: string[]): string[] {
  return ["node", "cli.js", "compare", ...args];
}

/** A single-metric, single-candidate result with an improved verdict, for tests that only care about output styling. */
function createColorSensitiveResult(): ComparisonResult {
  return createComparisonResult({
    baselineLabel: "main",
    candidates: [createCandidate({ label: "branch" })],
    metrics: {
      "decode/time": {
        baselineMedian: 100,
        baselineSpread: 1,
        candidates: [
          {
            median: 82.5,
            spread: 1,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -17.5,
              n: 10,
              p: 0.002,
              noisePct: 2.5,
              noiseAbs: 2.5,
            },
          },
        ],
        meta: metricMeta("decode/time", { unit: "ns" }),
      },
    },
  });
}

function stderrWrites(stderrSpy: ReturnType<typeof vi.spyOn>): unknown[] {
  return stderrSpy.mock.calls.map((c: unknown[]) => c[0]);
}

/**
 * Run `compare main branch <extraArgs>`, mocking `compare()` to resolve with
 * `result`, and hand back the stdout write spy.
 *
 * Callers that need TTY or env state in place before the run set it before
 * calling this helper — it only owns program creation, mock setup and the
 * write spy, in that order.
 */
async function runCompareCapturingStdout(
  result: ComparisonResult,
  ...extraArgs: string[]
): Promise<MockInstance<typeof process.stdout.write>> {
  const program = createRunnableProgram();
  await setupMocks(result);
  const writeSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
  await program.parseAsync(compareArgv("main", "branch", ...extraArgs));
  return writeSpy;
}

/**
 * Prevent process.exit from terminating the test runner.
 *
 * Converts exit calls into catchable rejections that carry the intended
 * exit code, so tests can assert on exit-code behavior without killing
 * the vitest worker.
 */
function mockProcessExit(): ReturnType<typeof vi.spyOn> {
  return vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
    throw Object.assign(new Error(`process.exit(${code})`), {
      exitCode: code,
    });
  }) as never);
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  mockSpinnerInstance.text = "";
  mockSpinnerInstance.isSpinning = false;
});

describe("createProgram", () => {
  it("reports the version declared in package.json", () => {
    // Arrange
    const declaredVersion = readDeclaredVersion();

    // Act
    const reportedVersion = createProgram().version();

    // Assert
    expect(reportedVersion).toBe(declaredVersion);
  });

  describe("when the package manifest has no string version field", () => {
    afterEach(() => {
      vi.doUnmock("node:fs");
      vi.resetModules();
    });

    it("throws a GymratError", async () => {
      // Arrange - reload the CLI against a manifest stripped of its version field
      vi.resetModules();
      vi.doMock("node:fs", async (importOriginal) => {
        const actual = await importOriginal<typeof import("node:fs")>();
        function readFileSyncWithoutVersion(
          ...args: Parameters<typeof actual.readFileSync>
        ): string | Buffer {
          const [path] = args;
          return String(path).endsWith("package.json")
            ? JSON.stringify({ name: "gymrat" })
            : actual.readFileSync(...args);
        }
        return {
          ...actual,
          default: { ...actual, readFileSync: readFileSyncWithoutVersion },
          readFileSync: readFileSyncWithoutVersion,
        };
      });
      const [
        { createProgram: createProgramWithBrokenManifest },
        { GymratError: FreshGymratError },
      ] = await Promise.all([import("../src/cli.js"), import("../src/errors.js")]);
      const act = (): void => {
        createProgramWithBrokenManifest();
      };

      // Act + Assert
      expect(act).toThrow(FreshGymratError);
      expect(act).toThrow("package.json has no string version field");
    });
  });

  describe("compare command", () => {
    describe("when valid positional arguments provided", () => {
      it.each([
        {
          form: "label=ref",
          positionals: ["baseline=main", "candidate=branch"],
          expected: {
            baseline: { target: "main", label: "baseline" },
            candidates: [{ target: "branch", label: "candidate" }],
          },
        },
        {
          form: "bare ref",
          positionals: ["main", "branch"],
          expected: {
            baseline: { target: "main", label: undefined },
            candidates: [{ target: "branch", label: undefined }],
          },
        },
        {
          form: "one baseline and two candidates",
          positionals: ["main", "branch-a", "second=branch-b"],
          expected: {
            baseline: { target: "main", label: undefined },
            candidates: [
              { target: "branch-a", label: undefined },
              { target: "branch-b", label: "second" },
            ],
          },
        },
      ])(
        "extracts targets and labels from $form positionals",
        async ({ positionals, expected }) => {
          // Arrange
          const program = createRunnableProgram();
          const { compareMock } = await setupMocks();

          // Act
          await program.parseAsync(compareArgv(...positionals));

          // Assert
          expect(compareMock).toHaveBeenCalledWith(expect.objectContaining(expected));
        },
      );
    });

    describe("when a positional has an empty label or target", () => {
      it.each([
        {
          form: "an empty label",
          positionals: ["=main", "branch"],
          expected: /label.*empty|empty.*label/i,
        },
        {
          form: "an empty target",
          positionals: ["main", "old="],
          expected: /target.*empty|empty.*target/i,
        },
      ])("rejects $form with a usage error", async ({ positionals, expected }) => {
        // Arrange
        const program = createSilentProgram();
        await setupMocks();
        mockProcessExit();

        // Act
        const parsing = program.parseAsync(compareArgv(...positionals));

        // Assert
        await expect(parsing).rejects.toThrow(expected);
      });
    });

    describe("when flags provided", () => {
      it.each([
        { flag: "--bench", value: "my-bench", expected: { bench: "my-bench" } },
        { flag: "--prepare", value: "setup.sh", expected: { prepare: "setup.sh" } },
        { flag: "--adapter", value: "mitata", expected: { adapter: "mitata" } },
        { flag: "--samples", value: "100", expected: { samples: 100 } },
        { flag: "--timeout", value: "5000", expected: { timeout: 5000 } },
        { flag: "--config", value: "gymrat.json", expected: { config: "gymrat.json" } },
      ])("passes $flag through to resolveConfig", async ({ flag, value, expected }) => {
        // Arrange
        const program = createRunnableProgram();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync(compareArgv("main", "branch", flag, value));

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining(expected));
      });
    });

    describe("when the resolved config carries settings compare needs", () => {
      it.each([
        {
          desc: "per-metric overrides",
          resolved: {
            metrics: {
              "decode/time": { direction: "higher" as const, gating: false, exact: true },
            },
          },
          expected: {
            configMetrics: {
              "decode/time": { direction: "higher" as const, gating: false, exact: true },
            },
          },
        },
        {
          desc: "an unstable noise threshold",
          resolved: { unstableNoisePct: 150.5 },
          expected: { unstableNoisePct: 150.5 },
        },
      ])("passes $desc through to compare", async ({ resolved, expected }) => {
        // Arrange
        const program = createRunnableProgram();
        const { compareMock } = await setupMocks(undefined, resolved);

        // Act
        await program.parseAsync(compareArgv("main", "branch"));

        // Assert
        expect(compareMock).toHaveBeenCalledWith(expect.objectContaining(expected));
      });
    });

    describe("when a numeric flag receives an invalid value", () => {
      it.each([
        { flag: "--samples", value: "abc", why: "non-numeric" },
        { flag: "--samples", value: "0", why: "zero" },
        { flag: "--samples", value: "-5", why: "negative" },
        { flag: "--samples", value: "1.5", why: "non-integer" },
        { flag: "--samples", value: "10abc", why: "trailing garbage" },
        { flag: "--timeout", value: "abc", why: "non-numeric" },
        { flag: "--timeout", value: "0", why: "zero" },
        { flag: "--timeout", value: "-5", why: "negative" },
        { flag: "--timeout", value: "1.5", why: "non-integer" },
        { flag: "--timeout", value: "10abc", why: "trailing garbage" },
      ])("rejects $flag $value ($why) with a usage error", async ({ flag, value }) => {
        // Arrange - the coercion error is raised by the `compare` subcommand, and
        // Commander copies the exit callback to subcommands at .command() time, so
        // overriding on the parent alone lets the error reach process.exit instead.
        const program = createSilentProgram();

        // Act
        const parsing = program.parseAsync(compareArgv("main", "branch", flag, value));

        // Assert - Commander renders the flag and the coercion reason
        await expect(parsing).rejects.toThrow(
          new RegExp(`option '${flag}[^']*'.*is invalid\\..*positive integer`),
        );
      });
    });

    describe("when a numeric flag exceeds the range its consumer can represent", () => {
      it.each([
        {
          flag: "--timeout",
          value: "2147484",
          why: "its millisecond expansion overflows the 32-bit timer limit",
          bound: "2147483",
        },
        {
          flag: "--samples",
          value: "9007199254740992",
          why: "it leaves the safe integer range",
          bound: "9007199254740991",
        },
      ])("rejects $flag $value because $why", async ({ flag, value, bound }) => {
        // Arrange
        const program = createSilentProgram();

        // Act
        const parsing = program.parseAsync(compareArgv("main", "branch", flag, value));

        // Assert - Commander renders the flag, and the reason names the bound
        await expect(parsing).rejects.toThrow(
          new RegExp(`option '${flag}[^']*' argument '${value}' is invalid\\..*${bound}`),
        );
      });

      it.each([
        { flag: "--timeout", value: "2147483", expected: { timeout: 2_147_483 } },
        {
          flag: "--samples",
          value: "9007199254740991",
          expected: { samples: 9_007_199_254_740_991 },
        },
      ])("accepts $flag at its maximum $value", async ({ flag, value, expected }) => {
        // Arrange
        const program = createRunnableProgram();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync(compareArgv("main", "branch", flag, value));

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining(expected));
      });
    });

    describe("when unknown flag provided", () => {
      it("rejects with a usage error naming the unknown option", async () => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();

        // Act & Assert
        await expect(
          program.parseAsync(compareArgv("main", "branch", "--bogus", "value")),
        ).rejects.toThrow(/unknown option '--bogus'/);
      });
    });

    describe("when insufficient positionals", () => {
      it.each([
        { description: "only one positional", args: ["main"] },
        { description: "no positionals", args: [] },
      ])("rejects with a usage error when $description provided", async ({ args }) => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();

        // Act & Assert
        await expect(program.parseAsync(compareArgv(...args))).rejects.toThrow(
          /missing required argument/,
        );
      });
    });

    describe("when --help requested", () => {
      it("writes the usage text and names the baseline and candidate roles and how they relate", async () => {
        // Arrange
        vi.stubEnv("FORCE_COLOR", undefined);

        // Act
        const helpOutput = await captureCompareHelp();

        // Assert
        expect.soft(helpOutput).toContain("Usage: gymrat compare");
        expect.soft(helpOutput).toContain("<baseline> <candidates...>");
        expect(helpOutput).toContain("judged against the baseline");
      });
    });

    describe("on successful compare", () => {
      it("renders the comparison data compare returned and writes it to stdout", async () => {
        // Arrange
        const result = createComparisonResult({
          baselineLabel: "main",
          candidates: [createCandidate({ label: "branch" })],
        });

        // Act
        const writeSpy = await runCompareCapturingStdout(result);

        // Assert
        expect(writeSpy).toHaveBeenCalledWith(`${renderReport(result)}\n`);
      });
    });

    describe("when deciding whether to color the report", () => {
      const originalIsTTY = process.stdout.isTTY;

      afterEach(() => {
        process.stdout.isTTY = originalIsTTY;
      });

      const ANSI_RE = /\x1b\[/;

      it("includes ANSI escapes when stdout is a terminal", async () => {
        // Arrange
        process.stdout.isTTY = true;
        vi.stubEnv("NO_COLOR", undefined);

        // Act
        const writeSpy = await runCompareCapturingStdout(createColorSensitiveResult());

        // Assert
        const output = writeSpy.mock.calls[0]![0];
        expect(output).toMatch(ANSI_RE);
      });

      it("omits ANSI escapes when stdout is redirected", async () => {
        // Arrange
        process.stdout.isTTY = false;
        vi.stubEnv("FORCE_COLOR", undefined);
        vi.stubEnv("NO_COLOR", undefined);

        // Act
        const writeSpy = await runCompareCapturingStdout(createColorSensitiveResult());

        // Assert
        const output = writeSpy.mock.calls[0]![0];
        expect(output).not.toMatch(ANSI_RE);
      });

      it("sets process.env.NO_COLOR when --no-color is passed", async () => {
        // Arrange
        vi.stubEnv("NO_COLOR", undefined);

        // Act
        await runCompareCapturingStdout(createComparisonResult(), "--no-color");

        // Assert
        expect(process.env.NO_COLOR).toBe("1");
      });
    });

    describe("when --format flag provided", () => {
      it("routes to renderJson for --format json", async () => {
        // Arrange
        const result = createComparisonResult();

        // Act
        const writeSpy = await runCompareCapturingStdout(result, "--format", "json");

        // Assert
        expect(vi.mocked(renderJson)).toHaveBeenCalledWith(result);
        expect(writeSpy).toHaveBeenCalledWith('{"report": true}\n');
      });

      it.each([
        { desc: "a format that never existed", value: "csv" },
        { desc: "a format that no longer exists", value: "markdown" },
      ])(
        "rejects $desc with Commander's invalid-argument error naming the surviving choices",
        async ({ value }) => {
          // Arrange
          const program = createSilentProgram();

          // Act
          const parsing = program.parseAsync(compareArgv("main", "branch", "--format", value));

          // Assert
          await expect(parsing).rejects.toThrow(
            new RegExp(
              `option '--format <value>' argument '${value}' is invalid\\. Allowed choices are text, json\\.`,
            ),
          );
        },
      );
    });

    describe("when --verbose is passed", () => {
      /** Runs `compare` over a signed-rank result and returns what reached stdout. */
      async function renderWith(...extraArgs: string[]): Promise<string> {
        const writeSpy = await runCompareCapturingStdout(
          createColorSensitiveResult(),
          ...extraArgs,
        );
        return String(writeSpy.mock.calls[0]?.[0]);
      }

      it("adds the method footer to the text report", async () => {
        // Act
        const output = await renderWith("--verbose");

        // Assert
        expect(output).toContain("Wilcoxon signed-rank");
      });

      it("leaves the method footer out of the text report by default", async () => {
        // Act
        const output = await renderWith();

        // Assert
        expect(output).not.toContain("Wilcoxon signed-rank");
      });

      it("leaves the JSON renderer untouched", async () => {
        // Arrange
        const result = createComparisonResult();

        // Act
        await runCompareCapturingStdout(result, "--format", "json", "--verbose");

        // Assert
        expect(vi.mocked(renderJson)).toHaveBeenCalledWith(result);
      });

      it("is documented in the compare help", async () => {
        // Act
        const helpOutput = await captureCompareHelp();

        // Assert
        expect(helpOutput).toContain("--verbose");
      });
    });

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
      ): Promise<{ stderrSpy: ReturnType<typeof vi.spyOn> }> {
        const { compareMock } = await setupMocks(mockReturn);

        // Progress steps fire before compare settles, mirroring the real onProgress timing.
        vi.mocked(compareMock).mockImplementation((opts: CompareOptions) => {
          for (const step of steps) {
            opts.onProgress?.(step);
          }
          return mockReturn instanceof Error
            ? Promise.reject(mockReturn)
            : Promise.resolve(mockReturn ?? createComparisonResult());
        });

        const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
        // Suppress stdout report output so tests focus on stderr
        vi.spyOn(process.stdout, "write").mockReturnValue(true);

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
          // Arrange
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          const writes = stderrWrites(stderrSpy);
          expect(writes).toContainEqual(expect.stringContaining(PREPARE_BASELINE_LINE));
          expect(writes).not.toContainEqual(expect.stringContaining("· bench"));
        });

        it("formats sample steps as 'sample <i>/<n> · <label>' without trailing '· bench'", async () => {
          // Arrange
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 1, total: 10, label: "baseline" },
          ]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          const writes = stderrWrites(stderrSpy);
          expect(writes).toContainEqual(expect.stringContaining("sample 1/10 · baseline"));
          expect(writes).not.toContainEqual(expect.stringContaining("· bench"));
        });
      });

      describe("when stderr is a TTY and color is allowed", () => {
        it("constructs a yocto-spinner with yellow color and stderr stream", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          expect(mockYoctoSpinner).toHaveBeenCalledWith({
            color: "yellow",
            stream: process.stderr,
          });
        });

        it("starts the spinner", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([{ kind: "prepare", label: "baseline" }]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          expect(mockSpinnerInstance.start).toHaveBeenCalled();
        });

        it("renders the prepare step word in default foreground (no dim) with cyan label", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([{ kind: "prepare", label: "baseline" }]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — step word has no dim wrapping, label is cyan, nothing is yellow
          const text = mockSpinnerInstance.text;
          expect.soft(text).toContain("prepare");
          expect.soft(text).not.toContain(DIM_O + "prepare" + DIM_C);
          expect.soft(text).toContain(CYN_O + "baseline" + CYN_C);
          expect(text).not.toContain("\x1b[33m");
        });

        it("renders the sample step word in default foreground (no dim) with bold counter and cyan label", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([{ kind: "sample", index: 1, total: 5, label: "baseline" }]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — step word has no dim wrapping, counter is bold, label is cyan
          const text = mockSpinnerInstance.text;
          expect.soft(text).toContain("sample");
          expect.soft(text).not.toContain(DIM_O + "sample" + DIM_C);
          expect.soft(text).toContain(BOLD_O + "1/5" + BOLD_C);
          expect(text).toContain(CYN_O + "baseline" + CYN_C);
        });

        it("appends a dim ETA segment when the tracker yields an estimate", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          mockFormatEta.mockReturnValue("~2m 10s left");
          await setupProgressMocks([{ kind: "sample", index: 2, total: 5, label: "baseline" }]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — ETA segment is rendered dim
          const text = mockSpinnerInstance.text;
          expect(text).toContain(DIM_O + " · ~2m 10s left" + DIM_C);
        });

        it("omits both ETA and placeholder for prepare steps", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(undefined);
          await setupProgressMocks([{ kind: "prepare", label: "baseline" }]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — no ETA segment and no placeholder
          const text = mockSpinnerInstance.text;
          expect.soft(text).not.toContain("left");
          expect(text).not.toContain("estimating");
        });

        it("renders a dim placeholder when ETA is not yet available for a sample step", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(undefined);
          await setupProgressMocks([{ kind: "sample", index: 1, total: 5, label: "baseline" }]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — placeholder is dim-styled
          const text = mockSpinnerInstance.text;
          expect(text).toContain(DIM_O + " · estimating time left…" + DIM_C);
        });
      });

      describe("when stderr is a TTY but color is vetoed", () => {
        it("does not construct a spinner when NO_COLOR is set", async () => {
          // Arrange
          const program = createRunnableProgram();
          useNoColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          expect(mockYoctoSpinner).not.toHaveBeenCalled();
        });

        it("does not construct a spinner when --no-color is passed", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch", "--no-color"));

          // Assert
          expect(mockYoctoSpinner).not.toHaveBeenCalled();
        });

        it("falls back to \\r\\x1b[K overwrite with unstyled text", async () => {
          // Arrange
          const program = createRunnableProgram();
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert - TTY fallback uses \r\x1b[K prefix with plain text
          const progressWrite = findStderrWrite(stderrSpy, PREPARE_BASELINE_LINE);
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toMatch(/^\r\x1b\[K/);
        });

        it("clears the last progress line with \\r\\x1b[K before the report", async () => {
          // Arrange
          const program = createRunnableProgram();
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 1, total: 1, label: "baseline" },
          ]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert - final stderr write clears the line
          const lastWrite = stderrWrites(stderrSpy)
            .map((w) => String(w))
            .at(-1);
          expect(lastWrite).toBe("\r\x1b[K");
        });

        it("clears progress before error text when compare throws", async () => {
          // Arrange
          const program = createProgramWithSubcommandOverrides();
          useNoColorTty();
          const { stderrSpy } = await setupProgressMocks(
            [{ kind: "sample", index: 1, total: 5, label: "baseline" }],
            new Error("adapter parse failed"),
          );
          mockProcessExit();

          // Act
          await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
            "exitCode",
            2,
          );

          // Assert - \r\x1b[K must appear before the error message in stderr writes
          const writes = stderrWrites(stderrSpy).map((w) => String(w));
          const clearIndex = writes.findIndex((w) => w === "\r\x1b[K");
          const errorIndex = writes.findIndex((w) => w.includes("adapter parse failed"));
          expect(clearIndex).toBeGreaterThanOrEqual(0);
          expect(errorIndex).toBeGreaterThanOrEqual(0);
          expect(clearIndex).toBeLessThan(errorIndex);
        });

        it("appends a plain ETA segment when the tracker yields an estimate", async () => {
          // Arrange
          const program = createRunnableProgram();
          useNoColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          mockFormatEta.mockReturnValue("~2m 10s left");
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — plain text with ETA, no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, "~2m 10s left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · ~2m 10s left");
          expect(progressWrite).not.toContain("\x1b[2m");
        });

        it("renders a plain placeholder when ETA is not yet available for a sample step", async () => {
          // Arrange
          const program = createRunnableProgram();
          useNoColorTty();
          mockEtaRecord.mockReturnValue(undefined);
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — placeholder as plain text, no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, "estimating time left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · estimating time left…");
          expect(progressWrite).not.toContain("\x1b[2m");
        });
      });

      describe("when stderr is not a TTY", () => {
        it("does not construct a spinner", async () => {
          // Arrange
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          expect(mockYoctoSpinner).not.toHaveBeenCalled();
        });

        it("writes newline-terminated lines without ANSI escapes", async () => {
          // Arrange
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          const { stderrSpy } = await setupProgressMocks([PREPARE_BASELINE_STEP]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert - non-TTY lines end with \n and contain no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, PREPARE_BASELINE_LINE);
          expect(progressWrite).toBeDefined();
          expect(progressWrite).not.toContain("\x1b[");
          expect(progressWrite).toMatch(/\n$/);
        });

        it("includes a plain ETA segment in newline-terminated output when estimate exists", async () => {
          // Arrange
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          mockEtaRecord.mockReturnValue(130_000);
          mockFormatEta.mockReturnValue("~2m 10s left");
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — newline-terminated, plain text with ETA
          const progressWrite = findStderrWrite(stderrSpy, "~2m 10s left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · ~2m 10s left");
          expect(progressWrite).toMatch(/\n$/);
          expect(progressWrite).not.toContain("\x1b[");
        });

        it("renders a plain placeholder when ETA is not yet available for a sample step", async () => {
          // Arrange
          const program = createRunnableProgram();
          process.stderr.isTTY = false;
          mockEtaRecord.mockReturnValue(undefined);
          const { stderrSpy } = await setupProgressMocks([
            { kind: "sample", index: 3, total: 10, label: "baseline" },
          ]);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert — placeholder as plain text, no ANSI escapes
          const progressWrite = findStderrWrite(stderrSpy, "estimating time left");
          expect(progressWrite).toBeDefined();
          expect(progressWrite).toContain("sample 3/10 · baseline · estimating time left…");
          expect(progressWrite).not.toContain("\x1b[");
        });
      });

      describe("spinner cleared before output", () => {
        it("stops the spinner before the report prints to stdout", async () => {
          // Arrange
          const program = createRunnableProgram();
          useColorTty();
          await setupProgressMocks([PREPARE_BASELINE_STEP]);
          const stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);

          // Act
          await program.parseAsync(compareArgv("main", "branch"));

          // Assert
          expect(mockSpinnerInstance.stop).toHaveBeenCalled();
          const stopOrder = mockSpinnerInstance.stop.mock.invocationCallOrder[0]!;
          const reportOrder = stdoutSpy.mock.invocationCallOrder[0]!;
          expect(stopOrder).toBeLessThan(reportOrder);
        });

        it("stops the spinner before a formatted error prints to stderr", async () => {
          // Arrange
          const program = createProgramWithSubcommandOverrides();
          useColorTty();
          const { stderrSpy } = await setupProgressMocks(
            [PREPARE_BASELINE_STEP],
            new Error("benchmark crashed"),
          );
          mockProcessExit();

          // Act
          await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
            "exitCode",
            2,
          );

          // Assert - spinner stopped before error message written
          expect(mockSpinnerInstance.stop).toHaveBeenCalled();
          const stopOrder = mockSpinnerInstance.stop.mock.invocationCallOrder[0]!;
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
          // Arrange
          const program = createProgramWithSubcommandOverrides();
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

          // Act
          await expect(
            program.parseAsync(compareArgv("main", "branch", "--fail-on", "regressed")),
          ).rejects.toHaveProperty("exitCode", 1);

          // Assert
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

          const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
          vi.spyOn(process.stdout, "write").mockReturnValue(true);

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
          // Arrange
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest([
            { kind: "sample", index: 2, total: 5, label: "baseline" },
          ]);

          // Act
          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          await vi.advanceTimersByTimeAsync(1000);

          // Assert — spinner text reflects the decremented ETA
          expect(mockSpinnerInstance.text).toContain("~129s left");

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });

        it("resets the countdown to the fresh ETA when a new emit arrives", async () => {
          // Arrange
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValueOnce(10_000).mockReturnValue(60_000);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest(
            [{ kind: "sample", index: 2, total: 5, label: "baseline" }],
            { step: { kind: "sample", index: 3, total: 5, label: "baseline" }, delayMs: 5000 },
          );

          // Act
          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          // Advance past the first ETA and trigger the second emit at t=5000
          await vi.advanceTimersByTimeAsync(5000);
          mockFormatEta.mockClear();
          // Advance 1s after the second emit — countdown should use new ETA (60_000)
          await vi.advanceTimersByTimeAsync(1000);

          // Assert — the countdown is from the new ETA (60_000 - 1000 = 59_000),
          // not the old depleted one (10_000 - 6000 = clamped to 0)
          expect(mockSpinnerInstance.text).toContain("~59s left");

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });

        it("clears the countdown interval when stop is called", async () => {
          // Arrange
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(130_000);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest([
            { kind: "sample", index: 2, total: 5, label: "baseline" },
          ]);

          // Act — start, advance to prove countdown is active, then stop
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
          // Arrange
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValueOnce(130_000).mockReturnValue(undefined);
          stubFormatEtaInSeconds();

          const { resolveCompare } = await setupCountdownTest(
            [{ kind: "sample", index: 2, total: 5, label: "baseline" }],
            { step: { kind: "sample", index: 3, total: 5, label: "baseline" }, delayMs: 2000 },
          );

          // Act
          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          // Advance 2s — the second emit fires with undefined ETA, clearing the interval
          await vi.advanceTimersByTimeAsync(2000);
          const textAfterUndefinedEta = mockSpinnerInstance.text;

          // Advance 2s more — if interval were still active, spinner text would keep changing
          await vi.advanceTimersByTimeAsync(2000);

          // Assert — no further updates after the interval was cleared
          expect(mockSpinnerInstance.text).toBe(textAfterUndefinedEta);

          // Cleanup
          await settleCompare(resolveCompare, parsePromise);
        });

        it("clamps the countdown ETA to zero instead of going negative", async () => {
          // Arrange
          vi.useFakeTimers();
          const program = createRunnableProgram();
          useColorTty();
          mockEtaRecord.mockReturnValue(1500);
          mockFormatEta.mockImplementation((ms: number) => `~${Math.ceil(ms / 1000)}s left`);

          const { resolveCompare } = await setupCountdownTest([
            { kind: "sample", index: 2, total: 5, label: "baseline" },
          ]);

          // Act — advance past the ETA
          const parsePromise = program.parseAsync(compareArgv("main", "branch"));
          await vi.advanceTimersByTimeAsync(5000);

          // Assert — every formatEta call used a non-negative value
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

    describe("on compare error", () => {
      it("exits 2 and writes the error to stderr", async () => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();
        await setupMocks(new Error("Compare failed"));
        vi.spyOn(process.stdout, "write").mockReturnValue(true);
        const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
        mockProcessExit();

        // Act & Assert
        await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
          "exitCode",
          2,
        );

        expect(stderrWrites(stderrSpy)).toContainEqual(expect.stringContaining("Compare failed"));
      });

      it("waits for stderr to drain before exiting", async () => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();
        await setupMocks(new Error("Compare failed"));
        vi.spyOn(process.stdout, "write").mockReturnValue(true);
        const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(false);
        const exitSpy = mockProcessExit();

        // Act
        const parsing = program.parseAsync(compareArgv("main", "branch")).catch((e: unknown) => e);
        await vi.waitFor(() => {
          expect(stderrSpy).toHaveBeenCalled();
        });

        // Assert — exit is held until stderr drains
        expect(exitSpy).not.toHaveBeenCalled();
        process.stderr.emit("drain");
        await expect(parsing).resolves.toHaveProperty("exitCode", 2);
      });
    });

    describe("--fail-on", () => {
      /**
       * A single-candidate result whose sole metric carries the given verdict.
       *
       * Defaults to a gating, lower-is-better metric so gate tests exercise the
       * standard path. Override `gating` or `geomeanValue` to test edge cases.
       */
      function createGatingResult(
        verdict: "regressed" | "improved" | "no-signal" | "unstable",
        {
          gating = true,
          geomeanValue = -5.0,
          geomeanN = 10,
        }: { gating?: boolean; geomeanValue?: number; geomeanN?: number } = {},
      ): ComparisonResult {
        const deltaByVerdict: Record<typeof verdict, number> = {
          regressed: 8,
          improved: -10,
          "no-signal": 0.2,
          unstable: 0.2,
        };
        const delta = deltaByVerdict[verdict];
        const aggregate = geomeanOf(geomeanValue, geomeanN);
        return createComparisonResult({
          candidates: [
            createCandidate({
              kinds: [
                {
                  kind: "other",
                  hasGating: gating,
                  geomean: aggregate,
                  groups: [],
                  ...(gating ? { gatedGeomean: aggregate } : {}),
                },
              ],
            }),
          ],
          metrics: {
            "decode/time": {
              baselineMedian: 100,
              baselineSpread: 1,
              candidates: [
                {
                  median: 100 + delta,
                  spread: 1,
                  verdict: {
                    verdict,
                    method: "signed-rank",
                    delta,
                    n: 10,
                    p: 0.01,
                    noisePct: verdict === "unstable" ? 300 : 2.5,
                    noiseAbs: verdict === "unstable" ? 300 : 2.5,
                  },
                },
              ],
              meta: metricMeta("decode/time", { gating }),
            },
          },
        });
      }

      /**
       * Arrange a --fail-on test: a program with subcommand overrides, `compare`
       * mocked to resolve (or reject) with `compareMockReturn`, stdout/stderr
       * stubbed, and process.exit converted into a catchable rejection that
       * carries the intended exit code.
       */
      async function setupFailOnTest(compareMockReturn: ComparisonResult | Error): Promise<{
        program: Command;
        stdoutSpy: ReturnType<typeof vi.spyOn>;
        stderrSpy: ReturnType<typeof vi.spyOn>;
        exitSpy: ReturnType<typeof vi.spyOn>;
      }> {
        const program = createProgramWithSubcommandOverrides();
        await setupMocks(compareMockReturn);
        const stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
        const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
        const exitSpy = mockProcessExit();
        return { program, stdoutSpy, stderrSpy, exitSpy };
      }

      /** The stderr line warning that a geomean gate had no stable metrics to measure, if any. */
      function geomeanGateWarning(stderrSpy: ReturnType<typeof vi.spyOn>): string | undefined {
        return stderrWrites(stderrSpy)
          .map((write) => String(write))
          .find((write) => write.includes("geomean gate"));
      }

      /** Forces plain (uncolored) output regardless of the ambient environment. */
      function disableColor(): void {
        vi.stubEnv("FORCE_COLOR", undefined);
        vi.stubEnv("NO_COLOR", "1");
      }

      it("exits 0 when no gating metric regressed", async () => {
        // Arrange
        const { program, stdoutSpy, exitSpy } = await setupFailOnTest(
          createGatingResult("improved"),
        );

        // Act
        await program.parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"));

        // Assert
        expect(stdoutSpy).toHaveBeenCalled();
        expect(exitSpy).not.toHaveBeenCalled();
      });

      it.each([
        {
          desc: "exceeds",
          geomeanValue: 5.0,
          expected: expect.objectContaining({ exitCode: 1 }),
        },
        {
          desc: "sits exactly on",
          geomeanValue: 2.0,
          expected: expect.objectContaining({ exitCode: 1 }),
        },
        { desc: "is within", geomeanValue: 1.5, expected: undefined },
      ])(
        "$desc the threshold when geomean delta is $geomeanValue%",
        async ({ geomeanValue, expected }) => {
          // Arrange - a 2% gate; trips exit 1 "at or worse than" the threshold
          const { program, stdoutSpy } = await setupFailOnTest(
            createGatingResult("no-signal", { geomeanValue }),
          );

          // Act - a gate trip rejects with exitCode 1, otherwise the parse resolves
          const error = await program
            .parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2"))
            .then(
              () => undefined,
              (e: unknown) => e,
            );

          // Assert - the report must be written before exit — distinguishes a gate
          // trip from a Commander parse error, which would not produce any report output.
          expect(error).toStrictEqual(expected);
          expect(stdoutSpy).toHaveBeenCalled();
        },
      );

      describe("when a candidate's geomean covers no stable gating metrics", () => {
        /** A candidate whose geomean aggregated nothing, gated on the most permissive threshold. */
        async function setupEmptyGeomeanGate(): Promise<
          Awaited<ReturnType<typeof setupFailOnTest>>
        > {
          disableColor();
          return setupFailOnTest(createGatingResult("unstable", { geomeanValue: 0, geomeanN: 0 }));
        }

        it("warns on stderr that the gate had nothing to measure", async () => {
          // Arrange
          const { program, stderrSpy } = await setupEmptyGeomeanGate();
          const { label } = createCandidate();

          // Act
          await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:0"));

          // Assert
          expect(geomeanGateWarning(stderrSpy)).toMatch(
            new RegExp(
              `warning: geomean gate for "${label}" had no stable gating metrics to measure`,
            ),
          );
        });

        it("does not trip the gate", async () => {
          // Arrange - a geomean of 0 would otherwise sit exactly on the 0% threshold
          const { program, exitSpy } = await setupEmptyGeomeanGate();

          // Act
          await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:0"));

          // Assert
          expect(exitSpy).not.toHaveBeenCalled();
        });
      });

      describe("when a candidate spans several metric kinds", () => {
        /** A kind whose metrics gate, aggregating to `gated` both overall and when gated. */
        function gatingKind(kind: string, gated: GeomeanResult): KindAggregate {
          return { kind, hasGating: true, geomean: gated, groups: [], gatedGeomean: gated };
        }

        /** An informational kind: it aggregates a geomean but nothing of it gates. */
        function informationalKind(kind: string, geomean: GeomeanResult): KindAggregate {
          return { kind, hasGating: false, geomean, groups: [] };
        }

        /**
         * A one-candidate result with the given kind aggregates and one metric
         * per kind so the rendered report and the aggregates describe the same run.
         */
        function createKindGatingResult(kinds: readonly KindAggregate[]): ComparisonResult {
          const metrics: Metrics = Object.fromEntries(
            kinds.map((aggregate) => [
              `decode/${aggregate.kind}`,
              kindMetric({
                kind: aggregate.kind,
                shortName: "decode",
                verdict: "no-signal",
                delta: 0.2,
                gating: aggregate.hasGating,
              }),
            ]),
          );
          return createComparisonResult({
            candidates: [createCandidate({ kinds })],
            metrics,
          });
        }

        it.each([
          {
            shape: "a gating kind is at or worse than the threshold",
            kinds: [
              gatingKind("time", geomeanOf(5, 1)),
              informationalKind("memory", geomeanOf(-5, 1)),
            ],
          },
          {
            shape: "the gating kind with data when another gating kind has none",
            kinds: [gatingKind("time", geomeanOf(0, 0)), gatingKind("cpu", geomeanOf(5, 2))],
          },
        ])("exits 1 when $shape", async ({ kinds }) => {
          const { program } = await setupFailOnTest(createKindGatingResult(kinds));

          // Act & Assert
          await expect(
            program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2")),
          ).rejects.toHaveProperty("exitCode", 1);
        });

        it("does not trip on a non-gating kind's geomean", async () => {
          const { program, exitSpy } = await setupFailOnTest(
            createKindGatingResult([
              gatingKind("time", geomeanOf(-1, 1)),
              informationalKind("memory", geomeanOf(11, 1)),
            ]),
          );

          // Act
          await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2"));

          // Assert
          expect(exitSpy).not.toHaveBeenCalled();
        });

        describe("when no gating kind measured a stable metric", () => {
          const vacuousKinds: readonly KindAggregate[] = [
            gatingKind("time", geomeanOf(0, 0)),
            informationalKind("memory", geomeanOf(-5, 1)),
          ];

          it.each([
            { shape: "every gating kind measured nothing", kinds: vacuousKinds },
            {
              shape: "no kind gates at all",
              kinds: [
                informationalKind("memory", geomeanOf(-5, 1)),
                informationalKind("other", geomeanOf(1, 1)),
              ],
            },
          ])("warns on stderr when $shape", async ({ kinds }) => {
            disableColor();
            const { program, stderrSpy } = await setupFailOnTest(createKindGatingResult(kinds));
            const { label } = createCandidate();

            // Act
            await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:0"));

            // Assert
            expect(geomeanGateWarning(stderrSpy)).toContain(
              `warning: geomean gate for "${label}" had no stable gating metrics to measure`,
            );
          });
        });
      });

      it("hands the conditions to the renderer, so the report echoes the tripped gate", async () => {
        // Arrange - a gating "time" kind at +5.0%, well past the 2% gate
        disableColor();
        const gated = geomeanOf(5, 1);
        const { program, stdoutSpy } = await setupFailOnTest(
          createComparisonResult({
            candidates: [
              createCandidate({
                kinds: [
                  {
                    kind: "time",
                    hasGating: true,
                    geomean: gated,
                    groups: [],
                    gatedGeomean: gated,
                  },
                ],
              }),
            ],
            metrics: {
              "decode/time": kindMetric({
                kind: "time",
                shortName: "decode",
                verdict: "regressed",
                delta: 5,
              }),
            },
          }),
        );

        // Act & Assert
        await expect(
          program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2")),
        ).rejects.toHaveProperty("exitCode", 1);
        const report = stdoutSpy.mock.calls.map((call: unknown[]) => String(call[0])).join("");
        expect(report).toContain("⚑ time geomean +5.0% exceeded --fail-on geomean:2");
      });

      it("trips when any of multiple conditions matches", async () => {
        // Arrange - no regression verdict, but geomean +5.0% exceeds the 2% gate
        const { program, stdoutSpy } = await setupFailOnTest(
          createGatingResult("improved", { geomeanValue: 5.0 }),
        );

        // Act & Assert
        await expect(
          program.parseAsync(
            compareArgv("main", "branch", "--fail-on", "regressed", "--fail-on", "geomean:2"),
          ),
        ).rejects.toHaveProperty("exitCode", 1);
        // The report must be written — distinguishes a gate trip from a parse error
        expect(stdoutSpy).toHaveBeenCalled();
      });

      it("rejects a malformed condition with the allowed grammar in the error", async () => {
        // Arrange
        const program = createSilentProgram();
        mockProcessExit();

        // Act & Assert
        await expect(
          program.parseAsync(compareArgv("main", "branch", "--fail-on", "bogus")),
        ).rejects.toThrow(/regressed.*geomean|geomean.*regressed/);
      });

      it.each([
        { form: "an empty percentage", value: "geomean:" },
        { form: "a whitespace percentage", value: "geomean: " },
        { form: "a hexadecimal percentage", value: "geomean:0x10" },
      ])("rejects geomean with $form", async ({ value }) => {
        // Arrange
        const program = createSilentProgram();
        await setupMocks();
        mockProcessExit();

        // Act
        const parsing = program.parseAsync(compareArgv("main", "branch", "--fail-on", value));

        // Assert
        await expect(parsing).rejects.toThrow(/regressed.*geomean|geomean.*regressed/);
      });

      it("prints the report to stdout before exiting 1 on gate trip", async () => {
        // Arrange
        const { program, stdoutSpy, exitSpy } = await setupFailOnTest(
          createGatingResult("regressed"),
        );

        // Act
        const error = await program
          .parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"))
          .catch((e: unknown) => e);

        // Assert - report was written before exit was called
        expect(error).toHaveProperty("exitCode", 1);
        expect(stdoutSpy).toHaveBeenCalled();
        const reportOrder = stdoutSpy.mock.invocationCallOrder[0]!;
        const exitOrder = exitSpy.mock.invocationCallOrder[0]!;
        expect(reportOrder).toBeLessThan(exitOrder);
      });

      it("does not trip when the regressed metric is non-gating", async () => {
        // Arrange - the only regressed metric has gating: false
        const { program, stdoutSpy } = await setupFailOnTest(
          createGatingResult("regressed", { gating: false }),
        );

        // Act
        await program.parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"));

        // Assert
        expect(stdoutSpy).toHaveBeenCalled();
      });

      describe("when stdout applies backpressure", () => {
        it("waits for the report to flush before exiting", async () => {
          // Arrange - a write returning false means the pipe buffer is full and
          // the data is queued; exiting now would truncate the report.
          const { program, stdoutSpy, exitSpy } = await setupFailOnTest(
            createGatingResult("regressed"),
          );
          stdoutSpy.mockReturnValue(false);

          // Act
          const parsing = program
            .parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"))
            .catch((e: unknown) => e);
          await vi.waitFor(() => {
            expect(stdoutSpy).toHaveBeenCalled();
          });

          // Assert - the exit is held until the stream drains
          expect(exitSpy).not.toHaveBeenCalled();
          process.stdout.emit("drain");
          await expect(parsing).resolves.toHaveProperty("exitCode", 1);
        });
      });

      it("exits 2 for Commander usage errors", async () => {
        // Arrange - the production exitOverride (which sets exit code 2) must
        // survive here rather than being replaced by the test helper's plain one
        const program = createSilentProgramWithProductionExit();
        mockProcessExit();

        // Act & Assert
        await expect(
          program.parseAsync(compareArgv("main", "branch", "--bogus")),
        ).rejects.toHaveProperty("exitCode", 2);
      });
    });
  });
});

describe("formatCliError", () => {
  it("labels an adapter failure with its error class", () => {
    // Arrange - adapter messages name the parse failure but not the layer it came from
    const error = new AdapterError("No valid METRIC lines found");

    // Act
    const rendered = formatCliError(error);

    // Assert
    expect(rendered).toBe("AdapterError: No valid METRIC lines found");
  });

  it.each([
    {
      description: "a plain Error",
      error: new Error("git rev-parse failed"),
      expected: "git rev-parse failed",
    },
    { description: "a non-Error throwable", error: "boom", expected: "boom" },
  ])("renders $description unlabelled", ({ error, expected }) => {
    // Act
    const rendered = formatCliError(error);

    // Assert
    expect(rendered).toBe(expected);
  });

  it("does not append a hint line when CommandError has undefined hint", () => {
    // Arrange - in-place targets have no hint
    const error = createCommandError("in-place");

    // Act
    const rendered = formatCliError(error);

    // Assert
    expect(rendered).not.toContain("Hint:");
  });

  it("does not append a hint line for a plain Error without hint field", () => {
    // Arrange
    const error = new Error("git rev-parse failed");

    // Act
    const rendered = formatCliError(error);

    // Assert
    expect(rendered).not.toContain("Hint:");
  });

  it("styles the Hint label with ANSI yellow+underline when color is forced", () => {
    // Arrange
    vi.stubEnv("FORCE_COLOR", "1");
    const error = createCommandError("ref");

    // Act
    const rendered = formatCliError(error);

    // Assert - \x1b[33m = yellow, \x1b[4m = underline
    expect.soft(rendered).toContain("\x1b[33m");
    expect.soft(rendered).toContain("\x1b[4m");
    expect(rendered).toContain("Hint");
  });

  it("renders Hint: as plain text when NO_COLOR is set", () => {
    // Arrange
    vi.stubEnv("FORCE_COLOR", undefined);
    vi.stubEnv("NO_COLOR", "1");
    const error = createCommandError("ref");

    // Act
    const rendered = formatCliError(error);

    // Assert
    expect.soft(rendered).toContain("\nHint: ");
    expect(rendered).not.toContain("\x1b[");
  });
});

interface CliProcessResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

/**
 * Run a CLI entry file in a child process and collect its exit code and streams.
 *
 * The entry-point block only runs when the file is the process entry, so these
 * behaviors are unreachable in-process. The child also yields the real exit code
 * and the raw stderr text, both of which the assertions inspect.
 */
function runCliProcess(entry: string, args: string[]): Promise<CliProcessResult> {
  return new Promise<CliProcessResult>((settle) => {
    execFile(
      process.execPath,
      ["--import", "tsx", entry, ...args],
      {
        timeout: 10000,
        env: { ...process.env, FORCE_COLOR: undefined, NO_COLOR: "1" },
      },
      (error, stdout, stderr) => {
        settle({
          exitCode: typeof error?.code === "number" ? error.code : 0,
          stdout,
          stderr,
        });
      },
    );
  });
}

describe("entry point", () => {
  it("executes CLI when invoked through symlink", async () => {
    // Arrange
    const tmpDir = mkdtempSync(join(tmpdir(), "gymrat-cli-test-"));
    const cliPath = resolve("src/cli.ts");
    const symlinkPath = join(tmpDir, "cli-symlink.ts");

    try {
      symlinkSync(cliPath, symlinkPath);

      // Act
      const { stdout } = await runCliProcess(symlinkPath, ["compare", "--help"]);

      // Assert
      expect(stdout).toContain("Usage: gymrat compare");
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  }, 20_000);

  describe("when Commander rejects the arguments", () => {
    it("reports the usage error on stderr exactly once", async () => {
      // Act - Commander writes the error itself before the entry point sees it
      const { exitCode, stderr } = await runCliProcess(resolve("src/cli.ts"), [
        "compare",
        "main",
        "branch",
        "--bogus",
      ]);

      // Assert
      expect.soft(exitCode).toBe(2);
      expect(stderr.match(/unknown option '--bogus'/g) ?? []).toHaveLength(1);
    }, 20_000);
  });

  describe("when the process has no entry-path argument", () => {
    const originalArgv = process.argv;

    afterEach(() => {
      process.argv = originalArgv;
      vi.resetModules();
    });

    it("imports cleanly when process.argv has no entry path", async () => {
      // Arrange - `node -e` and the REPL both leave process.argv[1] undefined
      vi.resetModules();
      process.argv = [process.execPath];

      // Act
      const loading = import("../src/cli.js");

      // Assert
      await expect(loading).resolves.toHaveProperty("createProgram");
    });
  });
});
