/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */
import { readFileSync } from "node:fs";

import { afterEach, describe, expect, it, type MockInstance, vi } from "vitest";

import { createProgram } from "../src/cli.js";
import type { ResolvedConfig } from "../src/config.js";
import { renderJson } from "../src/report/json.js";
import { renderReport } from "../src/report/text.js";
import type { ComparisonResult } from "../src/report/types.js";
import {
  createRunnableProgram,
  mockProcessExit,
  stubWrite,
  writtenChunks,
} from "./fixtures/cli-harness.js";
import {
  createCandidate,
  createComparisonResult,
  metricMeta,
} from "./fixtures/comparison-result.js";
import { ANSI_RE } from "./fixtures/constants.js";

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

/** The `compare` subcommand's help text. */
async function captureCompareHelp(): Promise<string> {
  return captureHelp(compareArgv("--help"));
}

/** Prepends the `["node", "cli.js", "compare"]` prefix Commander expects. */
function compareArgv(...args: string[]): string[] {
  return ["node", "cli.js", "compare", ...args];
}

/** Prepends the `["node", "cli.js", "measure"]` prefix Commander expects. */
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
  const writeSpy = stubWrite(process.stdout);
  await program.parseAsync(compareArgv("main", "branch", ...extraArgs));
  return writeSpy;
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
  it("reports the version declared in package.json", () => {
    const declaredVersion = readDeclaredVersion();

    const reportedVersion = createProgram().version();

    expect(reportedVersion).toBe(declaredVersion);
  });

  describe("when the package manifest has no string version field", () => {
    afterEach(() => {
      vi.doUnmock("node:fs");
      vi.resetModules();
    });

    it("throws a GymratError", async () => {
      // Reload the CLI against a manifest stripped of its version field
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

      expect(act).toThrow(FreshGymratError);
      expect(act).toThrow("package.json has no string version field");
    });
  });

  describe("when the root --help is requested", () => {
    /** The Unicode Box Drawing block, which every boxen border style draws from. */
    const BOX_DRAWING_RE = /[─-╿]/gu;

    /** The distinct box-drawing characters framing a help text, sorted. */
    function borderCharsOf(help: string): string[] {
      return [...new Set(help.match(BOX_DRAWING_RE) ?? [])].toSorted();
    }

    it("uses the same border style for root and subcommand help", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const [rootHelp, compareHelp] = await Promise.all([
        captureHelp(["node", "cli.js", "--help"]),
        captureCompareHelp(),
      ]);

      expect(borderCharsOf(rootHelp)).toStrictEqual(borderCharsOf(compareHelp));
    });

    it("ends with an examples block, a docs URL, and a bugs URL", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

      expect.soft(helpOutput).toContain("Examples:");
      expect.soft(helpOutput).toContain("• gymrat compare main my-branch --bench");
      expect.soft(helpOutput).toContain("• gymrat compare old=main new=perf/decode --bench");
      expect.soft(helpOutput).toContain("• gymrat measure --bench");
      expect.soft(helpOutput).toContain("Docs: https://github.com/jeffzi/gymrat#readme");
      expect(helpOutput).toContain("Bugs: https://github.com/jeffzi/gymrat/issues");
    });

    it("separates examples with blank lines and uses bullet markers", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

      // Each example is preceded by a blank line (except the first after the header)
      const lines = helpOutput.split("\n");
      const bulletIndices = lines
        .map((line, i) => (line.trim().startsWith("•") ? i : -1))
        .filter((i) => i >= 0);
      expect.soft(bulletIndices.length).toBeGreaterThanOrEqual(3);
      // After the first bullet, each subsequent bullet should be preceded by a blank line
      for (const idx of bulletIndices.slice(1)) {
        const prevLine = lines[idx - 1];
        if (prevLine === undefined) {
          throw new Error(`expected a line at index ${idx - 1}`);
        }
        expect(prevLine.trim()).toBe("");
      }
    });

    it("renders epilogue without ANSI escapes when stdout is not a TTY", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", undefined);

      const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

      // captureHelp captures via configureOutput (not a real TTY),
      // so the epilogue text must be ANSI-free
      const epilogueStart = helpOutput.indexOf("Examples:");
      expect.soft(epilogueStart).toBeGreaterThan(-1);
      const epilogue = helpOutput.slice(epilogueStart);
      expect(epilogue).not.toMatch(ANSI_RE);
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
          const program = createRunnableProgram();
          const { compareMock } = await setupMocks();

          await program.parseAsync(compareArgv(...positionals));

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
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        await setupMocks();
        mockProcessExit();

        const parsing = program.parseAsync(compareArgv(...positionals));

        await expect(parsing).rejects.toThrow(expected);
      });
    });

    describe("when flags provided", () => {
      it.each(CONFIG_FLAG_TABLE)(
        "passes $flag through to resolveConfig",
        async ({ flag, value, expected }) => {
          const program = createRunnableProgram();
          const { resolveConfigMock } = await setupMocks();

          await program.parseAsync(compareArgv("main", "branch", flag, value));

          expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining(expected));
        },
      );
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
        const program = createRunnableProgram();
        const { compareMock } = await setupMocks(undefined, resolved);

        await program.parseAsync(compareArgv("main", "branch"));

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
        // The coercion error is raised by the `compare` subcommand, and
        // Commander copies the exit callback to subcommands at .command() time, so
        // overriding on the parent alone lets the error reach process.exit instead.
        const program = createRunnableProgram({ exitOverride: "all", silent: true });

        const parsing = program.parseAsync(compareArgv("main", "branch", flag, value));

        // Commander renders the flag and the coercion reason
        await expect(parsing).rejects.toThrow(
          new RegExp(`option '[^']*${flag}[^']*'.*is invalid\\..*positive integer`),
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
        const program = createRunnableProgram({ exitOverride: "all", silent: true });

        const parsing = program.parseAsync(compareArgv("main", "branch", flag, value));

        // Commander renders the flag, and the reason names the bound
        await expect(parsing).rejects.toThrow(
          new RegExp(`option '[^']*${flag}[^']*' argument '${value}' is invalid\\..*${bound}`),
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
        const program = createRunnableProgram();
        const { resolveConfigMock } = await setupMocks();

        await program.parseAsync(compareArgv("main", "branch", flag, value));

        expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining(expected));
      });
    });

    describe("when unknown flag provided", () => {
      it("rejects with a usage error naming the unknown option", async () => {
        const program = createRunnableProgram({ exitOverride: "all" });

        await expect(
          program.parseAsync(compareArgv("main", "branch", "--bogus", "value")),
        ).rejects.toThrow(/unknown option '--bogus'/);
      });

      it("rejects --record, which only the single-target command carries", async () => {
        // Recording a comparison would have to name two sides; only
        // `measure` writes a baseline, so the flag stops at the command boundary.
        const program = createRunnableProgram({ exitOverride: "all" });

        await expect(program.parseAsync(compareArgv("main", "branch", "--record"))).rejects.toThrow(
          /unknown option '--record'/,
        );
      });
    });

    describe("when insufficient positionals", () => {
      it.each([
        { description: "only one positional", args: ["main"] },
        { description: "no positionals", args: [] },
      ])("rejects with a usage error when $description provided", async ({ args }) => {
        const program = createRunnableProgram({ exitOverride: "all" });

        await expect(program.parseAsync(compareArgv(...args))).rejects.toThrow(
          /missing required argument/,
        );
      });
    });

    describe("when --help requested", () => {
      it("writes the usage text and names the baseline and candidate roles and how they relate", async () => {
        vi.stubEnv("FORCE_COLOR", undefined);

        const helpOutput = await captureCompareHelp();

        expect.soft(helpOutput).toContain("Usage: gymrat compare");
        expect.soft(helpOutput).toContain("<baseline> <candidates...>");
        expect(helpOutput).toContain("judged against the baseline");
      });

      it("ends with a compare-specific examples block", async () => {
        vi.stubEnv("FORCE_COLOR", undefined);

        const helpOutput = await captureCompareHelp();

        expect.soft(helpOutput).toContain("Examples:");
        expect.soft(helpOutput).toContain("• gymrat compare main perf/faster-decode --bench");
        expect
          .soft(helpOutput)
          .toContain("• gymrat compare main perf/simd perf/lookup-table --bench");
        expect
          .soft(helpOutput)
          .toContain("• gymrat compare old=main new=perf/faster-decode --bench");
        expect(helpOutput).toContain("• gymrat compare main my-branch --bench");
      });
    });

    describe("on successful compare", () => {
      it("renders the comparison data compare returned and writes it to stdout", async () => {
        const result = createComparisonResult({
          baselineLabel: "main",
          candidates: [createCandidate({ label: "branch" })],
        });

        const writeSpy = await runCompareCapturingStdout(result);

        expect(writtenChunks(writeSpy)).toStrictEqual([`${renderReport(result)}\n`]);
      });
    });

    describeColorDecision((...flags) =>
      runCompareCapturingStdout(createColorSensitiveResult(), ...flags),
    );

    describe("when --format flag provided", () => {
      it("routes to renderJson for --format json", async () => {
        const result = createComparisonResult();

        const writeSpy = await runCompareCapturingStdout(result, "--format", "json");

        expect(vi.mocked(renderJson)).toHaveBeenCalledWith(result);
        expect(writtenChunks(writeSpy)).toStrictEqual(['{"report": true}\n']);
      });

      it.each([
        { desc: "a format that never existed", value: "csv" },
        { desc: "a format that no longer exists", value: "markdown" },
      ])(
        "rejects $desc with Commander's invalid-argument error naming the surviving choices",
        async ({ value }) => {
          const program = createRunnableProgram({ exitOverride: "all", silent: true });

          const parsing = program.parseAsync(compareArgv("main", "branch", "--format", value));

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
        const output = await renderWith("--verbose");

        expect(output).toContain("Wilcoxon signed-rank");
      });

      it("leaves the method footer out of the text report by default", async () => {
        const output = await renderWith();

        expect(output).not.toContain("Wilcoxon signed-rank");
      });

      it("leaves the JSON renderer untouched", async () => {
        const result = createComparisonResult();

        await runCompareCapturingStdout(result, "--format", "json", "--verbose");

        expect(vi.mocked(renderJson)).toHaveBeenCalledWith(result);
      });

      it("is documented in the compare help", async () => {
        const helpOutput = await captureCompareHelp();

        expect(helpOutput).toContain("--verbose");
      });
    });
  });
});
