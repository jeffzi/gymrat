/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */

import { afterEach, describe, expect, it, vi } from "vitest";

import { AdapterError } from "../src/adapters/index.js";
import { createProgram, formatCliError } from "../src/cli.js";
import type { ResolvedConfig } from "../src/config.js";
import type { ExecResult } from "../src/exec.js";
import type { ComparisonResult } from "../src/report/types.js";
import { CommandError, type CommandErrorContext } from "../src/sampling.js";
import { createRunnableProgram, mockProcessExit, stubWrite } from "./fixtures/cli-harness.js";
import { createComparisonResult } from "./fixtures/comparison-result.js";
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
  const failure: ExecResult = {
    exitCode: 1,
    stderr: "failed",
    stdout: "",
    stdoutBytes: 0,
    stderrBytes: Buffer.byteLength("failed", "utf-8"),
  };
  return new CommandError(context, failure);
}

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

describe("formatCliError", () => {
  describe("Error: label", () => {
    it("opens every error with a plain Error: label when NO_COLOR is set", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new Error("git rev-parse failed");

      const rendered = formatCliError(error);

      expect(rendered).toMatch(/^Error: git rev-parse failed/);
    });

    it("opens AdapterError with Error: followed by the class-name prefix", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new AdapterError("No valid METRIC lines found");

      const rendered = formatCliError(error);

      expect(rendered).toMatch(/^Error: AdapterError: No valid METRIC lines found/);
    });

    it("styles the Error label with ANSI red when color is forced", () => {
      vi.stubEnv("FORCE_COLOR", "1");
      const error = new Error("something broke");

      const rendered = formatCliError(error);

      // \x1b[31m = red
      expect.soft(rendered).toContain("\x1b[31m");
      expect(rendered).toContain("Error");
    });

    it("renders Error: as plain text when NO_COLOR is set", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new Error("something broke");

      const rendered = formatCliError(error);

      expect.soft(rendered).toMatch(/^Error: /);
      expect(rendered).not.toMatch(ANSI_RE);
    });

    it("renders Error: as plain text for a non-Error throwable", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const rendered = formatCliError("boom");

      expect(rendered).toMatch(/^Error: boom/);
    });
  });

  it("does not append a hint line when CommandError has undefined hint", () => {
    // In-place targets have no hint
    const error = createCommandError("in-place");

    const rendered = formatCliError(error);

    expect(rendered).not.toContain("Hint:");
  });

  it("does not append a hint line for a plain Error without hint field", () => {
    const error = new Error("git rev-parse failed");

    const rendered = formatCliError(error);

    expect(rendered).not.toContain("Hint:");
  });

  it("styles the Hint label with ANSI yellow+underline when color is forced", () => {
    vi.stubEnv("FORCE_COLOR", "1");
    const error = createCommandError("ref");

    const rendered = formatCliError(error);

    // \x1b[33m = yellow, \x1b[4m = underline
    expect.soft(rendered).toContain("\x1b[33m");
    expect.soft(rendered).toContain("\x1b[4m");
    expect(rendered).toContain("Hint");
  });

  it("renders Hint: as plain text when NO_COLOR is set", () => {
    vi.stubEnv("FORCE_COLOR", undefined);
    vi.stubEnv("NO_COLOR", "1");
    const error = createCommandError("ref");

    const rendered = formatCliError(error);

    expect.soft(rendered).toContain("\nHint: ");
    expect(rendered).not.toMatch(ANSI_RE);
  });

  describe("--debug stack trace", () => {
    /** Matches a V8 stack frame line: optional whitespace, then "at ". */
    const STACK_FRAME_RE = /^\s+at /m;

    it("includes the stack trace when debug is true", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new Error("something broke");

      const rendered = formatCliError(error, { debug: true });

      expect.soft(rendered).toContain("something broke");
      expect(rendered).toMatch(STACK_FRAME_RE);
    });

    it("omits the stack trace when debug is false", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new Error("something broke");

      const rendered = formatCliError(error, { debug: false });

      expect(rendered).not.toMatch(STACK_FRAME_RE);
    });

    it("omits the stack trace when debug is omitted", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new Error("something broke");

      const rendered = formatCliError(error);

      expect(rendered).not.toMatch(STACK_FRAME_RE);
    });

    it("places the stack trace between the message and any hint", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = createCommandError("ref");

      const rendered = formatCliError(error, { debug: true });

      // A stack frame line appears before the Hint: label
      const stackMatch = STACK_FRAME_RE.exec(rendered);
      const hintIndex = rendered.indexOf("Hint:");
      expect(stackMatch).not.toBeNull();
      if (stackMatch === null) throw new Error("expected stack match");
      expect.soft(hintIndex).toBeGreaterThan(0);
      expect(stackMatch.index).toBeLessThan(hintIndex);
    });
  });

  describe("bug-report footer", () => {
    it("appends a bug-report footer for a plain Error (unexpected)", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new Error("unexpected crash");

      const rendered = formatCliError(error);

      expect.soft(rendered).toContain("gymrat --debug");
      expect(rendered).toContain("https://github.com/jeffzi/gymrat/issues");
    });

    it("does not append a bug-report footer for GymratError", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = createCommandError("in-place");

      const rendered = formatCliError(error);

      expect(rendered).not.toContain("https://github.com/jeffzi/gymrat/issues");
    });

    it("does not append a bug-report footer for AdapterError", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const error = new AdapterError("parse failed");

      const rendered = formatCliError(error);

      expect(rendered).not.toContain("https://github.com/jeffzi/gymrat/issues");
    });

    it("appends a bug-report footer for a non-Error throwable", () => {
      // A thrown string is unexpected but not an Error instance
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const rendered = formatCliError("boom");

      expect.soft(rendered).toContain("gymrat --debug");
      expect(rendered).toContain("https://github.com/jeffzi/gymrat/issues");
    });
  });
});

describe("global --debug option", () => {
  /** Matches a V8 stack frame line: optional whitespace, then "at ". */
  const STACK_FRAME_RE = /^\s+at /m;

  it("is listed in the root --help output", async () => {
    vi.stubEnv("FORCE_COLOR", undefined);

    const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

    expect.soft(helpOutput).toContain("-d");
    expect(helpOutput).toContain("--debug");
  });

  it("defaults to false and does not include a stack trace on error", async () => {
    const program = createRunnableProgram({ exitOverride: "all" });
    await setupMocks(new Error("Compare failed"));
    stubWrite(process.stdout);
    const stderrSpy = stubWrite(process.stderr);
    mockProcessExit();

    await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
      "exitCode",
      2,
    );

    const errorOutput = stderrWrites(stderrSpy).join("");
    expect(errorOutput).not.toMatch(STACK_FRAME_RE);
  });

  it("includes a stack trace when --debug is passed", async () => {
    const program = createRunnableProgram({ exitOverride: "all" });
    await setupMocks(new Error("Compare failed"));
    stubWrite(process.stdout);
    const stderrSpy = stubWrite(process.stderr);
    mockProcessExit();

    await expect(
      program.parseAsync(compareArgv("main", "branch", "--debug")),
    ).rejects.toHaveProperty("exitCode", 2);

    const errorOutput = stderrWrites(stderrSpy).join("");
    expect.soft(errorOutput).toContain("Compare failed");
    expect(errorOutput).toMatch(STACK_FRAME_RE);
  });

  it("accepts -d as a short alias for --debug", async () => {
    const program = createRunnableProgram({ exitOverride: "all" });
    await setupMocks(new Error("Compare failed"));
    stubWrite(process.stdout);
    const stderrSpy = stubWrite(process.stderr);
    mockProcessExit();

    await expect(program.parseAsync(compareArgv("main", "branch", "-d"))).rejects.toHaveProperty(
      "exitCode",
      2,
    );

    expect(stderrWrites(stderrSpy).join("")).toMatch(STACK_FRAME_RE);
  });
});
