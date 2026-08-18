/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createProgram } from "../src/cli.js";
import type { ResolvedConfig } from "../src/config.js";
import {
  captureStdout,
  createRunnableProgram,
  mockProcessExit,
  stubWrite,
  writtenChunks,
} from "./fixtures/cli-harness.js";

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

// ---------------------------------------------------------------------------
// Doctor command
// ---------------------------------------------------------------------------

describe("the doctor command", () => {
  /** Prepends the `["node", "cli.js", "doctor"]` prefix Commander expects. */
  function doctorArgv(...args: string[]): string[] {
    return ["node", "cli.js", "doctor", ...args];
  }

  /** Set up the doctor mocks so the command has a report to work with. */
  async function setupDoctorMocks(
    options: {
      hasFailures?: boolean;
      configFailure?: boolean;
      throwError?: Error;
    } = {},
  ): Promise<void> {
    const { inspectConfig: inspectConfigMock } = await import("../src/config-inspect.js");
    const {
      buildEnvironmentSection: buildEnvMock,
      buildConfigSection: buildConfigMock,
      buildWorkflowSection: buildWorkflowMock,
    } = await import("../src/doctor/checks.js");
    const { buildBenchSection: buildBenchMock } = await import("../src/doctor/bench.js");

    const envSection = {
      title: "Environment",
      checks: [{ name: "git", status: "ok" as const, detail: "available" }],
    };
    const configSection = {
      title: "Configuration",
      checks: options.configFailure
        ? [
            {
              name: "config",
              status: "fail" as const,
              detail: "not found",
              hint: "create gymrat.json",
            },
          ]
        : [{ name: "config", status: "ok" as const, detail: "/project/gymrat.json" }],
    };
    const workflowSection = {
      title: "Workflow",
      checks: [{ name: "skill file", status: "ok" as const, detail: "found" }],
    };
    const benchSection = {
      title: "Bench",
      checks: options.hasFailures
        ? [{ name: "smoke run", status: "fail" as const, detail: "bench crashed" }]
        : [{ name: "smoke run", status: "ok" as const, detail: "1 metric found" }],
    };

    vi.mocked(inspectConfigMock).mockReturnValue({
      configPath: "/project/gymrat.json",
      configExists: true,
      problems: [],
      config: resolvedConfigFixture(),
    });

    if (options.throwError !== undefined) {
      const error = options.throwError;
      vi.mocked(buildEnvMock).mockImplementation(() => {
        throw error;
      });
    } else {
      vi.mocked(buildEnvMock).mockReturnValue(envSection);
    }
    vi.mocked(buildConfigMock).mockReturnValue(configSection);
    vi.mocked(buildWorkflowMock).mockReturnValue(workflowSection);
    vi.mocked(buildBenchMock).mockResolvedValue(benchSection);
  }

  describe("when --help is requested", () => {
    it("lists doctor in the root help", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

      expect(helpOutput).toContain("doctor");
    });

    it("documents --no-bench, --no-color, and --format", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const helpOutput = await captureHelp(doctorArgv("--help"));

      expect.soft(helpOutput).toContain("--no-bench");
      expect.soft(helpOutput).toContain("--no-color");
      expect(helpOutput).toContain("--format");
    });
  });

  describe("when the report has no failures", () => {
    it("exits 0 and writes the text report to stdout", async () => {
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ hasFailures: false });
      const stdoutSpy = stubWrite(process.stdout);
      stubWrite(process.stderr);

      await program.parseAsync(doctorArgv());

      expect(writtenChunks(stdoutSpy).join("")).toContain("doctor text report");
    });
  });

  describe("when the report has failures", () => {
    it("exits 1 after writing the report", async () => {
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ hasFailures: true });
      const stdoutSpy = stubWrite(process.stdout);
      stubWrite(process.stderr);
      mockProcessExit();

      const error = await program.parseAsync(doctorArgv()).catch((e: unknown) => e);

      // The report was written before exit — distinguishes a doctor
      // exit 1 from a Commander parse error, which would not produce any output.
      expect(error).toHaveProperty("exitCode", 1);
      expect(writtenChunks(stdoutSpy).join("")).toContain("doctor text report");
    });
  });

  describe("when --format json is passed", () => {
    it("writes JSON to stdout with nothing else", async () => {
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ hasFailures: false });
      const stdoutSpy = stubWrite(process.stdout);
      stubWrite(process.stderr);

      await program.parseAsync(doctorArgv("--format", "json"));

      expect(writtenChunks(stdoutSpy)).toStrictEqual(['{"doctor": true}\n']);
    });
  });

  describe("when --no-color is passed", () => {
    it("sets process.env.NO_COLOR", async () => {
      vi.stubEnv("NO_COLOR", undefined);
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ hasFailures: false });
      stubWrite(process.stdout);
      stubWrite(process.stderr);

      await program.parseAsync(doctorArgv("--no-color"));

      expect(process.env.NO_COLOR).toBe("1");
    });
  });

  describe("--no-bench flag", () => {
    it.each([
      { argv: ["--no-bench"], noBench: true },
      { argv: [], noBench: false },
    ])("reaches buildBenchSection with noBench $noBench", async ({ argv, noBench }) => {
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ hasFailures: false });
      stubWrite(process.stdout);
      stubWrite(process.stderr);

      await program.parseAsync(doctorArgv(...argv));

      const { buildBenchSection: buildBenchMock } = await import("../src/doctor/bench.js");
      expect(vi.mocked(buildBenchMock)).toHaveBeenCalledWith(expect.objectContaining({ noBench }));
    });
  });

  describe("when --config names a missing path", () => {
    it("does not abort — surfaces the missing file as a config FAIL and continues", async () => {
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ configFailure: true });
      const { inspectConfig: inspectConfigMock } = await import("../src/config-inspect.js");
      vi.mocked(inspectConfigMock).mockReturnValue({
        configPath: "/missing/gymrat.json",
        configExists: false,
        problems: ["Config file not found at /missing/gymrat.json"],
      });
      const stdoutSpy = stubWrite(process.stdout);
      stubWrite(process.stderr);
      mockProcessExit();

      const error = await program
        .parseAsync(doctorArgv("--config", "/missing/gymrat.json"))
        .catch((e: unknown) => e);

      // Exits 1 (failures in report), NOT exit 2 (crash).
      // The report must be written — distinguishes from a Commander parse error.
      expect(error).toHaveProperty("exitCode", 1);
      expect(writtenChunks(stdoutSpy).join("")).toContain("doctor text report");
    });
  });

  describe("when doctor itself crashes", () => {
    it("routes through exitWithError with exit code 2", async () => {
      const program = createRunnableProgram({ exitOverride: "all" });
      await setupDoctorMocks({ throwError: new Error("unexpected doctor crash") });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      await expect(program.parseAsync(doctorArgv())).rejects.toHaveProperty("exitCode", 2);
      expect(stderrWrites(stderrSpy)).toContainEqual(
        expect.stringContaining("unexpected doctor crash"),
      );
    });
  });
});

// ---------------------------------------------------------------------------
// Init command
// ---------------------------------------------------------------------------

describe("the init command", () => {
  /** Prepends the `["node", "cli.js", "init"]` prefix Commander expects. */
  function initArgv(...args: string[]): string[] {
    return ["node", "cli.js", "init", ...args];
  }

  /** A default wizard result with all artifacts requested. */
  function createWizardResult(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return {
      bench: "npm run bench",
      runbook: { path: "gymrat-runbook.md" },
      installSkill: true,
      ...overrides,
    };
  }

  /** A default scaffold result with all artifacts created. */
  function createScaffoldResult(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return {
      config: { path: "gymrat.json", status: "created" },
      runbook: { path: "gymrat-runbook.md", status: "created" },
      skill: { path: ".claude/skills/gymrat/SKILL.md", status: "created" },
      ...overrides,
    };
  }

  describe("when --help is requested", () => {
    it("lists init in the root help", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const helpOutput = await captureHelp(["node", "cli.js", "--help"]);

      expect(helpOutput).toContain("init");
    });

    it("documents --bench, --adapter, --runbook, --skill, and --yes flags", async () => {
      vi.stubEnv("FORCE_COLOR", undefined);

      const helpOutput = await captureHelp(initArgv("--help"));

      expect.soft(helpOutput).toContain("--bench");
      expect.soft(helpOutput).toContain("--adapter");
      expect.soft(helpOutput).toContain("--runbook");
      expect.soft(helpOutput).toContain("--skill");
      expect.soft(helpOutput).toContain("--yes");
      expect(helpOutput).toContain("-y");
    });
  });

  describe("when gymrat.json already exists at the resolved base", () => {
    let tempDir: string;

    beforeEach(() => {
      tempDir = mkdtempSync(join(tmpdir(), "gymrat-init-test-"));
    });

    afterEach(() => {
      rmSync(tempDir, { recursive: true, force: true });
    });

    it("exits 2 with a message about the existing config, editing directly, and gymrat doctor", async () => {
      writeFileSync(join(tempDir, "gymrat.json"), "{}");
      vi.spyOn(process, "cwd").mockReturnValue(tempDir);
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(initArgv());

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const output = stderrWrites(stderrSpy).map(String).join("");
      expect.soft(output).toMatch(/already exists/i);
      expect(output).toContain("gymrat doctor");
    });
  });

  describe("when the wizard rejects because --bench is missing in non-interactive mode", () => {
    it("exits 2 with an error naming --bench", async () => {
      const program = createRunnableProgram({ exitOverride: "all" });
      mockRunWizard.mockRejectedValue(new Error("--bench is required in non-interactive mode"));
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(initArgv("--yes"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      expect(stderrWrites(stderrSpy).map(String).join("")).toContain("--bench");
    });
  });

  describe("when scaffolding succeeds", () => {
    it("writes a summary to stdout naming each artifact and closes with a gymrat doctor pointer", async () => {
      const program = createRunnableProgram();
      mockRunWizard.mockResolvedValue(createWizardResult());
      mockScaffold.mockReturnValue(createScaffoldResult());
      const getStdout = captureStdout({ silenceStderr: true });

      await program.parseAsync(initArgv("--bench", "npm run bench", "--yes"));

      const output = getStdout();
      expect.soft(output).toContain("gymrat.json");
      expect.soft(output).toContain("created");
      expect(output).toContain("gymrat doctor");
    });

    it("reports a runbook that already existed", async () => {
      const program = createRunnableProgram();
      mockRunWizard.mockResolvedValue(createWizardResult());
      mockScaffold.mockReturnValue(
        createScaffoldResult({
          runbook: { path: "gymrat-runbook.md", status: "exists" },
        }),
      );
      const getStdout = captureStdout({ silenceStderr: true });

      await program.parseAsync(initArgv("--bench", "npm run bench", "--yes"));

      expect(getStdout()).toMatch(/runbook.*already exist/i);
    });

    it("reports a declined runbook", async () => {
      const program = createRunnableProgram();
      mockRunWizard.mockResolvedValue(createWizardResult({ runbook: false }));
      mockScaffold.mockReturnValue(
        createScaffoldResult({
          runbook: { path: "", status: "declined" },
        }),
      );
      const getStdout = captureStdout({ silenceStderr: true });

      await program.parseAsync(initArgv("--bench", "npm run bench", "--yes", "--no-runbook"));

      expect(getStdout()).toMatch(/runbook.*decline|skip/i);
    });

    it("reports a declined skill", async () => {
      const program = createRunnableProgram();
      mockRunWizard.mockResolvedValue(createWizardResult({ installSkill: false }));
      mockScaffold.mockReturnValue(
        createScaffoldResult({
          skill: { path: "", status: "declined" },
        }),
      );
      const getStdout = captureStdout({ silenceStderr: true });

      await program.parseAsync(initArgv("--bench", "npm run bench", "--yes", "--no-skill"));

      expect(getStdout()).toMatch(/skill.*decline|skip/i);
    });
  });

  describe("when flags are forwarded to the wizard", () => {
    it.each([
      {
        argv: ["--bench", "my-bench.sh", "--yes"],
        expected: { bench: "my-bench.sh" },
      },
      {
        argv: ["--bench", "bench.sh", "--adapter", "mitata", "--yes"],
        expected: { adapter: "mitata" },
      },
      {
        argv: ["--bench", "bench.sh", "--stop-target", "1.5", "--primary", "latency", "--yes"],
        expected: { stopTarget: 1.5 },
      },
    ])("passes $expected to runWizard", async ({ argv, expected }) => {
      const program = createRunnableProgram();
      mockRunWizard.mockResolvedValue(createWizardResult());
      mockScaffold.mockReturnValue(createScaffoldResult());
      stubWrite(process.stdout);
      stubWrite(process.stderr);

      await program.parseAsync(initArgv(...argv));

      expect(mockRunWizard).toHaveBeenCalledWith(expect.objectContaining(expected));
    });
  });

  describe("when --stop-target receives NaN", () => {
    it("rejects with a usage error", async () => {
      const program = createRunnableProgram({ exitOverride: "all", silent: true });

      const parsing = program.parseAsync(initArgv("--stop-target", "not-a-number"));

      await expect(parsing).rejects.toThrow(/stop-target/);
    });
  });

  describe("when --stop-target receives trailing garbage or a non-finite value", () => {
    it.each([
      { value: "1.5x", why: "trailing garbage" },
      { value: "Infinity", why: "positive infinity" },
      { value: "-Infinity", why: "negative infinity" },
    ])("rejects $value ($why) with a usage error naming --stop-target", async ({ value }) => {
      const program = createRunnableProgram({ exitOverride: "all", silent: true });

      const parsing = program.parseAsync(initArgv("--stop-target", value));

      await expect(parsing).rejects.toThrow(/stop-target/);
    });
  });

  describe("when --stop-max-iterations receives a non-positive-integer", () => {
    it.each([
      { value: "abc", why: "non-numeric" },
      { value: "-1", why: "negative" },
      { value: "1.5", why: "non-integer" },
    ])("rejects $value ($why) with a usage error", async ({ value }) => {
      const program = createRunnableProgram({ exitOverride: "all", silent: true });

      const parsing = program.parseAsync(initArgv("--stop-max-iterations", value));

      await expect(parsing).rejects.toThrow(/stop-max-iterations/);
    });
  });

  describe("init does not share config-command flags", () => {
    it.each([
      { flag: "--config", value: "gymrat.json", origin: "config-reading commands" },
      { flag: "--samples", value: "5", origin: "bench-running commands" },
      { flag: "--timeout", value: "300", origin: "bench-running commands" },
    ])("rejects $flag, which belongs to $origin", async ({ flag, value }) => {
      const program = createRunnableProgram({ exitOverride: "all" });

      await expect(program.parseAsync(initArgv(flag, value))).rejects.toThrow(
        new RegExp(`unknown option '${flag}'`),
      );
    });
  });
});
