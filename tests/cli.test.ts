import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

import { Command } from "commander";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AdapterError } from "../src/adapters/index.js";
import { createProgram, formatCliError } from "../src/cli.js";
import type { CommandErrorContext, ExitFailure } from "../src/compare.js";
import { CommandError } from "../src/compare.js";
import type { ResolvedConfig } from "../src/config.js";
import { renderReport } from "../src/report/text.js";
import type { ComparisonResult } from "../src/report/types.js";
import { createComparisonResult } from "./fixtures/comparison-result.js";

// Mock the compare module — pass through CommandError so tests can construct instances
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
  const failure: ExitFailure = { exitCode: 1, stderr: "failed", stdout: "" };
  return new CommandError(context, failure);
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe("createProgram", () => {
  it("returns a Command instance", () => {
    // Arrange
    const program = createProgram();

    // Act & Assert
    expect(program).toBeInstanceOf(Command);
  });

  it("reports the version declared in package.json", () => {
    // Arrange
    const declaredVersion = readDeclaredVersion();

    // Act
    const reportedVersion = createProgram().version();

    // Assert
    expect(reportedVersion).toBe(declaredVersion);
  });

  describe("compare command", () => {
    describe("when valid positional arguments provided", () => {
      it.each([
        {
          form: "label=ref",
          positionals: ["baseline=main", "candidate=branch"],
          expected: {
            oldTarget: "main",
            newTarget: "branch",
            oldLabel: "baseline",
            newLabel: "candidate",
          },
        },
        {
          form: "bare ref",
          positionals: ["main", "branch"],
          expected: {
            oldTarget: "main",
            newTarget: "branch",
            oldLabel: undefined,
            newLabel: undefined,
          },
        },
      ])(
        "extracts targets and labels from $form positionals",
        async ({ positionals, expected }) => {
          // Arrange
          const program = createProgram();
          program.exitOverride();
          const { compareMock } = await setupMocks();

          // Act
          await program.parseAsync(["node", "cli.js", "compare", ...positionals]);

          // Assert
          expect(compareMock).toHaveBeenCalledWith(expect.objectContaining(expected));
        },
      );
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
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch", flag, value]);

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
        const program = createProgram();
        program.exitOverride();
        const { compareMock } = await setupMocks(undefined, resolved);

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

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
        const program = createProgram();
        for (const command of [program, ...program.commands]) {
          command.exitOverride();
          command.configureOutput({ writeErr: () => {} });
        }

        // Act
        const parsing = program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          flag,
          value,
        ]);

        // Assert - Commander renders the flag and the coercion reason
        await expect(parsing).rejects.toThrow(
          new RegExp(`option '${flag}[^']*'.*is invalid\\..*positive integer`),
        );
      });
    });

    describe("when unknown flag provided", () => {
      it("throws CommanderError for unknown flag", async () => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();

        // Act & Assert
        await expect(
          program.parseAsync(["node", "cli.js", "compare", "main", "branch", "--bogus", "value"]),
        ).rejects.toThrow(/unknown option '--bogus'/);
      });
    });

    describe("when insufficient positionals", () => {
      it.each([
        { description: "only one positional", args: ["main"] },
        { description: "no positionals", args: [] },
      ])("throws CommanderError when $description provided", async ({ args }) => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();

        // Act & Assert
        await expect(program.parseAsync(["node", "cli.js", "compare", ...args])).rejects.toThrow(
          /missing required argument/,
        );
      });
    });

    describe("when --help requested", () => {
      it("writes the compare usage text", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        let helpOutput = "";
        // The subcommand renders its own help, so it needs its own output config.
        for (const command of [program, ...program.commands]) {
          command.configureOutput({
            writeOut: (str) => {
              helpOutput += str;
            },
          });
        }

        // Act - --help throws when exitOverride is used
        await expect(program.parseAsync(["node", "cli.js", "compare", "--help"])).rejects.toThrow();

        // Assert
        expect(helpOutput).toContain("Usage: gymrat compare");
      });
    });

    describe("on successful compare", () => {
      it("renders the comparison data compare returned and writes it to stdout", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const result = createComparisonResult({ labels: ["main", "branch"] });
        await setupMocks(result);
        const writeSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

        // Assert
        expect(writeSpy).toHaveBeenCalledWith(`${renderReport(result)}\n`);
      });
    });

    describe("on compare error", () => {
      it("rejects when compare fails", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        await setupMocks(new Error("Compare failed"));

        // Act & Assert
        await expect(
          program.parseAsync(["node", "cli.js", "compare", "main", "branch"]),
        ).rejects.toThrow("Compare failed");
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

  it("appends a hint line when the error carries a hint", () => {
    // Arrange - ref targets produce a hint about worktree file visibility
    const error = createCommandError("ref");

    // Act
    const rendered = formatCliError(error, false);

    // Assert
    expect(rendered).toContain("\nHint: ");
  });

  it("does not append a hint line when CommandError has undefined hint", () => {
    // Arrange - in-place targets have no hint
    const error = createCommandError("in-place");

    // Act
    const rendered = formatCliError(error, false);

    // Assert
    expect(rendered).not.toContain("Hint:");
  });

  it("does not append a hint line for a plain Error without hint field", () => {
    // Arrange
    const error = new Error("git rev-parse failed");

    // Act
    const rendered = formatCliError(error, false);

    // Assert
    expect(rendered).not.toContain("Hint:");
  });

  it("styles the Hint: label with ANSI yellow+underline when useColor is true", () => {
    // Arrange
    const error = createCommandError("ref");

    // Act
    const rendered = formatCliError(error, true);

    // Assert - \x1b[33m = yellow, \x1b[4m = underline
    expect(rendered).toContain("\x1b[33m");
    expect(rendered).toContain("\x1b[4m");
    expect(rendered).toContain("Hint:");
  });

  it("renders Hint: as plain text when useColor is false", () => {
    // Arrange
    const error = createCommandError("ref");

    // Act
    const rendered = formatCliError(error, false);

    // Assert
    expect(rendered).toContain("\nHint: ");
    expect(rendered).not.toContain("\x1b[");
  });
});

describe("entry point", () => {
  it("executes CLI when invoked through symlink", async () => {
    // Arrange
    const tmpDir = mkdtempSync(join(tmpdir(), "gymrat-cli-test-"));
    const cliPath = resolve("src/cli.ts");
    const symlinkPath = join(tmpDir, "cli-symlink.ts");

    try {
      symlinkSync(cliPath, symlinkPath);

      // Act
      const execFileAsync = promisify(execFile);
      const { stdout } = await execFileAsync(
        process.execPath,
        ["--import", "tsx", symlinkPath, "compare", "--help"],
        {
          timeout: 10000,
        },
      );

      // Assert
      expect(stdout).toContain("Usage: gymrat compare");
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
