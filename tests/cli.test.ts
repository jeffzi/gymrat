import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

import { Command } from "commander";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AdapterError } from "../src/adapters/index.js";
import { createProgram, formatCliError } from "../src/cli.js";
import type { ResolvedConfig } from "../src/config.js";

// Mock the compare module
vi.mock("../src/compare.js", () => ({
  compare: vi.fn(),
}));

// Mock the config module - we'll set up specific return values per test
vi.mock("../src/config.js", () => ({
  resolveConfig: vi.fn(),
}));

// Helper to get properly typed mocks and configure them
async function setupMocks(
  compareMockReturn?: string | Error,
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
    ...resolveConfigMockReturn,
  };

  typedConfigMock.mockReturnValue(resolvedConfig);
  if (compareMockReturn instanceof Error) {
    typedCompareMock.mockRejectedValue(compareMockReturn);
  } else {
    typedCompareMock.mockResolvedValue(compareMockReturn ?? "OK");
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
          const { compareMock } = await setupMocks("report text");

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

    describe("when the resolved config carries per-metric overrides", () => {
      it("passes them to compare so the overrides reach the report", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const metrics = {
          "decode/time": { direction: "higher" as const, gating: false, exact: true },
        };
        const { compareMock } = await setupMocks("report text", { metrics });

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

        // Assert
        expect(compareMock).toHaveBeenCalledWith(
          expect.objectContaining({ configMetrics: metrics }),
        );
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
      it("throws CommanderError when only one positional provided", async () => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();

        // Act & Assert
        await expect(program.parseAsync(["node", "cli.js", "compare", "main"])).rejects.toThrow(
          /missing required argument/,
        );
      });

      it("throws CommanderError when no positionals provided", async () => {
        // Arrange
        const program = createProgramWithSubcommandOverrides();

        // Act & Assert
        await expect(program.parseAsync(["node", "cli.js", "compare"])).rejects.toThrow(
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
      it("writes the rendered report to stdout", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        await setupMocks("report text");
        const writeSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

        // Assert
        expect(writeSpy).toHaveBeenCalledWith("report text\n");
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
  });

  describe("entry point", () => {
    it("executes CLI when invoked through symlink", async () => {
      // Arrange
      const tmpDir = mkdtempSync(join(tmpdir(), "gymrat-cli-test-"));
      const cliPath = resolve("src/cli.ts");
      const symlinkPath = join(tmpDir, "cli-symlink.ts");

      try {
        // Create a symlink to the CLI entry point
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
        // Clean up
        rmSync(tmpDir, { recursive: true, force: true });
      }
    });
  });
});
