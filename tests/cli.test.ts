import { Command } from "commander";
import { afterEach, vi } from "vitest";

import { createProgram } from "../src/cli.js";
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

afterEach(() => {
  vi.clearAllMocks();
});

describe("createProgram", () => {
  it("returns a Command instance", () => {
    // Arrange
    const program = createProgram();

    // Act & Assert
    expect(program).toBeInstanceOf(Command);
  });

  describe("compare command", () => {
    describe("when valid positional arguments provided", () => {
      it("parses two positional arguments", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { compareMock } = await setupMocks();

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

        // Assert - should not throw and should call compare
        expect(compareMock).toHaveBeenCalled();
      });

      it("extracts label and ref from label=ref format", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { compareMock } = await setupMocks("report text", {
          bench: "my-bench",
          adapter: "metric-lines",
          samples: 10,
          timeoutSeconds: 1800,
        });

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "baseline=main",
          "candidate=branch",
        ]);

        // Assert - should parse labels from positionals
        expect(compareMock).toHaveBeenCalledWith(
          expect.objectContaining({
            oldTarget: "main",
            newTarget: "branch",
            oldLabel: "baseline",
            newLabel: "candidate",
          }),
        );
      });

      it("uses positional as ref when no label prefix", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { compareMock } = await setupMocks("report text", {
          bench: "my-bench",
          adapter: "metric-lines",
          samples: 10,
          timeoutSeconds: 1800,
        });

        // Act
        await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

        // Assert - should use positionals as refs with undefined labels
        expect(compareMock).toHaveBeenCalledWith(
          expect.objectContaining({
            oldTarget: "main",
            newTarget: "branch",
            oldLabel: undefined,
            newLabel: undefined,
          }),
        );
      });
    });

    describe("when flags provided", () => {
      it("parses --bench flag and passes to resolveConfig", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--bench",
          "my-bench",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(
          expect.objectContaining({ bench: "my-bench" }),
        );
      });

      it("parses --prepare flag and passes to resolveConfig", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--prepare",
          "setup.sh",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(
          expect.objectContaining({ prepare: "setup.sh" }),
        );
      });

      it("parses --adapter flag and passes to resolveConfig", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--adapter",
          "mitata",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(
          expect.objectContaining({ adapter: "mitata" }),
        );
      });

      it("parses --samples as number", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--samples",
          "100",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining({ samples: 100 }));
      });

      it("parses --timeout as number", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--timeout",
          "5000",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(expect.objectContaining({ timeout: 5000 }));
      });

      it("parses --config flag and passes to resolveConfig", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--config",
          "gymrat.json",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(
          expect.objectContaining({ config: "gymrat.json" }),
        );
      });

      it("passes all flags together to resolveConfig", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        const { resolveConfigMock } = await setupMocks();

        // Act
        await program.parseAsync([
          "node",
          "cli.js",
          "compare",
          "main",
          "branch",
          "--bench",
          "my-bench",
          "--prepare",
          "setup.sh",
          "--adapter",
          "mitata",
          "--samples",
          "50",
          "--timeout",
          "3000",
          "--config",
          "config.json",
        ]);

        // Assert
        expect(resolveConfigMock).toHaveBeenCalledWith(
          expect.objectContaining({
            bench: "my-bench",
            prepare: "setup.sh",
            adapter: "mitata",
            samples: 50,
            timeout: 3000,
            config: "config.json",
          }),
        );
      });
    });

    describe("when unknown flag provided", () => {
      it("throws CommanderError for unknown flag", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();

        // Act & Assert
        await expect(
          program.parseAsync(["node", "cli.js", "compare", "main", "branch", "--bogus", "value"]),
        ).rejects.toThrow();
      });
    });

    describe("when insufficient positionals", () => {
      it("throws CommanderError when only one positional provided", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();

        // Act & Assert
        await expect(program.parseAsync(["node", "cli.js", "compare", "main"])).rejects.toThrow();
      });

      it("throws CommanderError when no positionals provided", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();

        // Act & Assert
        await expect(program.parseAsync(["node", "cli.js", "compare"])).rejects.toThrow();
      });
    });

    describe("when --help requested", () => {
      it("displays help output", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();

        // Act & Assert
        await expect(program.parseAsync(["node", "cli.js", "compare", "--help"])).rejects.toThrow();
        // Note: --help throws when exitOverride is used
      });
    });

    describe("on successful compare", () => {
      it("exits with code 0", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        await setupMocks("Success");

        // Act
        const result = await program.parseAsync(["node", "cli.js", "compare", "main", "branch"]);

        // Assert - should complete without throwing
        expect(result).toBeDefined();
      });
    });

    describe("on compare error", () => {
      it("exits with code 1 when compare fails", async () => {
        // Arrange
        const program = createProgram();
        program.exitOverride();
        await setupMocks(new Error("Compare failed"));

        // Act & Assert
        await expect(
          program.parseAsync(["node", "cli.js", "compare", "main", "branch"]),
        ).rejects.toThrow();
      });
    });
  });
});
