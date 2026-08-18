import { describe, expect, it, vi, beforeEach } from "vitest";

import type { WarnSink } from "../../src/adapters/types.js";
import { AdapterError } from "../../src/adapters/types.js";
import type { ConfigKinds, ConfigMetrics } from "../../src/config.js";
import type { BenchSectionInput } from "../../src/doctor/bench.js";
import { buildBenchSection } from "../../src/doctor/bench.js";
import { GymratError } from "../../src/errors.js";
import { createMockAdapter } from "../fixtures/adapter.js";
import { createExecResult, createExecTimeout } from "../fixtures/exec.js";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("../../src/exec.js", () => ({
  exec: vi.fn(),
}));

vi.mock("../../src/adapters/index.js", () => ({
  getAdapter: vi.fn(),
}));

// Re-import mocked modules so we can control them
const { exec: mockExec } = await import("../../src/exec.js");
const { getAdapter: mockGetAdapter } = await import("../../src/adapters/index.js");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function defaultInput(overrides: Partial<BenchSectionInput> = {}): BenchSectionInput {
  return {
    bench: "node bench.js",
    adapter: "metric-lines",
    timeoutSeconds: 30,
    primary: "geomean",
    repoRoot: "/project",
    ...overrides,
  };
}

function benchExecResult(
  overrides: Parameters<typeof createExecResult>[0] = {},
): ReturnType<typeof createExecResult> {
  return createExecResult({ stdout: "METRIC latency=42\n", ...overrides });
}

function benchAdapter(
  overrides: Parameters<typeof createMockAdapter>[0] = {},
): ReturnType<typeof createMockAdapter> {
  return createMockAdapter({
    parse(_stdout: string, _warn?: WarnSink) {
      return { latency: 42 };
    },
    ...overrides,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.resetAllMocks();
});

// ---------------------------------------------------------------------------
// Behavior 1: No resolvable bench command
// ---------------------------------------------------------------------------

describe("buildBenchSection", () => {
  describe("when bench command is not resolvable", () => {
    it("returns a FAIL check with a hint naming the flag and config key", async () => {
      const section = await buildBenchSection(defaultInput({ bench: undefined }));

      expect(section.checks).toHaveLength(1);
      const check = section.checks[0];
      expect(check).toBeDefined();
      if (!check) return;
      expect(check.status).toBe("fail");
      expect(check.detail).toMatch(/bench/i);
      expect(check.hint).toMatch(/--bench/);
      expect(check.hint).toMatch(/bench/);
    });

    it("does not call exec", async () => {
      await buildBenchSection(defaultInput({ bench: undefined }));

      expect(mockExec).not.toHaveBeenCalled();
    });
  });

  // ---------------------------------------------------------------------------
  // Behavior 2: Unknown adapter name
  // ---------------------------------------------------------------------------

  describe("when the adapter name is unknown", () => {
    it("returns a FAIL check carrying getAdapter's error message and hint", async () => {
      const error = new GymratError(
        'Unknown adapter "bogus".',
        "valid adapters are: metric-lines, mitata",
      );
      vi.mocked(mockGetAdapter).mockImplementation(() => {
        throw error;
      });

      const section = await buildBenchSection(defaultInput({ adapter: "bogus" }));

      expect(section.checks).toHaveLength(1);
      const check = section.checks[0];
      expect(check).toBeDefined();
      if (!check) return;
      expect(check.status).toBe("fail");
      expect(check.detail).toMatch(/Unknown adapter/);
      expect(check.hint).toBe("valid adapters are: metric-lines, mitata");
    });

    it("does not call exec when the adapter is unknown", async () => {
      vi.mocked(mockGetAdapter).mockImplementation(() => {
        throw new GymratError("Unknown adapter 'bogus'");
      });

      await buildBenchSection(defaultInput({ adapter: "bogus" }));

      expect(mockExec).not.toHaveBeenCalled();
    });
  });

  // ---------------------------------------------------------------------------
  // Behavior 3: Bench command execution
  // ---------------------------------------------------------------------------

  describe("when the bench command runs", () => {
    it("calls exec with the command, cwd = repoRoot, and timeout", async () => {
      const adapter = benchAdapter();
      vi.mocked(mockGetAdapter).mockReturnValue(adapter);
      vi.mocked(mockExec).mockResolvedValue(benchExecResult());

      await buildBenchSection(
        defaultInput({ bench: "node bench.js", repoRoot: "/my/repo", timeoutSeconds: 60 }),
      );

      expect(mockExec).toHaveBeenCalledWith(
        "node bench.js",
        expect.objectContaining({
          cwd: "/my/repo",
          timeoutMs: 60_000,
        }),
      );
    });

    describe("when exec returns non-zero exit code", () => {
      it("returns a FAIL check with the exit code and stderr excerpt", async () => {
        vi.mocked(mockGetAdapter).mockReturnValue(benchAdapter());
        vi.mocked(mockExec).mockResolvedValue(
          benchExecResult({ exitCode: 1, stderr: "Error: something broke" }),
        );

        const section = await buildBenchSection(defaultInput());

        const failChecks = section.checks.filter((c) => c.status === "fail");
        expect(failChecks.length).toBeGreaterThanOrEqual(1);
        const check = failChecks[0];
        expect(check).toBeDefined();
        if (!check) return;
        expect(check.detail).toContain("1");
        expect(check.detail).toMatch(/something broke/);
      });
    });

    describe("when exec returns a timeout", () => {
      it("returns a FAIL check naming the limit and the flag/config key", async () => {
        vi.mocked(mockGetAdapter).mockReturnValue(benchAdapter());
        vi.mocked(mockExec).mockResolvedValue(createExecTimeout({ timeoutMs: 30_000 }));

        const section = await buildBenchSection(defaultInput({ timeoutSeconds: 30 }));

        const failChecks = section.checks.filter((c) => c.status === "fail");
        expect(failChecks.length).toBeGreaterThanOrEqual(1);
        const check = failChecks[0];
        expect(check).toBeDefined();
        if (!check) return;
        expect(check.detail).toMatch(/30/);
        expect(check.hint).toMatch(/timeout/i);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Behavior 4: Parse result handling
  // ---------------------------------------------------------------------------

  describe("when the bench command exits successfully", () => {
    describe("when adapter.parse throws AdapterError", () => {
      it("returns a FAIL check carrying the adapter's message", async () => {
        const adapter = benchAdapter({
          parse() {
            throw new AdapterError("No usable metrics found in output");
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult({ stdout: "garbage" }));

        const section = await buildBenchSection(defaultInput());

        const failChecks = section.checks.filter((c) => c.status === "fail");
        expect(failChecks.length).toBeGreaterThanOrEqual(1);
        const check = failChecks[0];
        expect(check).toBeDefined();
        if (!check) return;
        expect(check.detail).toMatch(/No usable metrics/);
      });
    });

    describe("when adapter.parse succeeds", () => {
      it("returns an OK check reporting the metric count and names", async () => {
        const adapter = benchAdapter({
          parse() {
            return { latency: 42, throughput: 100 };
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult());

        const section = await buildBenchSection(defaultInput());

        const okChecks = section.checks.filter((c) => c.status === "ok");
        expect(okChecks.length).toBeGreaterThanOrEqual(1);
        const check = okChecks[0];
        expect(check).toBeDefined();
        if (!check) return;
        expect(check.detail).toContain("2");
        expect(check.detail).toMatch(/latency/);
        expect(check.detail).toMatch(/throughput/);
      });

      it("collects adapter warnings into the check detail", async () => {
        const adapter = benchAdapter({
          parse(_stdout: string, warn?: WarnSink) {
            warn?.("Skipped line 3: unrecognized format");
            return { latency: 42 };
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult());

        const section = await buildBenchSection(defaultInput());

        const detail = section.checks.map((c) => c.detail).join(" ");
        expect(detail).toMatch(/Skipped line 3/);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Behavior 5: Post-parse cross-checks
  // ---------------------------------------------------------------------------

  describe("post-parse cross-checks", () => {
    describe("when primary names a metric not in the parse output", () => {
      it("returns a FAIL check when primary is not geomean and not in parsed metrics", async () => {
        const adapter = benchAdapter({
          parse() {
            return { latency: 42 };
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult());

        const section = await buildBenchSection(defaultInput({ primary: "throughput" }));

        const failChecks = section.checks.filter((c) => c.status === "fail");
        expect(failChecks.length).toBeGreaterThanOrEqual(1);
        const check = failChecks.find((c) => c.detail.includes("throughput"));
        expect(check).toBeDefined();
        if (!check) return;
        expect(check.detail).toMatch(/primary/i);
      });
    });

    describe("when primary is geomean", () => {
      it("does not fail even though geomean is not in parsed metrics", async () => {
        const adapter = benchAdapter({
          parse() {
            return { latency: 42 };
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult());

        const section = await buildBenchSection(defaultInput({ primary: "geomean" }));

        const failChecks = section.checks.filter((c) => c.status === "fail");
        expect(failChecks).toHaveLength(0);
      });
    });

    describe("when config metrics reference names the parse did not produce", () => {
      it("returns a WARN check listing the missing metric names", async () => {
        const adapter = benchAdapter({
          parse() {
            return { latency: 42 };
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult());

        const section = await buildBenchSection(
          defaultInput({
            metrics: {
              missing_metric: { direction: "lower", gating: true },
            } satisfies ConfigMetrics,
          }),
        );

        const warnChecks = section.checks.filter((c) => c.status === "warn");
        expect(warnChecks.length).toBeGreaterThanOrEqual(1);
        const check = warnChecks.find((c) => c.detail.includes("missing_metric"));
        expect(check).toBeDefined();
      });
    });

    describe("when config kinds reference kinds no parsed metric reports", () => {
      it("returns a WARN check listing the unused kind names", async () => {
        const adapter = benchAdapter({
          parse() {
            return { latency: 42 };
          },
          defaults() {
            return { direction: "lower" as const, kind: "time" };
          },
        });
        vi.mocked(mockGetAdapter).mockReturnValue(adapter);
        vi.mocked(mockExec).mockResolvedValue(benchExecResult());

        const section = await buildBenchSection(
          defaultInput({
            kinds: {
              memory: { gating: true },
            } satisfies ConfigKinds,
          }),
        );

        const warnChecks = section.checks.filter((c) => c.status === "warn");
        expect(warnChecks.length).toBeGreaterThanOrEqual(1);
        const check = warnChecks.find((c) => c.detail.includes("memory"));
        expect(check).toBeDefined();
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Behavior 6: Skip
  // ---------------------------------------------------------------------------

  describe("when --no-bench was passed", () => {
    it("returns a section with a single OK check explaining the skip", async () => {
      const section = await buildBenchSection(defaultInput({ noBench: true }));

      expect(section.checks).toHaveLength(1);
      const check = section.checks[0];
      expect(check).toBeDefined();
      if (!check) return;
      expect(check.status).toBe("ok");
      expect(check.detail).toMatch(/skip/i);
      expect(mockExec).not.toHaveBeenCalled();
    });
  });

  describe("when the config section already failed", () => {
    it("returns a section with a single OK check explaining the skip", async () => {
      const section = await buildBenchSection(defaultInput({ configFailed: true }));

      expect(section.checks).toHaveLength(1);
      const check = section.checks[0];
      expect(check).toBeDefined();
      if (!check) return;
      expect(check.status).toBe("ok");
      expect(check.detail).toMatch(/skip|config/i);
      expect(mockExec).not.toHaveBeenCalled();
    });
  });

  // ---------------------------------------------------------------------------
  // Adapter validation before skip
  // ---------------------------------------------------------------------------

  describe("when the adapter name is invalid under skip conditions", () => {
    it.each([
      { label: "--no-bench", overrides: { noBench: true } },
      { label: "configFailed", overrides: { configFailed: true } },
    ])(
      "returns a FAIL check for unknown adapter even when $label is set",
      async ({ overrides }) => {
        const error = new GymratError(
          'Unknown adapter "bogus".',
          "valid adapters are: metric-lines, mitata",
        );
        vi.mocked(mockGetAdapter).mockImplementation(() => {
          throw error;
        });

        const section = await buildBenchSection(defaultInput({ adapter: "bogus", ...overrides }));

        expect(section.checks).toHaveLength(1);
        const check = section.checks[0];
        expect(check).toBeDefined();
        if (!check) return;
        expect(check.status).toBe("fail");
        expect(check.detail).toMatch(/Unknown adapter/);
        expect(check.hint).toBe("valid adapters are: metric-lines, mitata");
      },
    );
  });

  // ---------------------------------------------------------------------------
  // Section metadata
  // ---------------------------------------------------------------------------

  describe("section metadata", () => {
    it("has the title 'Bench'", async () => {
      const section = await buildBenchSection(defaultInput({ noBench: true }));

      expect(section.title).toBe("Bench");
    });
  });
});
