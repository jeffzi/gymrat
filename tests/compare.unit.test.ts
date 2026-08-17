import fs from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";

import { compare } from "../src/compare.js";
import type { CompareOptions } from "../src/compare.js";
import { GymratError } from "../src/errors.js";
import type { ExecResult, ExecTimeoutError } from "../src/exec.js";
import type { ComparisonResult } from "../src/report/types.js";
import { CommandError } from "../src/sampling.js";
import type { CommandErrorContext } from "../src/sampling.js";
import type { InPlaceTarget, RefTarget } from "../src/targets.js";
import { REF_TARGET_HINT } from "./fixtures/constants.js";
import { createExecResult, createExecTimeout } from "./fixtures/exec.js";
import { freshRoot } from "./fixtures/scratch-repo.js";

/**
 * What every `parse` call hands back, whichever target produced the output.
 *
 * The bundled adapters raise AdapterError when a bench emits nothing parseable,
 * so compare()'s own empty-metrics guard is only reachable behind an adapter
 * that parses successfully into no metrics at all — hence the empty default.
 * Both sides of a run see the same map, so every delta is 0 and the shape of
 * the aggregation is what a test can read off the result.
 */
const parsed = vi.hoisted((): { metrics: Record<string, number> } => ({ metrics: {} }));

vi.mock("../src/adapters/index.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/adapters/index.js")>();
  return {
    ...actual,
    getAdapter: (name: string) => ({ ...actual.getAdapter(name), parse: () => parsed.metrics }),
  };
});

const cleanupSpy = vi.hoisted(() => vi.fn());
vi.mock("../src/targets.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/targets.js")>();
  return {
    ...actual,
    cleanupWorktrees: (...args: Parameters<typeof actual.cleanupWorktrees>) => {
      cleanupSpy(...args);
      return actual.cleanupWorktrees(...args);
    },
  };
});

function createRefTarget(ref = "abc123", resolvedSha = "def456"): RefTarget {
  return { kind: "ref", ref, resolvedSha };
}

function createInPlaceTarget(dir = "/projects/my-app"): InPlaceTarget {
  return { kind: "in-place", dir };
}

function createContext(overrides: Partial<CommandErrorContext> = {}): CommandErrorContext {
  return {
    phase: "bench",
    position: "old",
    label: "baseline",
    command: "npm run bench",
    target: createRefTarget(),
    dir: "/tmp/worktree-abc",
    ...overrides,
  };
}

function failedExecResult(
  overrides: Parameters<typeof createExecResult>[0] = {},
): ReturnType<typeof createExecResult> {
  return createExecResult({ exitCode: 1, stderr: "Error: benchmark crashed", ...overrides });
}

describe("CommandError", () => {
  describe("when the command fails", () => {
    it.each([
      {
        name: "exit code, ref target",
        context: { target: createRefTarget("main", "abc123") },
        failure: createExecResult({ exitCode: 2, stderr: "segfault in runner" }),
        expectedFragments: [
          "bench",
          "old",
          "baseline",
          "main",
          "/tmp/worktree-abc",
          "npm run bench",
          "exit code: 2",
          "segfault in runner",
        ],
        absentFragments: [],
      },
      {
        name: "exit code, in-place target",
        context: {
          position: "new",
          label: "candidate",
          target: createInPlaceTarget("/projects/my-app"),
          dir: "/projects/my-app",
        },
        failure: createExecResult({ exitCode: 3, stderr: "module not found" }),
        expectedFragments: [
          "bench",
          "new",
          "candidate",
          "/projects/my-app",
          "npm run bench",
          "exit code: 3",
          "module not found",
        ],
        absentFragments: [],
      },
      {
        name: "timeout, ref target",
        context: {
          phase: "prepare",
          command: "npm install",
          target: createRefTarget("feature-branch", "sha789"),
          dir: "/tmp/worktree-feature",
        },
        failure: createExecTimeout({ timeoutMs: 60000, stderr: "still installing..." }),
        expectedFragments: [
          "prepare",
          "old",
          "baseline",
          "feature-branch",
          "/tmp/worktree-feature",
          "npm install",
          "60000",
          "still installing...",
        ],
        absentFragments: ["exit code"],
      },
      {
        name: "timeout, in-place target",
        context: {
          phase: "prepare",
          position: "new",
          label: "candidate",
          command: "npm install",
          target: createInPlaceTarget("/projects/other"),
          dir: "/projects/other",
        },
        failure: createExecTimeout({ timeoutMs: 45000, stderr: "hanging on postinstall" }),
        expectedFragments: [
          "prepare",
          "new",
          "candidate",
          "/projects/other",
          "npm install",
          "45000",
          "hanging on postinstall",
        ],
        absentFragments: [],
      },
    ] satisfies {
      name: string;
      context: Partial<CommandErrorContext>;
      failure: ExecResult | ExecTimeoutError;
      expectedFragments: string[];
      absentFragments: string[];
    }[])("includes the relevant fields in the message ($name)", (testCase) => {
      const ctx = createContext(testCase.context);

      const error = new CommandError(ctx, testCase.failure);

      for (const fragment of testCase.expectedFragments) {
        expect.soft(error.message).toContain(fragment);
      }
      for (const fragment of testCase.absentFragments) {
        expect.soft(error.message).not.toContain(fragment);
      }
    });
  });

  describe("when phase is bench", () => {
    it("includes the 1-indexed sample number in the message", () => {
      const ctx = createContext({ sample: 3 });
      const failure = failedExecResult();

      const error = new CommandError(ctx, failure);

      expect(error.message).toContain("sample 3");
    });
  });

  describe("when target is a ref", () => {
    it("does not include the hint text in the message", () => {
      const ctx = createContext();
      const failure = failedExecResult();

      const error = new CommandError(ctx, failure);

      expect(error.message).not.toContain("hint:");
      expect(error.message).not.toContain("the worktree only contains files tracked at this ref");
    });

    it("carries the worktree hint", () => {
      const ctx = createContext({ target: createRefTarget() });
      const failure = failedExecResult();

      const error = new CommandError(ctx, failure);

      expect(error.hint).toBe(REF_TARGET_HINT);
    });
  });

  describe("when target is in-place", () => {
    it("carries no hint", () => {
      const ctx = createContext({ target: createInPlaceTarget() });
      const failure = failedExecResult();

      const error = new CommandError(ctx, failure);

      expect(error.hint).toBeUndefined();
    });
  });

  describe("when stderr and stdout are both non-empty", () => {
    it("shows both under labeled separators", () => {
      const ctx = createContext();
      const failure = failedExecResult({
        stderr: "error output here",
        stdout: "normal output here",
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("--- stderr ---");
      expect.soft(error.message).toContain("error output here");
      expect.soft(error.message).toContain("--- stdout ---");
      expect(error.message).toContain("normal output here");
    });
  });

  describe("when only stdout is non-empty", () => {
    it("shows stdout without separators", () => {
      const ctx = createContext();
      const failure = failedExecResult({ stderr: "", stdout: "bench output here" });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("bench output here");
      expect.soft(error.message).not.toContain("--- stderr ---");
      expect(error.message).not.toContain("--- stdout ---");
    });
  });

  describe("when a single stream was truncated", () => {
    it("shows the stream under a labeled separator with the total byte count", () => {
      const ctx = createContext();
      const failure = failedExecResult({
        stderr: "partial output",
        stdout: "",
        stderrBytes: 128_000_000,
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("--- stderr (truncated, 128000000 bytes total) ---");
      expect(error.message).toContain("partial output");
    });
  });

  describe("when both streams were truncated", () => {
    it("shows both under labeled separators with their total byte counts", () => {
      const ctx = createContext();
      const failure = failedExecResult({
        stderr: "err chunk",
        stdout: "out chunk",
        stderrBytes: 200_000_000,
        stdoutBytes: 100_000_000,
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("--- stderr (truncated, 200000000 bytes total) ---");
      expect.soft(error.message).toContain("err chunk");
      expect.soft(error.message).toContain("--- stdout (truncated, 100000000 bytes total) ---");
      expect(error.message).toContain("out chunk");
    });
  });

  describe("when stderr and stdout are both empty", () => {
    it("does not include any output separator or body", () => {
      const ctx = createContext();
      const failure = failedExecResult({ stderr: "", stdout: "" });

      const error = new CommandError(ctx, failure);

      expect(error.message).not.toContain("--- stderr ---");
      expect(error.message).not.toContain("--- stdout ---");
    });
  });
});

/**
 * Two sibling directories holding no files, usable as in-place targets.
 *
 * In-place targets need not live in a git repository, so a bare temp tree is
 * enough to drive a whole comparison run.
 */
function createInPlaceDirs(): { root: string; cleanup: () => void } {
  const root = freshRoot("gymrat-unit-");
  fs.mkdirSync(path.join(root, "old"));
  fs.mkdirSync(path.join(root, "new"));
  return { root, cleanup: () => {} };
}

/**
 * Run a whole comparison over two throwaway in-place directories.
 *
 * The directories are cleaned up and the working directory restored whether the
 * run settles or throws, so a caller may `.catch()` the rejection and assert on
 * it without a try/finally of its own.
 */
async function runCompare(overrides: Partial<CompareOptions> = {}): Promise<ComparisonResult> {
  const dirs = createInPlaceDirs();
  const savedCwd = process.cwd();

  try {
    process.chdir(dirs.root);

    const options: CompareOptions = {
      baseline: { target: "old" },
      candidates: [{ target: "new" }],
      bench: "echo benched",
      adapter: "metric-lines",
      samples: 1,
      timeoutSeconds: 10,
      ...overrides,
    };

    return await compare(options);
  } finally {
    process.chdir(savedCwd);
    dirs.cleanup();
  }
}

/** The sole candidate of a comparison, or a failure. */
function onlyCandidate(result: ComparisonResult): ComparisonResult["candidates"][number] {
  const [candidate] = result.candidates;
  if (!candidate) throw new Error("expected one candidate in the comparison result");
  return candidate;
}

describe("compare", () => {
  afterEach(() => {
    parsed.metrics = {};
  });

  describe("when the run produces no metrics at all", () => {
    it("rejects with a GymratError so the CLI reports it as a gymrat failure", async () => {
      const error = await runCompare().catch((cause: unknown) => cause);

      expect.soft(error).toBeInstanceOf(GymratError);
      expect(error).toHaveProperty("message", "No metrics found in benchmark output");
    });
  });

  describe("when metric names carry a dotted prefix", () => {
    beforeEach(() => {
      // metric-lines reports no kind, so every metric lands in "other" and the
      // group is whatever precedes the first dot of the metric's own name.
      parsed.metrics = { "decode.time": 100, "decode.alloc": 50, warmup: 10 };
    });

    it("gives the candidate one aggregate per kind, grouped by the dotted prefix", async () => {
      const candidate = onlyCandidate(await runCompare());

      expect.soft(candidate.kinds.map((kind) => kind.kind)).toStrictEqual(["other"]);
      expect.soft(candidate.kinds[0]?.groups.map((group) => group.group)).toStrictEqual(["decode"]);
      expect.soft(candidate.kinds[0]?.geomean.n).toBe(3);
      expect(candidate.kinds[0]?.groups[0]?.geomean.n).toBe(2);
    });

    describe("and the config turns gating off for their kind", () => {
      it("reports the kind as non-gating with no gated geomean", async () => {
        const candidate = onlyCandidate(
          await runCompare({ configKinds: { other: { gating: false } } }),
        );

        expect(candidate.kinds[0]?.gatedGeomean).toBeUndefined();
      });
    });
  });

  describe("when a metric's median is zero", () => {
    beforeEach(() => {
      parsed.metrics = { latency: 0 };
    });

    it("reports no spread rather than a measured ± 0%", async () => {
      const result = await runCompare({ samples: 2 });

      const metric = result.metrics["latency"];
      expect(metric).toBeDefined();
      expect.soft(metric!.baselineSpread).toBeUndefined();
      expect(metric!.candidates[0]!.spread).toBeUndefined();
    });
  });

  describe("cleanup sweep", () => {
    it("calls cleanupWorktrees exactly once on the success path", async () => {
      parsed.metrics = { latency: 100 };
      cleanupSpy.mockClear();

      await runCompare();

      expect(cleanupSpy).toHaveBeenCalledOnce();
    });
  });
});
