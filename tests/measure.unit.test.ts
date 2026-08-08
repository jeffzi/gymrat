import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { beforeEach, describe, expect, it, vi } from "vitest";

import { messageOf } from "../src/errors.js";
import { measure } from "../src/measure.js";
import type { MeasureOptions } from "../src/measure.js";
import type { MeasurementResult } from "../src/report/types.js";
import { CommandError } from "../src/sampling.js";
import type { CleanupResult } from "../src/targets.js";
import { metricMeta } from "./fixtures/comparison-result.js";
import { metricRecord } from "./fixtures/metrics.js";

/**
 * One metric map per bench run, handed out in order.
 *
 * Driving the values through the adapter rather than through the bench script's
 * stdout is what lets a test state the exact medians and spreads it expects
 * without a shell script that counts its own invocations.
 */
const benchOutput = vi.hoisted((): { queue: Record<string, number>[]; runs: number } => ({
  queue: [],
  runs: 0,
}));

vi.mock("../src/adapters/index.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/adapters/index.js")>();
  return {
    ...actual,
    getAdapter: (name: string) => ({
      ...actual.getAdapter(name),
      parse: () => benchOutput.queue[benchOutput.runs++] ?? {},
    }),
  };
});

/**
 * A cleanup outcome to return instead of sweeping for real.
 *
 * An in-place target creates no worktree, so the only way a unit test can reach
 * the removed/left-behind/prune reporting is to state what the sweep found.
 */
const cleanupStub = vi.hoisted((): { result?: CleanupResult } => ({}));

vi.mock("../src/targets.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/targets.js")>();
  return {
    ...actual,
    cleanupWorktrees: (...args: Parameters<typeof actual.cleanupWorktrees>) =>
      cleanupStub.result ?? actual.cleanupWorktrees(...args),
  };
});

/**
 * Run `fn` with an empty `target/` directory as the working directory's only child.
 *
 * In-place targets need not live in a git repository, so a bare temp tree is
 * enough to drive a whole measurement run. The tree is removed and the working
 * directory restored whether `fn` settles or throws.
 */
async function withTargetDir<T>(
  fn: (dirs: { root: string; targetDir: string }) => Promise<T>,
): Promise<T> {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-measure-"));
  const targetDir = path.join(root, "target");
  fs.mkdirSync(targetDir);
  const savedCwd = process.cwd();

  try {
    process.chdir(root);
    return await fn({ root, targetDir });
  } finally {
    process.chdir(savedCwd);
    fs.rmSync(root, { recursive: true, force: true });
  }
}

function measureOptions(overrides: Partial<MeasureOptions> = {}): MeasureOptions {
  return {
    target: { target: "target" },
    bench: "echo benched",
    adapter: "metric-lines",
    samples: 1,
    timeoutSeconds: 10,
    ...overrides,
  };
}

async function runMeasure(overrides: Partial<MeasureOptions> = {}): Promise<MeasurementResult> {
  return withTargetDir(async () => measure(measureOptions(overrides)));
}

async function measureQueue(
  queue: Record<string, number>[],
  overrides: Partial<MeasureOptions> = {},
): Promise<MeasurementResult> {
  benchOutput.queue = queue;
  return runMeasure({ samples: queue.length, ...overrides });
}

/** Whatever a failed run rejected with, cleanup wrapping and all. */
async function measureFailure(overrides: Partial<MeasureOptions> = {}): Promise<unknown> {
  return runMeasure(overrides).then(
    () => undefined,
    (cause: unknown) => cause,
  );
}

async function measureCommandError(overrides: Partial<MeasureOptions> = {}): Promise<CommandError> {
  const error = await measureFailure(overrides);
  if (!(error instanceof CommandError)) {
    throw new Error(`expected the run to reject with a CommandError, got: ${messageOf(error)}`);
  }
  return error;
}

describe("measure", () => {
  beforeEach(() => {
    benchOutput.queue = [{ latency: 100 }];
    benchOutput.runs = 0;
    cleanupStub.result = undefined;
  });

  describe("when the bench reports metrics over several samples", () => {
    const queue: Record<string, number>[] = [
      { latency: 90, alloc: 10 },
      { latency: 110 },
      { latency: 100, alloc: 30 },
    ];

    it("returns the target's label, sample count and adapter", async () => {
      const result = await measureQueue(queue);

      expect.soft(result.label).toBe("target");
      expect.soft(result.samples).toBe(3);
      expect(result.adapter).toBe("metric-lines");
    });

    it("gives every metric the target reported a median and a spread", async () => {
      const result = await measureQueue(queue);

      expect(result.metrics).toStrictEqual(
        metricRecord({
          latency: { median: 100, spread: 10, meta: metricMeta("latency") },
          alloc: { median: 20, spread: 50, meta: metricMeta("alloc") },
        }),
      );
    });
  });

  describe("when a metric has no run-to-run jitter to report", () => {
    it.each([
      { case: "a single sample leaves nothing to compare against", queue: [{ latency: 100 }] },
      { case: "a zero median has no scale", queue: [{ latency: 0 }, { latency: 0 }] },
    ])("reports no spread because $case", async ({ queue }) => {
      const result = await measureQueue(queue);

      expect(result.metrics["latency"]?.spread).toBeUndefined();
    });
  });

  describe("when the config overrides a metric", () => {
    it("resolves the metric's metadata from the config and the adapter defaults", async () => {
      const result = await measureQueue([{ latency: 100 }], {
        configMetrics: { latency: { direction: "higher", gating: false, exact: true } },
      });

      expect(result.metrics["latency"]?.meta).toStrictEqual(
        metricMeta("latency", { direction: "higher", gating: false, exact: true }),
      );
    });
  });

  describe("when the config carries a kinds section", () => {
    it("echoes it onto the result so the report can name the line behind a verdict", async () => {
      const configKinds = { other: { gating: false } };

      const result = await measureQueue([{ latency: 100 }], { configKinds });

      expect(result.configKinds).toStrictEqual(configKinds);
    });
  });

  describe("labelling", () => {
    it("labels the result with the explicit label when one is given", async () => {
      const result = await runMeasure({ target: { target: "target", label: "my-build" } });

      expect(result.label).toBe("my-build");
    });
  });

  describe("when a prepare command is configured", () => {
    it("runs it once, before the first sample", async () => {
      const append = (word: string): string =>
        `node -e "require('fs').appendFileSync('steps.log','${word}\\n')"`;

      const log = await withTargetDir(async ({ targetDir }) => {
        benchOutput.queue = [{ latency: 100 }, { latency: 100 }];
        await measure(
          measureOptions({
            prepare: append("prepare"),
            bench: append("bench"),
            samples: 2,
          }),
        );
        return fs.readFileSync(path.join(targetDir, "steps.log"), "utf8");
      });

      expect(log).toBe("prepare\nbench\nbench\n");
    });
  });

  describe("when a command fails", () => {
    it.each([
      {
        case: "the bench exits non-zero",
        overrides: {
          bench: `node -e "process.stderr.write('bench-boom\\n');process.exit(3)"`,
        },
        expected: ["bench", "probe", "process.exit(3)", "exit code: 3", "bench-boom", "sample 1"],
        absent: [],
      },
      {
        case: "prepare exits non-zero",
        overrides: {
          prepare: `node -e "process.stderr.write('prep-boom\\n');process.exit(4)"`,
        },
        expected: ["prepare", "probe", "process.exit(4)", "exit code: 4", "prep-boom"],
        absent: ["sample "],
      },
      {
        case: "the bench exceeds the timeout",
        overrides: {
          bench: `node -e "setTimeout(()=>{},60000)"`,
          timeoutSeconds: 0.5,
        },
        expected: ["bench", "probe", "setTimeout", "500ms", "timed out"],
        absent: ["exit code"],
      },
    ] satisfies {
      case: string;
      overrides: Partial<MeasureOptions>;
      expected: string[];
      absent: string[];
    }[])("rejects with a CommandError naming the failure when $case", async (testCase) => {
      const error = await measureCommandError({
        ...testCase.overrides,
        target: { target: "target", label: "probe" },
      });

      for (const fragment of testCase.expected) {
        expect.soft(error.message).toContain(fragment);
      }
      for (const fragment of testCase.absent) {
        expect.soft(error.message).not.toContain(fragment);
      }
      // A single target has no baseline/candidate role, so the message must not
      // claim one the way a comparison's does.
      expect(error.message).not.toMatch(/\b(old|new)\b/);
    });
  });

  describe("cleanup", () => {
    it("carries the sweep's outcome on the result", async () => {
      cleanupStub.result = {
        removed: 2,
        failures: [{ dir: "/tmp/stuck-worktree", error: "worktree is busy" }],
        pruneError: "prune refused",
      };

      const result = await runMeasure();

      expect.soft(result.worktreesRemoved).toBe(2);
      expect
        .soft(result.worktreesLeftBehind)
        .toStrictEqual([{ dir: "/tmp/stuck-worktree", error: "worktree is busy" }]);
      expect(result.worktreePruneError).toBe("prune refused");
    });

    it("names what it left behind in a failed run's error", async () => {
      cleanupStub.result = {
        removed: 0,
        failures: [{ dir: "/tmp/stranded-worktree", error: "worktree is busy" }],
        pruneError: undefined,
      };

      const error = await measureFailure({ bench: "exit 1" });

      expect.soft(messageOf(error)).toContain("left behind");
      expect(messageOf(error)).toContain("/tmp/stranded-worktree");
    });
  });

  describe("recording", () => {
    it("writes nothing to disk beyond what the bench itself wrote", async () => {
      const entries = await withTargetDir(async ({ root, targetDir }) => {
        await measure(measureOptions());
        return { root: fs.readdirSync(root), target: fs.readdirSync(targetDir) };
      });

      expect.soft(entries.target).toStrictEqual([]);
      expect(entries.root).toStrictEqual(["target"]);
    });
  });
});
