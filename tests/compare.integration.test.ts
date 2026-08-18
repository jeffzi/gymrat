import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { describe, it, expect, afterAll, beforeAll } from "vitest";

import { compare } from "../src/compare.js";
import type { CompareOptions } from "../src/compare.js";
import { renderReport } from "../src/report/text.js";
import type { ComparisonResult } from "../src/report/types.js";
import { CommandError } from "../src/sampling.js";
import { captureRejection } from "./fixtures/errors.js";
import {
  createInPlaceTargetDir,
  createScratchRepo,
  listWorktreeDirs,
  removeStrandedWorktrees,
  toShellPath,
} from "./fixtures/scratch-repo.js";
import type { ScratchRepo } from "./fixtures/scratch-repo.js";

function assertCommandError(error: Error): asserts error is CommandError {
  expect(error).toBeInstanceOf(CommandError);
}

interface BranchSetup {
  name: string;
  benchScript: string;
  prepareScript?: string;
}

function createBranch(
  repo: ReturnType<typeof createScratchRepo>,
  setup: BranchSetup,
  baseRef = "main",
) {
  execFileSync("git", ["checkout", "-b", setup.name, baseRef], {
    cwd: repo.dir,
    stdio: "pipe",
  });

  fs.writeFileSync(path.join(repo.dir, "bench.sh"), setup.benchScript);
  const scripts = ["bench.sh"];

  if (setup.prepareScript) {
    fs.writeFileSync(path.join(repo.dir, "prepare.sh"), setup.prepareScript);
    scripts.push("prepare.sh");
  }

  execFileSync("git", ["add", ...scripts], {
    cwd: repo.dir,
    stdio: "pipe",
  });

  execFileSync("git", ["commit", "-m", setup.name], {
    cwd: repo.dir,
    stdio: "pipe",
  });
}

/**
 * Create a scratch repo, change into it for the duration of `fn`, then sweep
 * any stranded worktrees and clean the repo up — even if `fn` throws.
 *
 * Centralizes the create-repo/chdir/try/cleanup scaffolding shared by every
 * self-contained integration test below.
 */
async function withScratchRepo<T>(fn: (repo: ScratchRepo) => Promise<T>): Promise<T> {
  const repo = createScratchRepo();
  try {
    process.chdir(repo.dir);
    return await fn(repo);
  } finally {
    removeStrandedWorktrees(repo);
    repo.cleanup();
  }
}

/**
 * The CompareOptions fields shared by every integration test call site, with
 * only `baseline` and `candidates` required — everything else follows the
 * standard bench/adapter/samples/timeout defaults unless overridden.
 */
function compareOptions(
  overrides: Partial<CompareOptions> & Pick<CompareOptions, "baseline" | "candidates">,
): CompareOptions {
  return {
    bench: "sh bench.sh",
    adapter: "metric-lines",
    samples: 3,
    timeoutSeconds: 10,
    ...overrides,
  };
}

/**
 * The single report line matching `predicate`, or a failure naming the whole report.
 */
function findLine(report: string, predicate: (line: string) => boolean): string {
  const line = report.split("\n").find(predicate);
  if (line === undefined) {
    throw new Error(`no matching line in report:\n${report}`);
  }
  return line;
}

/** Assert the repo lists no worktree beyond its own main directory. */
function assertWorktreesCleanedUp(repo: ReturnType<typeof createScratchRepo>) {
  expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
}

/** Generous timeout for tests whose run creates worktrees and spawns real bench processes. */
const LONG_RUN_TIMEOUT_MS = 60_000;

describe("compare – integration", () => {
  describe("when prepare command is provided", () => {
    it("runs prepare once before every bench run on both targets", async () => {
      await withScratchRepo(async (repo) => {
        // The scripts run inside throwaway worktrees, so they append to a log in
        // the main repo dir that outlives cleanup.
        const oldLog = path.join(repo.dir, "old-log.txt");
        const newLog = path.join(repo.dir, "new-log.txt");

        createBranch(repo, {
          name: "old-prep",
          benchScript: `#!/bin/sh\necho bench >> "${toShellPath(oldLog)}"\necho "METRIC latency=100"`,
          prepareScript: `#!/bin/sh\necho prepare >> "${toShellPath(oldLog)}"`,
        });

        createBranch(repo, {
          name: "new-prep",
          benchScript: `#!/bin/sh\necho bench >> "${toShellPath(newLog)}"\necho "METRIC latency=90"`,
          prepareScript: `#!/bin/sh\necho prepare >> "${toShellPath(newLog)}"`,
        });

        await compare(
          compareOptions({
            baseline: { target: "old-prep" },
            candidates: [{ target: "new-prep" }],
            prepare: "sh prepare.sh",
          }),
        );

        const readLog = (file: string) => fs.readFileSync(file, "utf-8").trim().split("\n");
        expect(readLog(oldLog)).toStrictEqual(["prepare", "bench", "bench", "bench"]);
        expect(readLog(newLog)).toStrictEqual(["prepare", "bench", "bench", "bench"]);
      });
    });
  });

  describe("when one baseline is compared against two candidates", () => {
    let repo: ReturnType<typeof createScratchRepo>;
    let benchOrder: string[];
    let result: ComparisonResult;

    beforeAll(async () => {
      repo = createScratchRepo();
      process.chdir(repo.dir);

      // The benches run inside throwaway worktrees, so they append to one log in
      // the main repo dir: a single file records the interleaving across targets.
      const log = path.join(repo.dir, "round-robin.txt");
      for (const [name, latency] of [
        ["base3", 100],
        ["candidate-a", 80],
        ["candidate-b", 120],
      ] as const) {
        createBranch(repo, {
          name,
          benchScript: `#!/bin/sh\necho ${name} >> "${toShellPath(log)}"\necho "METRIC latency=${latency}"`,
        });
      }

      result = await compare(
        compareOptions({
          baseline: { target: "base3" },
          candidates: [{ target: "candidate-a" }, { target: "candidate-b" }],
        }),
      );
      benchOrder = fs.readFileSync(log, "utf-8").trim().split("\n");
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      removeStrandedWorktrees(repo);
      repo.cleanup();
    });

    it("runs the bench once per target per round, baseline first, in argument order", () => {
      expect(benchOrder).toStrictEqual([
        "base3",
        "candidate-a",
        "candidate-b",
        "base3",
        "candidate-a",
        "candidate-b",
        "base3",
        "candidate-a",
        "candidate-b",
      ]);
    });

    it("carries the baseline once and every candidate in argument order", () => {
      expect.soft(result.baselineLabel).toBe("base3");
      expect
        .soft(result.candidates.map((candidate) => candidate.label))
        .toStrictEqual(["candidate-a", "candidate-b"]);
      expect(result.metrics["latency"]?.baselineMedian).toBe(100);
    });

    it("gives each candidate its own verdict and geomean against the shared baseline", () => {
      const latency = result.metrics["latency"];

      expect
        .soft(latency?.candidates.map((candidate) => candidate.median))
        .toStrictEqual([80, 120]);
      expect
        .soft(latency?.candidates.map((candidate) => candidate.verdict?.verdict))
        .toStrictEqual(["improved", "regressed"]);
      expect.soft(result.candidates[0]?.kinds[0]?.geomean.value).toBeLessThan(0);
      expect(result.candidates[1]?.kinds[0]?.geomean.value).toBeGreaterThan(0);
    });

    it("creates and removes one worktree per ref target", () => {
      expect.soft(result.worktreesRemoved).toBe(3);
      expect.soft(result.worktreesLeftBehind).toStrictEqual([]);
      assertWorktreesCleanedUp(repo);
    });
  });

  describe("when a candidate's bench fails in a three-target run", () => {
    it(
      "removes the worktree of every target the run created",
      async () => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: "base-fail3",
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });
          createBranch(repo, {
            name: "candidate-ok3",
            benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          });
          createBranch(repo, {
            name: "candidate-bad3",
            benchScript: '#!/bin/sh\necho "boom" >&2\nexit 1',
          });

          const failure = await captureRejection(
            compare(
              compareOptions({
                baseline: { target: "base-fail3" },
                candidates: [{ target: "candidate-ok3" }, { target: "candidate-bad3" }],
              }),
            ),
          );

          assertCommandError(failure);
          expect.soft(failure.message).toContain("candidate-bad3");
          expect.soft(failure.message).toContain("boom");
          expect.soft(failure.message).not.toContain("left behind");
          assertWorktreesCleanedUp(repo);
        });
      },
      LONG_RUN_TIMEOUT_MS,
    );
  });

  describe("when targets are plain directories rather than refs", () => {
    let repo: ReturnType<typeof createScratchRepo>;
    let result: ComparisonResult;
    let report: string;

    beforeAll(async () => {
      repo = createScratchRepo();
      process.chdir(repo.dir);

      for (const [name, latency] of [
        ["old-dir", 100],
        ["new-dir", 90],
      ] as const) {
        createInPlaceTargetDir(repo, name, `#!/bin/sh\necho "METRIC latency=${latency}"`);
      }

      result = await compare(
        compareOptions({
          baseline: { target: "old-dir" },
          candidates: [{ target: "new-dir" }],
        }),
      );
      report = renderReport(result);
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      repo.cleanup();
    });

    it("benches in place and labels each column with its directory name", () => {
      const headerRow = findLine(report, (line) => line.startsWith("metric"));
      expect.soft(headerRow).toContain("old-dir");
      expect.soft(headerRow).toContain("new-dir");
      // No worktree was created, so cleanup has nothing to report and stays
      // silent — the prune sweep is skipped, and sweeping anyway would fail
      // on targets that need not be git repositories at all.
      expect(report).not.toContain("worktrees removed");
    });

    it("returns the comparison data rather than the rendered report", () => {
      expect.soft(result.baselineLabel).toBe("old-dir");
      expect.soft(result.candidates.map((candidate) => candidate.label)).toStrictEqual(["new-dir"]);
      expect.soft(result.samples).toBe(3);
      expect.soft(result.adapter).toBe("metric-lines");
      expect.soft(result.metrics["latency"]?.baselineMedian).toBe(100);
      expect.soft(result.metrics["latency"]?.candidates[0]?.median).toBe(90);
      expect(result.worktreesRemoved).toBe(0);
    });
  });

  describe("when explicit labels are supplied", () => {
    it("uses them in the report instead of the ref names", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-labelled",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-labelled",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-labelled", label: "baseline" },
              candidates: [{ target: "new-labelled", label: "candidate" }],
            }),
          ),
        );

        const headerRow = findLine(report, (line) => line.startsWith("metric"));
        expect.soft(headerRow).toContain("baseline");
        expect.soft(headerRow).toContain("candidate");
        expect.soft(report).not.toContain("old-labelled");
        expect(report).not.toContain("new-labelled");
      });
    });
  });

  describe("when comparing refs with metric-lines adapter and different metric sets", () => {
    it("renders union of metrics from both refs with one-sided rows", async () => {
      await withScratchRepo(async (repo) => {
        createBranch(repo, {
          name: "old-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"\necho "METRIC memory=200"',
        });

        createBranch(repo, {
          name: "new-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=80"\necho "METRIC throughput=500"',
        });

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-branch" },
              candidates: [{ target: "new-branch" }],
            }),
          ),
        );

        expect.soft(report).toContain("latency");
        expect.soft(report).toContain("throughput");
        // memory is baseline-only, so its row's candidate cell is empty — a
        // trailing separator means the candidate and verdict cells were trimmed away.
        const memoryRow = findLine(report, (line) => line.startsWith("memory"));
        expect(memoryRow.endsWith("│")).toBe(true);
        assertWorktreesCleanedUp(repo);
      });
    });
  });

  describe("when a metric is named after an Object.prototype member", () => {
    it("renders a baseline-only toString metric as a one-sided row", async () => {
      await withScratchRepo(async (repo) => {
        // In-place targets keep the run to the two benches: no worktree is created.
        createInPlaceTargetDir(
          repo,
          "old-proto",
          '#!/bin/sh\necho "METRIC toString=100"\necho "METRIC latency=100"',
        );
        createInPlaceTargetDir(repo, "new-proto", '#!/bin/sh\necho "METRIC latency=90"');

        const report = renderReport(
          await compare(
            compareOptions({
              baseline: { target: "old-proto" },
              candidates: [{ target: "new-proto" }],
            }),
          ),
        );

        // The candidate bench never emitted toString, so its cell has to be empty —
        // a trailing separator means the candidate and verdict cells were trimmed
        // away. Reading the name off Object.prototype instead would fill them with
        // a value no bench produced.
        const toStringRow = findLine(report, (line) => line.startsWith("toString"));
        expect.soft(toStringRow).toContain("100");
        expect(toStringRow).toMatch(/│$/);
      });
    });
  });

  describe("when a warn sink is supplied and the bench emits a malformed METRIC line", () => {
    it("delivers the adapter's warning to the sink", async () => {
      await withScratchRepo(async (repo) => {
        createInPlaceTargetDir(
          repo,
          "old-warn",
          '#!/bin/sh\necho "METRIC foo=bar"\necho "METRIC latency=100"',
        );
        createInPlaceTargetDir(repo, "new-warn", '#!/bin/sh\necho "METRIC latency=90"');
        const warnings: string[] = [];

        await compare(
          compareOptions({
            baseline: { target: "old-warn" },
            candidates: [{ target: "new-warn" }],
            samples: 1,
            warn: (message) => {
              warnings.push(message);
            },
          }),
        );

        expect(warnings).toStrictEqual(["Failed to parse METRIC line: METRIC foo=bar"]);
      });
    });
  });

  describe("when using mitata adapter with fixture replay", () => {
    let repo: ReturnType<typeof createScratchRepo>;
    let result: ComparisonResult;

    beforeAll(async () => {
      const fixturePath = path.resolve("tests/fixtures/mitata.json");
      repo = createScratchRepo();
      process.chdir(repo.dir);
      const mitataBenchScript = `#!/bin/sh\ncat "${toShellPath(fixturePath)}"`;
      createBranch(repo, { name: "mitata-branch", benchScript: mitataBenchScript });
      createBranch(repo, { name: "mitata-branch-2", benchScript: mitataBenchScript });

      result = await compare(
        compareOptions({
          baseline: { target: "mitata-branch" },
          candidates: [{ target: "mitata-branch-2" }],
          adapter: "mitata",
        }),
      );
    });

    afterAll(() => {
      repo.cleanup();
    });

    it("parses the mitata JSON fixture into metrics keyed by benchmark and stat", () => {
      // Both branches replay the same fixture, so baseline and candidate
      // medians must match exactly — a swapped side or a wrong median would
      // make them differ.
      expect.soft(result.metrics["decode/text=digits/time"]?.baselineMedian).toBe(4.0791015625);
      expect
        .soft(result.metrics["decode/text=digits/time"]?.candidates[0]?.median)
        .toBe(4.0791015625);
      expect.soft(result.metrics["decode/text=words/time"]?.baselineMedian).toBe(7.8125);
      expect.soft(result.metrics["decode/text=words/time"]?.candidates[0]?.median).toBe(7.8125);
      expect.soft(result.metrics["encode/time"]?.baselineMedian).toBe(42.66357421875);
      expect(result.metrics["encode/time"]?.candidates[0]?.median).toBe(42.66357421875);
    });

    it("renders the parsed benchmarks in the report", () => {
      const report = renderReport(result);

      expect.soft(report).toContain("decode");
      expect.soft(report).toContain("encode");
      expect(report).toContain("time");
    });
  });

  describe("when the sample count selects the verdict method", () => {
    it.each([
      { samples: 10, newLatency: 80, expectedFooter: "Wilcoxon signed-rank" },
      { samples: 3, newLatency: 90, expectedFooter: "noise band ±(half-range × K)" },
    ])(
      "reports $expectedFooter for $samples samples",
      async ({ samples, newLatency, expectedFooter }) => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: `old-${samples}`,
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: `new-${samples}`,
            benchScript: `#!/bin/sh\necho "METRIC latency=${newLatency}"`,
          });

          const report = renderReport(
            await compare(
              compareOptions({
                baseline: { target: `old-${samples}` },
                candidates: [{ target: `new-${samples}` }],
                samples,
              }),
            ),
            { verbose: true },
          );

          expect(report).toContain(expectedFooter);
        });
      },
    );
  });

  describe("when an unstable noise threshold is configured", () => {
    /**
     * A bench whose successive runs emit 40, 50 then 60 for `latency`.
     *
     * The counter file lives in the main repo dir so it survives the throwaway
     * worktree each run happens in. Paired against a flat 100 on the other side,
     * the spread works out to a noise band of 30%: median 50, half-range 10,
     * 1.5 × 100 × (10 / 50).
     */
    const noisyBenchScript = (counterFile: string) => {
      const p = toShellPath(counterFile);
      return [
        "#!/bin/sh",
        `echo x >> "${p}"`,
        `n=$(wc -l < "${p}" | tr -d ' ')`,
        'echo "METRIC latency=$((30 + 10 * n))"',
      ].join("\n");
    };

    it.each([
      {
        unstableNoisePct: 20,
        expectedVerdict: "≈  unstable",
        expectedGeomean: "no stable metrics",
      },
      { unstableNoisePct: 40, expectedVerdict: "✓  -50.0%", expectedGeomean: "-50.0%" },
    ])(
      "renders '$expectedVerdict' for a metric with 30% noise when unstableNoisePct is $unstableNoisePct",
      async ({ unstableNoisePct, expectedVerdict, expectedGeomean }) => {
        await withScratchRepo(async (repo) => {
          createBranch(repo, {
            name: `old-noisy-${unstableNoisePct}`,
            benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          });

          createBranch(repo, {
            name: `new-noisy-${unstableNoisePct}`,
            benchScript: noisyBenchScript(path.join(repo.dir, "noisy-runs.txt")),
          });

          const report = renderReport(
            await compare(
              compareOptions({
                baseline: { target: `old-noisy-${unstableNoisePct}` },
                candidates: [{ target: `new-noisy-${unstableNoisePct}` }],
                unstableNoisePct,
              }),
            ),
          );

          const latencyRow = findLine(report, (line) => line.startsWith("latency"));
          expect.soft(latencyRow).toContain(expectedVerdict);
          // The run's only gating metric is the unstable one, so excluding it leaves the
          // geomean with nothing to average; a stable one keeps its ρ and tracks the -50%.
          expect(findLine(report, (line) => line.includes("geomean"))).toContain(expectedGeomean);
        });
      },
    );
  });
});
