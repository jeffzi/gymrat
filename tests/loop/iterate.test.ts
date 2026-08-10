import fs from "node:fs";
import path from "node:path";
import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ResolvedConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import { iterateSession } from "../../src/loop/iterate.js";
import type { TargetContext } from "../../src/sampling.js";
import { sessionJsonlPath } from "../../src/session/paths.js";
import type {
  DiscardRecord,
  HookRecord,
  IterationRecord,
  SessionLogRecord,
  SessionRecord,
} from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import { createRunnableProgram, mockProcessExit } from "../fixtures/cli-harness.js";
import { ISO_PATTERN, SESSION_ID, reportLines } from "../fixtures/constants.js";
import { captureRejectedGymratError } from "../fixtures/errors.js";
import type { HookScripts } from "../fixtures/hook-scripts.js";
import { hookScripts } from "../fixtures/hook-scripts.js";
import { metricRecord } from "../fixtures/metrics.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";
import {
  AT,
  committedKeep,
  expectedHookRecord,
  iterationRecord,
  resolvedConfig,
  sessionRecord as sessionRecordDefaults,
} from "../fixtures/session-records.js";

type CollectSamples = typeof import("../../src/sampling.js").collectSamples;

/**
 * The one boundary this file mocks: sampling shells out to the consumer's bench
 * script, which no test can run. Everything downstream of it — verdicts,
 * aggregation, the record, the report — runs for real.
 */
const collectSamplesMock = vi.hoisted(() => vi.fn<CollectSamples>());

vi.mock("../../src/sampling.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/sampling.js")>();
  return { ...actual, collectSamples: collectSamplesMock };
});

/** Ten rounds of a bench that stayed near 100. */
const BASELINE_MS = [100, 101, 99, 100, 102, 98, 100, 101, 99, 100];
/** The same ten rounds, an order of magnitude larger, so a second metric reads differently. */
const BASELINE_BYTES = BASELINE_MS.map((value) => value * 10);

/**
 * Scale every round by `factor`, moving the median by exactly that much.
 *
 * A constant factor leaves every pairwise difference the same sign, which is
 * what makes the signed-rank test call the move rather than shrug at it.
 */
function scaled(values: readonly number[], factor: number): number[] {
  return values.map((value) => value * factor);
}

/**
 * Nudge alternate rounds up by `up` and the rest down by `down`.
 *
 * The mixed signs leave the signed-rank test nothing to call, while the larger
 * upward nudge still drags the median above the baseline's — a run that moved
 * the wrong way without saying anything, which is what `no-signal` means.
 */
function jittered(values: readonly number[], up: number, down: number): number[] {
  return values.map((value, index) => (index % 2 === 0 ? value + up : value - down));
}

/** One round per entry, pairing each metric with the value it reported that round. */
function rounds(
  totalMs: readonly number[],
  allocBytes: readonly number[],
): Record<string, number>[] {
  return totalMs.map((value, index) => ({
    total_ms: value,
    alloc_bytes: allocBytes[index] ?? 0,
  }));
}

/** The ten rounds the baseline worktree reports in every test here. */
function baselineRounds(): Record<string, number>[] {
  return rounds(BASELINE_MS, BASELINE_BYTES);
}

/** Ten rounds 10% faster and 20% leaner than the baseline's. */
function improvedRounds(): Record<string, number>[] {
  return rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 0.8));
}

/** Ten rounds 10% slower and 10% fatter than the baseline's. */
function regressedRounds(): Record<string, number>[] {
  return rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1));
}

/** Ten rounds that drift half a percent the wrong way without ever settling. */
function noisyRounds(): Record<string, number>[] {
  return rounds(jittered(BASELINE_MS, 2, 1), jittered(BASELINE_BYTES, 20, 10));
}

/** Rounds reporting `name` alone, the shape a bench filtered to that metric reports. */
function filteredRounds(name: string, values: readonly number[]): Record<string, number>[] {
  return values.map((value) => ({ [name]: value }));
}

/** The confirm-rerun template a consumer configures when their bench can be narrowed. */
const FILTER = "npm run bench -- --filter {names}";

/** A settled run configuration, geomean-led unless a test names its own primary. */
function config(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return resolvedConfig(overrides);
}

/**
 * A session header whose worktrees sit beside the default paths rather than on
 * them, so a run that recomputed the paths instead of reading them off the
 * record would bench directories no test ever filled.
 */
function sessionRecord(root: string): SessionRecord {
  return sessionRecordDefaults({
    sessionId: SESSION_ID,
    worktrees: {
      experiment: path.join(root, "side-experiment"),
      baseline: path.join(root, "side-baseline"),
    },
  });
}

/** A measured iteration numbered `seq`, settled by nobody. */
function iteration(seq: number): IterationRecord {
  return iterationRecord({ seq });
}

/** The iteration numbered `seq`, measured at or past the configured target. */
function onTargetIteration(seq: number): IterationRecord {
  return iterationRecord({ seq, targetReached: true });
}

/** A discard of the iteration numbered `seq`. */
function discardOf(seq: number): DiscardRecord {
  return { type: "discard", seq, at: AT };
}

/** Write a session log opening on `sessionRecord(root)` and holding `history` after it. */
function writeSessionLog(root: string, history: SessionLogRecord[] = []): void {
  appendRecord(sessionJsonlPath(root), sessionRecord(root));
  for (const record of history) {
    appendRecord(sessionJsonlPath(root), record);
  }
}

/**
 * Answer every sampling call with `experiment` for the experiment worktree and
 * `baseline` for the baseline one, keyed on the directory each context names.
 *
 * Keying on the directory rather than on call order is what lets the assertions
 * downstream read as evidence: a side that landed in the wrong half of the
 * record could only have come from the wrong worktree.
 */
function stubSamples(
  root: string,
  experiment: Record<string, number>[],
  baseline: Record<string, number>[],
): void {
  const byDir = new Map<string, Record<string, number>[]>([
    [sessionRecord(root).worktrees.experiment, experiment],
    [sessionRecord(root).worktrees.baseline, baseline],
  ]);
  collectSamplesMock.mockImplementation((_adapter, targets) =>
    Promise.resolve(targets.map((ctx) => ({ ctx, samples: byDir.get(ctx.dir) ?? [] }))),
  );
}

/** One paired run's answer to a sampling call: the rounds each worktree reports. */
interface PairedRun {
  experiment: Record<string, number>[];
  baseline: Record<string, number>[];
}

/**
 * Answer the nth sampling call with the nth entry of `runs` — a first run, then
 * the confirmation rerun — keyed on the directory each context names.
 *
 * A `GymratError` entry rejects that call instead, standing in for a bench that
 * failed mid-run. A call past the end of `runs` rejects too, so an unexpected
 * extra rerun surfaces as a failure rather than as silently reused samples.
 */
function stubRuns(root: string, runs: readonly (PairedRun | GymratError)[]): void {
  const worktrees = sessionRecord(root).worktrees;
  let index = 0;
  collectSamplesMock.mockImplementation((_adapter, targets) => {
    const run = runs[index];
    index += 1;
    if (run === undefined) {
      return Promise.reject(new Error(`unexpected sampling call ${index}`));
    }
    if (run instanceof GymratError) {
      return Promise.reject(run);
    }
    const byDir = new Map<string, Record<string, number>[]>([
      [worktrees.experiment, run.experiment],
      [worktrees.baseline, run.baseline],
    ]);
    return Promise.resolve(targets.map((ctx) => ({ ctx, samples: byDir.get(ctx.dir) ?? [] })));
  });
}

/** The targets and bench command of sampling call `index`, failing the test if there was none. */
function samplingCall(index: number): { targets: readonly TargetContext[]; bench: string } {
  const call = collectSamplesMock.mock.calls[index];
  if (call === undefined) {
    throw new Error(`expected collectSamples to have been called ${index + 1} time(s)`);
  }
  return { targets: call[1], bench: call[2].bench };
}

/** The contexts handed to the first sampling call, failing the test if there was none. */
function sampledTargets(): readonly TargetContext[] {
  return samplingCall(0).targets;
}

/** The report's lines, stripped of color and of the indentation a grouped metric carries. */
function trimmedReportLines(report: string): string[] {
  return reportLines(report, { trimLines: true });
}

/**
 * `value` after the round trip through JSON the session log puts it through.
 *
 * A record read back off the log carries `Object.prototype`, because that is what
 * `JSON.parse` builds; the live record's metric-keyed maps are prototype-free, as
 * every metric-keyed map in the pipeline is. `toStrictEqual` compares prototypes,
 * so the two sides have to meet on the logged shape to compare field by field.
 */
function asLogged(value: unknown): unknown {
  return JSON.parse(JSON.stringify(value));
}

/** The iteration record `root`'s log ends on, failing the test when it ends on something else. */
function lastIterationOf(root: string): IterationRecord {
  const last = readRecords(sessionJsonlPath(root)).at(-1);
  if (last?.type !== "iteration") {
    throw new Error(`expected an iteration record at the end of ${sessionJsonlPath(root)}`);
  }
  return last;
}

let repo: ScratchRepo;
let originalCwd: string;

beforeEach(() => {
  originalCwd = process.cwd();
  repo = createScratchRepo();
});

afterEach(() => {
  vi.restoreAllMocks();
  collectSamplesMock.mockReset();
  process.chdir(originalCwd);
  repo.cleanup();
});

describe("iterateSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", async () => {
      // Act
      const error = await captureRejectedGymratError(() => iterateSession(repo.dir, config()));

      // Assert
      expect.soft(error.hint).toContain("gymrat start");
      expect(collectSamplesMock).not.toHaveBeenCalled();
    });
  });

  describe("when the last iteration is neither kept nor discarded", () => {
    it("refuses with a hint naming both ways to settle it", async () => {
      // Arrange
      writeSessionLog(repo.dir, [iteration(1)]);

      // Act
      const error = await captureRejectedGymratError(() => iterateSession(repo.dir, config()));

      // Assert
      expect.soft(error.hint).toContain("gymrat keep");
      expect.soft(error.hint).toContain("gymrat discard");
      expect(collectSamplesMock).not.toHaveBeenCalled();
    });
  });

  describe("when a configured stop condition has already been met", () => {
    it("refuses once the configured maximum of iterations is on file", async () => {
      // Arrange
      writeSessionLog(repo.dir, [iteration(1), committedKeep(1), iteration(2), committedKeep(2)]);
      stubRuns(repo.dir, []);

      // Act
      const error = await captureRejectedGymratError(() =>
        iterateSession(repo.dir, config({ stop: { maxIterations: 2 } })),
      );

      // Assert
      expect.soft(error.message).toContain("max iterations");
      expect.soft(error.message).toContain("2");
      expect.soft(collectSamplesMock).not.toHaveBeenCalled();
      expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(5);
    });

    it("refuses once a target-reaching iteration has been kept", async () => {
      // Arrange
      writeSessionLog(repo.dir, [onTargetIteration(1), committedKeep(1)]);
      stubRuns(repo.dir, []);

      // Act
      const error = await captureRejectedGymratError(() =>
        iterateSession(repo.dir, config({ primary: "total_ms", stop: { targetValue: 95 } })),
      );

      // Assert
      expect.soft(error.message).toContain("target reached");
      expect.soft(collectSamplesMock).not.toHaveBeenCalled();
      expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(3);
    });
  });

  describe("when the target was reached by an iteration nobody kept", () => {
    beforeEach(() => {
      stubSamples(repo.dir, improvedRounds(), baselineRounds());
    });

    it("measures again after that iteration was discarded", async () => {
      // Arrange
      writeSessionLog(repo.dir, [onTargetIteration(1), discardOf(1)]);

      // Act
      const result = await iterateSession(
        repo.dir,
        config({ primary: "total_ms", stop: { targetValue: 95 } }),
      );

      // Assert
      expect(result.record.seq).toBe(2);
    });
  });

  describe("when no stop condition is configured", () => {
    it("measures again past a kept target and past any number of iterations", async () => {
      // Arrange
      writeSessionLog(repo.dir, [
        onTargetIteration(1),
        committedKeep(1),
        iteration(2),
        committedKeep(2),
      ]);
      stubSamples(repo.dir, improvedRounds(), baselineRounds());

      // Act
      const result = await iterateSession(repo.dir, config());

      // Assert
      expect(result.record.seq).toBe(3);
    });
  });

  describe("when a settled session is on disk", () => {
    beforeEach(() => {
      writeSessionLog(repo.dir, [iteration(1), committedKeep(1)]);
      stubSamples(repo.dir, improvedRounds(), baselineRounds());
    });

    it("benches the session's two worktrees, baseline first", async () => {
      // Act
      await iterateSession(repo.dir, config());

      // Assert
      const worktrees = sessionRecord(repo.dir).worktrees;
      expect(sampledTargets()).toStrictEqual([
        {
          target: { kind: "in-place", dir: worktrees.baseline },
          dir: worktrees.baseline,
          label: "baseline",
          position: "old",
        },
        {
          target: { kind: "in-place", dir: worktrees.experiment },
          dir: worktrees.experiment,
          label: "experiment",
          position: "new",
        },
      ]);
    });

    it("appends the measurement as the iteration after the last one settled", async () => {
      // Act
      await iterateSession(repo.dir, config());

      // Assert
      expect(lastIterationOf(repo.dir)).toStrictEqual({
        type: "iteration",
        seq: 2,
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        at: expect.stringMatching(ISO_PATTERN),
        samples: { experiment: improvedRounds(), baseline: baselineRounds() },
        metrics: {
          total_ms: {
            // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
            deltaPct: expect.closeTo(-10, 6),
            verdict: "improved",
            method: "signed-rank",
            // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
            p: expect.any(Number),
            // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
            noisePct: expect.any(Number),
            gating: true,
            confirmed: false,
          },
          alloc_bytes: {
            // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
            deltaPct: expect.closeTo(-20, 6),
            verdict: "improved",
            method: "signed-rank",
            // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
            p: expect.any(Number),
            // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
            noisePct: expect.any(Number),
            gating: true,
            confirmed: false,
          },
        },
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        primary: { kind: "geomean", deltaPct: expect.closeTo(-15.1472, 3) },
        outcome: "improved",
        targetReached: false,
      });
    });

    it("hands back the record it appended", async () => {
      // Act
      const result = await iterateSession(repo.dir, config());

      // Assert
      expect(asLogged(result.record)).toStrictEqual(lastIterationOf(repo.dir));
    });

    it("reads the iteration on a named primary metric alone", async () => {
      // Act
      const result = await iterateSession(repo.dir, config({ primary: "total_ms" }));

      // Assert
      expect(result.record.primary).toStrictEqual({
        kind: "metric",
        name: "total_ms",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        deltaPct: expect.closeTo(-10, 6),
      });
    });

    it.each([
      { description: "no stop condition is configured", stop: undefined, expected: false },
      { description: "the target is still ahead", stop: { targetValue: 85 }, expected: false },
      { description: "the target is met", stop: { targetValue: 95 }, expected: true },
    ])("records targetReached as $expected when $description", async ({ stop, expected }) => {
      // Act
      const result = await iterateSession(repo.dir, config({ primary: "total_ms", stop }));

      // Assert
      expect(result.record.targetReached).toBe(expected);
    });

    it("reads a higher-is-better target from the other side", async () => {
      // Arrange
      const resolved = config({
        primary: "total_ms",
        stop: { targetValue: 85 },
        metrics: { total_ms: { direction: "higher" } },
      });

      // Act
      const result = await iterateSession(repo.dir, resolved);

      // Assert
      expect(result.record.targetReached).toBe(true);
    });

    it("calls the target in the verdict block, right above the next step", async () => {
      // Act
      const result = await iterateSession(
        repo.dir,
        config({ primary: "total_ms", stop: { targetValue: 95 } }),
      );

      // Assert
      expect(trimmedReportLines(result.report).at(-2)).toBe("target reached — keep it");
    });

    it.each([
      { description: "the target is still ahead", stop: { targetValue: 85 } },
      { description: "no stop condition is configured", stop: undefined },
    ])("leaves the target out of the report when $description", async ({ stop }) => {
      // Act
      const result = await iterateSession(repo.dir, config({ primary: "total_ms", stop }));

      // Assert
      expect(stripAnsi(result.report)).not.toContain("target reached");
    });

    it("opens the report on the loop's own header, above the comparison table", async () => {
      // Act
      const result = await iterateSession(repo.dir, config());

      // Assert
      const report = stripAnsi(result.report);
      expect
        .soft(report.split("\n")[0])
        .toBe("iteration 2 · experiment vs baseline · 10 paired samples");
      expect(report).toContain("total_ms");
    });
  });

  describe("when a gating metric comes back regressed", () => {
    beforeEach(() => {
      writeSessionLog(repo.dir);
    });

    it("reruns the same paired sampling through the filter template", async () => {
      // Arrange
      stubRuns(repo.dir, [
        { experiment: regressedRounds(), baseline: baselineRounds() },
        { experiment: regressedRounds(), baseline: baselineRounds() },
      ]);

      // Act
      await iterateSession(repo.dir, config({ filter: FILTER }));

      // Assert
      expect.soft(collectSamplesMock).toHaveBeenCalledTimes(2);
      expect.soft(samplingCall(1).targets).toStrictEqual(samplingCall(0).targets);
      expect(samplingCall(1).bench).toBe("npm run bench -- --filter total_ms alloc_bytes");
    });

    it("reruns the whole bench command when no filter template is configured", async () => {
      // Arrange
      stubRuns(repo.dir, [
        { experiment: regressedRounds(), baseline: baselineRounds() },
        { experiment: regressedRounds(), baseline: baselineRounds() },
      ]);

      // Act
      await iterateSession(repo.dir, config());

      // Assert
      expect(samplingCall(1).bench).toBe("npm run bench");
    });

    it("records the rerun's raw samples beside the metrics it re-measured", async () => {
      // Arrange
      const rerun = {
        experiment: filteredRounds("total_ms", scaled(BASELINE_MS, 1.2)),
        baseline: filteredRounds("total_ms", BASELINE_MS),
      };
      stubRuns(repo.dir, [{ experiment: regressedRounds(), baseline: baselineRounds() }, rerun]);
      const resolved = config({ filter: FILTER, metrics: { alloc_bytes: { gating: false } } });

      // Act
      const result = await iterateSession(repo.dir, resolved);

      // Assert
      expect(result.record.confirm).toStrictEqual({
        ran: true,
        filtered: ["total_ms"],
        samples: { experiment: rerun.experiment, baseline: rerun.baseline },
      });
    });

    it("marks the metric confirmed and reads the iteration as regressed when the rerun agrees", async () => {
      // Arrange
      stubRuns(repo.dir, [
        { experiment: regressedRounds(), baseline: baselineRounds() },
        {
          experiment: filteredRounds("total_ms", scaled(BASELINE_MS, 1.2)),
          baseline: filteredRounds("total_ms", BASELINE_MS),
        },
      ]);
      const resolved = config({ filter: FILTER, metrics: { alloc_bytes: { gating: false } } });

      // Act
      const result = await iterateSession(repo.dir, resolved);

      // Assert
      expect.soft(result.record.metrics.total_ms).toStrictEqual({
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        deltaPct: expect.closeTo(10, 6),
        verdict: "regressed",
        method: "signed-rank",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        p: expect.any(Number),
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        noisePct: expect.any(Number),
        gating: true,
        confirmed: true,
      });
      expect.soft(result.record.outcome).toBe("regressed");
      expect(trimmedReportLines(result.report)).toContain(
        "total_ms: regression confirmed on rerun",
      );
    });

    it.each([
      { rerun: "improved", experiment: scaled(BASELINE_MS, 0.9) },
      { rerun: "no-signal", experiment: jittered(BASELINE_MS, 2, 1) },
    ])(
      "demotes the metric to no-signal when the rerun comes back $rerun",
      async ({ experiment }) => {
        // Arrange
        stubRuns(repo.dir, [
          { experiment: regressedRounds(), baseline: baselineRounds() },
          {
            experiment: filteredRounds("total_ms", experiment),
            baseline: filteredRounds("total_ms", BASELINE_MS),
          },
        ]);
        const resolved = config({ filter: FILTER, metrics: { alloc_bytes: { gating: false } } });

        // Act
        const result = await iterateSession(repo.dir, resolved);

        // Assert
        expect.soft(result.record.metrics.total_ms).toStrictEqual({
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          deltaPct: expect.closeTo(10, 6),
          verdict: "no-signal",
          method: "signed-rank",
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          p: expect.any(Number),
          // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
          noisePct: expect.any(Number),
          gating: true,
          confirmed: false,
        });
        expect.soft(result.record.outcome).toBe("no-signal");
        expect(trimmedReportLines(result.report)).toContain(
          "total_ms: regression not confirmed on rerun",
        );
      },
    );

    it("fails the iterate and records nothing when the rerun's bench fails", async () => {
      // Arrange
      stubRuns(repo.dir, [
        { experiment: regressedRounds(), baseline: baselineRounds() },
        new GymratError("bench command failed"),
      ]);
      const resolved = config({ filter: FILTER, metrics: { alloc_bytes: { gating: false } } });

      // Act
      const error = await captureRejectedGymratError(() => iterateSession(repo.dir, resolved));

      // Assert
      expect.soft(error.message).toBe("bench command failed");
      expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(1);
    });
  });

  describe("when the regressed metric is one a rerun cannot inform", () => {
    beforeEach(() => {
      writeSessionLog(repo.dir);
    });

    it("gates an exact metric on the first run alone", async () => {
      // Arrange
      stubSamples(repo.dir, regressedRounds(), baselineRounds());
      const resolved = config({
        metrics: { total_ms: { exact: true }, alloc_bytes: { gating: false } },
      });

      // Act
      const result = await iterateSession(repo.dir, resolved);

      // Assert
      expect.soft(collectSamplesMock).toHaveBeenCalledTimes(1);
      expect.soft(result.record.metrics.total_ms).toStrictEqual({
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        deltaPct: expect.closeTo(10, 6),
        verdict: "regressed",
        method: "exact",
        gating: true,
        confirmed: false,
      });
      expect(result.record.outcome).toBe("regressed");
    });

    it("leaves an exact metric out of the rerun's filter list", async () => {
      // Arrange
      stubRuns(repo.dir, [
        { experiment: regressedRounds(), baseline: baselineRounds() },
        {
          experiment: filteredRounds("alloc_bytes", scaled(BASELINE_BYTES, 1.2)),
          baseline: filteredRounds("alloc_bytes", BASELINE_BYTES),
        },
      ]);

      // Act
      await iterateSession(
        repo.dir,
        config({ filter: FILTER, metrics: { total_ms: { exact: true } } }),
      );

      // Assert
      expect(samplingCall(1).bench).toBe("npm run bench -- --filter alloc_bytes");
    });

    it("leaves a non-gating metric to inform without rerunning", async () => {
      // Arrange
      const experiment = rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 1.1));
      stubSamples(repo.dir, experiment, baselineRounds());

      // Act
      const result = await iterateSession(
        repo.dir,
        config({ metrics: { alloc_bytes: { gating: false } } }),
      );

      // Assert
      expect.soft(collectSamplesMock).toHaveBeenCalledTimes(1);
      expect.soft(result.record).not.toHaveProperty("confirm");
      expect.soft(result.record.metrics.alloc_bytes?.verdict).toBe("regressed");
      expect(result.record.outcome).toBe("improved");
    });
  });

  describe("when the bench names a metric after an Object.prototype member", () => {
    // A plain object literal would read `__proto__` as its prototype rather than
    // as a metric name, so the key reaches every object here through a variable.
    const PROTO = "__proto__";

    /** One round per entry, pairing `total_ms` with the metric named `__proto__`. */
    function protoRounds(
      totalMs: readonly number[],
      proto: readonly number[],
    ): Record<string, number>[] {
      return totalMs.map((value, index) => ({ total_ms: value, [PROTO]: proto[index] ?? 0 }));
    }

    /** The verdict an improved signed-rank metric carries, having moved `deltaPct`. */
    function improvedVerdict(deltaPct: number): Record<string, unknown> {
      return {
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        deltaPct: expect.closeTo(deltaPct, 6),
        verdict: "improved",
        method: "signed-rank",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        p: expect.any(Number),
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        noisePct: expect.any(Number),
        gating: true,
        confirmed: false,
      };
    }

    beforeEach(() => {
      writeSessionLog(repo.dir);
      stubSamples(
        repo.dir,
        protoRounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 0.8)),
        protoRounds(BASELINE_MS, BASELINE_BYTES),
      );
    });

    it("keeps that metric's verdict as an own key of the record it writes", async () => {
      // Act
      const result = await iterateSession(repo.dir, config());

      // Assert
      expect(result.record.metrics).toStrictEqual(
        metricRecord({ total_ms: improvedVerdict(-10), [PROTO]: improvedVerdict(-20) }),
      );
    });

    it("counts that metric in the geomean the primary reads", async () => {
      // Act
      const result = await iterateSession(repo.dir, config());

      // Assert
      expect(result.record.primary).toStrictEqual({
        kind: "geomean",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        deltaPct: expect.closeTo(-15.1472, 3),
      });
    });
  });

  describe.each([
    {
      outcome: "improved",
      word: "IMPROVED",
      experiment: improvedRounds(),
      nextStep: "Hint: gymrat keep",
    },
    {
      outcome: "regressed",
      word: "REGRESSED",
      experiment: regressedRounds(),
      nextStep: "Hint: fix or gymrat discard",
    },
    {
      outcome: "no-signal",
      word: "NO-SIGNAL",
      experiment: noisyRounds(),
      nextStep: "Hint: gymrat keep or gymrat discard",
    },
  ])("when the measurement comes out $outcome", ({ outcome, word, experiment, nextStep }) => {
    it("closes the report on that verdict and the step it calls for", async () => {
      // Arrange
      writeSessionLog(repo.dir);
      stubSamples(repo.dir, experiment, baselineRounds());

      // Act
      const result = await iterateSession(repo.dir, config());

      // Assert
      const lines = stripAnsi(result.report).split("\n");
      expect.soft(result.record.outcome).toBe(outcome);
      expect.soft(lines.at(-2)).toContain(word);
      expect(lines.at(-1)).toBe(nextStep);
    });
  });

  describe("when the config declares a command for a stage", () => {
    /** The experiment worktree the hooks run in, laid down so a command can start there. */
    let experimentDir: string;
    let hookCommand: HookScripts["hookCommand"];
    let printing: HookScripts["printing"];

    /** A command filing its payload away where the assertions can read it back. */
    function capturingPayload(stage: "before" | "after"): string {
      return hookCommand(
        `import fs from "node:fs";\n` +
          `const chunks = [];\n` +
          `process.stdin.on("data", (chunk) => chunks.push(chunk));\n` +
          `process.stdin.on("end", () => {\n` +
          `  fs.writeFileSync(${JSON.stringify(`${stage}.json`)}, Buffer.concat(chunks));\n` +
          `});\n`,
      );
    }

    /**
     * The payload the `stage` hook was handed, as the hook itself saw it.
     *
     * The capturing command names the file relatively, so reading it back out of
     * the experiment worktree is also what proves the hook ran there.
     */
    function payloadOf(stage: "before" | "after"): unknown {
      return JSON.parse(fs.readFileSync(path.join(experimentDir, `${stage}.json`), "utf-8"));
    }

    /** Every hook record the session log holds, oldest first. */
    function hookRecords(): HookRecord[] {
      return readRecords(sessionJsonlPath(repo.dir)).filter((record) => record.type === "hook");
    }

    beforeEach(() => {
      experimentDir = sessionRecord(repo.dir).worktrees.experiment;
      fs.mkdirSync(experimentDir, { recursive: true });
      ({ hookCommand, printing } = hookScripts(repo.dir));
      writeSessionLog(repo.dir, [iteration(1), committedKeep(1)]);
      stubSamples(repo.dir, improvedRounds(), baselineRounds());
    });

    it("fires the before hook ahead of the measurement and the after hook once it is on file", async () => {
      // Arrange
      const hooks = { before: printing("hi"), after: printing("bye") };

      // Act
      await iterateSession(repo.dir, config({ hooks }));

      // Assert
      const records = readRecords(sessionJsonlPath(repo.dir));
      expect
        .soft(records.map((record) => record.type))
        .toStrictEqual(["session", "iteration", "keep", "hook", "iteration", "hook"]);
      expect(records.filter((record) => record.type === "hook")).toStrictEqual([
        expectedHookRecord({ stage: "before", seq: 2, exitCode: 0, stdoutBytes: 3 }),
        expectedHookRecord({ stage: "after", seq: 2, exitCode: 0, stdoutBytes: 4 }),
      ]);
    });

    it("tells each hook which iteration it sits next to", async () => {
      // Arrange
      const hooks = { before: capturingPayload("before"), after: capturingPayload("after") };

      // Act
      const result = await iterateSession(repo.dir, config({ hooks }));

      // Assert
      expect.soft(payloadOf("before")).toStrictEqual({
        stage: "before",
        experimentDir,
        seq: 2,
        lastIteration: iteration(1),
        session: {
          sessionId: SESSION_ID,
          baseline: sessionRecord(repo.dir).baseline,
          branch: `gymrat/${SESSION_ID}`,
          iterationCount: 1,
        },
      });
      expect(payloadOf("after")).toStrictEqual({
        stage: "after",
        experimentDir,
        seq: 2,
        lastIteration: asLogged(result.record),
        session: {
          sessionId: SESSION_ID,
          baseline: sessionRecord(repo.dir).baseline,
          branch: `gymrat/${SESSION_ID}`,
          iterationCount: 2,
        },
      });
    });

    it("prints each hook's output around the measurement it brackets", async () => {
      // Arrange
      const hooks = {
        before: printing("warmed the cache"),
        after: printing("archived the samples"),
      };

      // Act
      const result = await iterateSession(repo.dir, config({ hooks }));

      // Assert
      const lines = trimmedReportLines(result.report);
      expect.soft(lines[0]).toBe("[before] warmed the cache");
      expect.soft(lines.at(-1)).toBe("[after] archived the samples");
      expect(lines[1]).toBe("iteration 2 · experiment vs baseline · 10 paired samples");
    });

    it("measures on past a hook that failed, reporting and recording the failure", async () => {
      // Arrange
      const before = hookCommand(
        `process.stderr.write("no warm copy\\n");\nprocess.exitCode = 3;\n`,
      );

      // Act
      const result = await iterateSession(repo.dir, config({ hooks: { before } }));

      // Assert
      expect
        .soft(trimmedReportLines(result.report).slice(0, 2))
        .toStrictEqual(["[before] hook exited 3", "[before] no warm copy"]);
      expect
        .soft(hookRecords())
        .toStrictEqual([
          expectedHookRecord({ stage: "before", seq: 2, exitCode: 3, stdoutBytes: 0 }),
        ]);
      expect.soft(lastIterationOf(repo.dir).seq).toBe(2);
      expect(result.record.outcome).toBe("improved");
    });

    it.each([
      { description: "the config declares no hooks at all", withAfter: false },
      { description: "the config declares only the other stage", withAfter: true },
    ])("runs nothing for the before stage when $description", async ({ withAfter }) => {
      // Arrange
      const hooks = withAfter ? { after: printing("bye") } : undefined;

      // Act
      const result = await iterateSession(repo.dir, config({ hooks }));

      // Assert
      expect
        .soft(hookRecords().map((record) => record.stage))
        .toStrictEqual(withAfter ? ["after"] : []);
      expect(
        trimmedReportLines(result.report).filter((line) => line.startsWith("[before]")),
      ).toStrictEqual([]);
    });
  });
});

describe("the iterate command", () => {
  it("measures the session in the repository it runs in and reports it on stdout", async () => {
    // Arrange
    writeSessionLog(repo.dir);
    stubSamples(repo.dir, improvedRounds(), baselineRounds());
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    let stdout = "";
    vi.spyOn(process.stdout, "write").mockImplementation((chunk) => {
      stdout += String(chunk);
      return true;
    });

    // Act
    await program.parseAsync(["node", "cli.js", "iterate", "--bench", "npm run bench"]);

    // Assert
    const lines = stripAnsi(stdout).split("\n").filter(Boolean);
    expect.soft(lines[0]).toBe("iteration 1 · experiment vs baseline · 10 paired samples");
    expect.soft(lines.at(-1)).toBe("Hint: gymrat keep");
    expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(2);
  });

  it("exits 1 when a configured stop condition has already been met", async () => {
    // Arrange
    writeSessionLog(repo.dir, [iteration(1), committedKeep(1)]);
    fs.writeFileSync(
      path.join(repo.dir, "gymrat.json"),
      JSON.stringify({ bench: "npm run bench", stop: { maxIterations: 1 } }),
    );
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockProcessExit();

    // Act
    const parsing = program.parseAsync(["node", "cli.js", "iterate"]);

    // Assert
    await expect(parsing).rejects.toHaveProperty("exitCode", 1);
    const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
    expect.soft(stderrText).toContain("max iterations");
    expect(collectSamplesMock).not.toHaveBeenCalled();
  });

  it("exits 2 with a start hint when the repository holds no session", async () => {
    // Arrange
    process.chdir(repo.dir);
    const program = createRunnableProgram({ exitOverride: "all", silent: true });
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockProcessExit();

    // Act
    const parsing = program.parseAsync(["node", "cli.js", "iterate", "--bench", "npm run bench"]);

    // Assert
    await expect(parsing).rejects.toHaveProperty("exitCode", 2);
    const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
    expect(stderrText).toContain("gymrat start");
  });
});
