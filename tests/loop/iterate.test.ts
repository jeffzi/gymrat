import path from "node:path";
import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GymratError } from "../../src/errors.js";
import { iterateSession } from "../../src/loop/iterate.js";
import type { TargetContext } from "../../src/sampling.js";
import { sessionJsonlPath } from "../../src/session/paths.js";
import type { IterationRecord, SessionRecord } from "../../src/session/records.js";
import { readRecords } from "../../src/session/store.js";
import { ISO_PATTERN, SESSION_ID, reportLines } from "../fixtures/constants.js";
import { captureRejectedGymratError } from "../fixtures/errors.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";
import {
  committedKeep,
  discardRecord as discardOf,
  finalizeRecord,
  iterationRecord,
  resolvedConfig,
  sessionRecord as sessionRecordDefaults,
  writeSessionLog,
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
    Promise.resolve(
      targets.map((ctx) => {
        const samples = byDir.get(ctx.dir);
        if (samples === undefined) {
          throw new Error(`stubSamples: unrecognized worktree dir ${ctx.dir}`);
        }
        return { ctx, samples };
      }),
    ),
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
    return Promise.resolve(
      targets.map((ctx) => {
        const samples = byDir.get(ctx.dir);
        if (samples === undefined) {
          throw new Error(`stubRuns: unrecognized worktree dir ${ctx.dir}`);
        }
        return { ctx, samples };
      }),
    );
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

beforeEach(() => {
  repo = createScratchRepo();
});

afterEach(() => {
  vi.restoreAllMocks();
  collectSamplesMock.mockReset();
  repo.cleanup();
});

describe("iterateSession", () => {
  describe("when the repository holds no session", () => {
    it("refuses with a hint pointing at the command that opens one", async () => {
      const error = await captureRejectedGymratError(() =>
        iterateSession(repo.dir, resolvedConfig()),
      );

      expect.soft(error.hint).toContain("gymrat start");
      expect(collectSamplesMock).not.toHaveBeenCalled();
    });
  });

  describe("when the session on disk was finalized", () => {
    it("refuses with a hint pointing at a fresh start", async () => {
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [
        iteration(1),
        committedKeep(1),
        finalizeRecord(),
      ]);

      const error = await captureRejectedGymratError(() =>
        iterateSession(repo.dir, resolvedConfig()),
      );

      expect.soft(error.hint).toContain("gymrat start");
      expect(collectSamplesMock).not.toHaveBeenCalled();
    });
  });

  describe("when the last iteration is neither kept nor discarded", () => {
    it("refuses with a hint naming both ways to settle it", async () => {
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [iteration(1)]);

      const error = await captureRejectedGymratError(() =>
        iterateSession(repo.dir, resolvedConfig()),
      );

      expect.soft(error.hint).toContain("gymrat keep");
      expect.soft(error.hint).toContain("gymrat discard");
      expect(collectSamplesMock).not.toHaveBeenCalled();
    });
  });

  describe("when a configured stop condition has already been met", () => {
    it("refuses once the configured maximum of iterations is on file", async () => {
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [
        iteration(1),
        committedKeep(1),
        iteration(2),
        committedKeep(2),
      ]);
      stubRuns(repo.dir, []);

      const error = await captureRejectedGymratError(() =>
        iterateSession(repo.dir, resolvedConfig({ stop: { maxIterations: 2 } })),
      );

      expect.soft(error.message).toContain("max iterations");
      expect.soft(error.message).toContain("2");
      expect.soft(collectSamplesMock).not.toHaveBeenCalled();
      expect(readRecords(sessionJsonlPath(repo.dir))).toHaveLength(5);
    });

    it("refuses once a target-reaching iteration has been kept", async () => {
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [onTargetIteration(1), committedKeep(1)]);
      stubRuns(repo.dir, []);

      const error = await captureRejectedGymratError(() =>
        iterateSession(
          repo.dir,
          resolvedConfig({ primary: "total_ms", stop: { targetValue: 95 } }),
        ),
      );

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
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [onTargetIteration(1), discardOf(1)]);

      const result = await iterateSession(
        repo.dir,
        resolvedConfig({ primary: "total_ms", stop: { targetValue: 95 } }),
      );

      expect(result.record.seq).toBe(2);
    });
  });

  describe("when no stop condition is configured", () => {
    it("measures again past a kept target and past any number of iterations", async () => {
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [
        onTargetIteration(1),
        committedKeep(1),
        iteration(2),
        committedKeep(2),
      ]);
      stubSamples(repo.dir, improvedRounds(), baselineRounds());

      const result = await iterateSession(repo.dir, resolvedConfig());

      expect(result.record.seq).toBe(3);
    });
  });

  describe("when a settled session is on disk", () => {
    beforeEach(() => {
      writeSessionLog(repo.dir, sessionRecord(repo.dir), [iteration(1), committedKeep(1)]);
      stubSamples(repo.dir, improvedRounds(), baselineRounds());
    });

    it("benches the session's two worktrees, baseline first", async () => {
      await iterateSession(repo.dir, resolvedConfig());

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
      await iterateSession(repo.dir, resolvedConfig());

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
      const result = await iterateSession(repo.dir, resolvedConfig());

      expect(asLogged(result.record)).toStrictEqual(lastIterationOf(repo.dir));
    });

    it("reads the iteration on a named primary metric alone", async () => {
      const result = await iterateSession(repo.dir, resolvedConfig({ primary: "total_ms" }));

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
      const result = await iterateSession(repo.dir, resolvedConfig({ primary: "total_ms", stop }));

      expect(result.record.targetReached).toBe(expected);
    });

    it("reads a higher-is-better target from the other side", async () => {
      const resolved = resolvedConfig({
        primary: "total_ms",
        stop: { targetValue: 85 },
        metrics: { total_ms: { direction: "higher" } },
      });

      const result = await iterateSession(repo.dir, resolved);

      expect(result.record.targetReached).toBe(true);
    });

    it("calls the target in the verdict block, right above the next step", async () => {
      const result = await iterateSession(
        repo.dir,
        resolvedConfig({ primary: "total_ms", stop: { targetValue: 95 } }),
      );

      expect(trimmedReportLines(result.report).at(-2)).toBe("target reached — keep it");
    });

    it.each([
      { description: "the target is still ahead", stop: { targetValue: 85 } },
      { description: "no stop condition is configured", stop: undefined },
    ])("leaves the target out of the report when $description", async ({ stop }) => {
      const result = await iterateSession(repo.dir, resolvedConfig({ primary: "total_ms", stop }));

      expect(stripAnsi(result.report)).not.toContain("target reached");
    });

    it("opens the report on the loop's own header, above the comparison table", async () => {
      const result = await iterateSession(repo.dir, resolvedConfig());

      const report = stripAnsi(result.report);
      expect
        .soft(report.split("\n")[0])
        .toBe("iteration 2 · experiment vs baseline · 10 paired samples");
      expect(report).toContain("total_ms");
    });
  });
});
