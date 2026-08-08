import path from "node:path";
import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createProgram } from "../../src/cli.js";
import type { ResolvedConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import { iterateSession } from "../../src/loop/iterate.js";
import type { TargetContext } from "../../src/sampling.js";
import { sessionJsonlPath } from "../../src/session/paths.js";
import type {
  IterationRecord,
  KeepRecord,
  SessionLogRecord,
  SessionRecord,
} from "../../src/session/records.js";
import { appendRecord, readRecords } from "../../src/session/store.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";

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

const ISO_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const AT = "2026-08-08T14:15:30.000Z";
const COMMIT = "b".repeat(40);
const SESSION_ID = "20260808-141530-abcd";

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

/** A settled run configuration, geomean-led unless a test names its own primary. */
function config(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return {
    bench: "npm run bench",
    adapter: "metric-lines",
    samples: 10,
    timeoutSeconds: 1800,
    unstableNoisePct: 200,
    primary: "geomean",
    hooks: "gymrat.hooks",
    ...overrides,
  };
}

/**
 * A session header whose worktrees sit beside the default paths rather than on
 * them, so a run that recomputed the paths instead of reading them off the
 * record would bench directories no test ever filled.
 */
function sessionRecord(root: string): SessionRecord {
  return {
    type: "session",
    schemaVersion: 1,
    sessionId: SESSION_ID,
    createdAt: AT,
    baseline: { ref: "main", sha: "a".repeat(40) },
    branch: `gymrat/${SESSION_ID}`,
    worktrees: {
      experiment: path.join(root, "side-experiment"),
      baseline: path.join(root, "side-baseline"),
    },
    config: {
      bench: "npm run bench",
      adapter: "metric-lines",
      samples: 10,
      timeoutSeconds: 1800,
      primary: "geomean",
      hooks: "gymrat.hooks",
    },
  };
}

/** A measured iteration numbered `seq`, settled by nobody. */
function iteration(seq: number): IterationRecord {
  return {
    type: "iteration",
    seq,
    at: AT,
    samples: { experiment: [{ total_ms: 14100 }], baseline: [{ total_ms: 15200 }] },
    metrics: {
      total_ms: {
        deltaPct: -7.2,
        verdict: "improved",
        method: "signed-rank",
        p: 0.002,
        noisePct: 1.4,
        gating: true,
        confirmed: false,
      },
    },
    primary: { kind: "geomean", deltaPct: -7.2 },
    outcome: "improved",
    targetReached: false,
  };
}

/** A keep that committed the iteration numbered `seq`. */
function committedKeep(seq: number): KeepRecord {
  return {
    type: "keep",
    seq,
    at: AT,
    status: "committed",
    commit: COMMIT,
    message: "cache the regex",
    checks: { configured: true, passed: true },
  };
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

/** The contexts handed to the first sampling call, failing the test if there was none. */
function sampledTargets(): readonly TargetContext[] {
  const call = collectSamplesMock.mock.calls[0];
  if (call === undefined) {
    throw new Error("expected collectSamples to have been called");
  }
  return call[1];
}

/** Run `act` and hand back the GymratError it rejected with, failing the test if it threw none. */
async function captureGymratError(act: () => Promise<unknown>): Promise<GymratError> {
  try {
    await act();
  } catch (error) {
    if (error instanceof GymratError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected the call to reject with a GymratError");
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
      const error = await captureGymratError(() => iterateSession(repo.dir, config()));

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
      const error = await captureGymratError(() => iterateSession(repo.dir, config()));

      // Assert
      expect.soft(error.hint).toContain("gymrat keep");
      expect.soft(error.hint).toContain("gymrat discard");
      expect(collectSamplesMock).not.toHaveBeenCalled();
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
      expect(result.record).toStrictEqual(lastIterationOf(repo.dir));
    });

    it.each([
      {
        description: "a geomean primary over every gating metric",
        primary: "geomean",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        expected: { kind: "geomean", deltaPct: expect.closeTo(-15.1472, 3) },
      },
      {
        description: "a named primary metric alone",
        primary: "total_ms",
        // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
        expected: { kind: "metric", name: "total_ms", deltaPct: expect.closeTo(-10, 6) },
      },
    ])("reads the iteration on $description", async ({ primary, expected }) => {
      // Act
      const result = await iterateSession(repo.dir, config({ primary }));

      // Assert
      expect(result.record.primary).toStrictEqual(expected);
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
});

describe("the iterate command", () => {
  /** A program whose subcommands throw instead of exiting, with stderr silenced. */
  function createSilentProgram(): ReturnType<typeof createProgram> {
    const program = createProgram();
    for (const command of [program, ...program.commands]) {
      command.exitOverride();
      command.configureOutput({ writeErr: () => {} });
    }
    return program;
  }

  it("measures the session in the repository it runs in and reports it on stdout", async () => {
    // Arrange
    writeSessionLog(repo.dir);
    stubSamples(repo.dir, improvedRounds(), baselineRounds());
    process.chdir(repo.dir);
    const program = createSilentProgram();
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

  it("exits 2 with a start hint when the repository holds no session", async () => {
    // Arrange
    process.chdir(repo.dir);
    const program = createSilentProgram();
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- vitest mock requires cast
    vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw Object.assign(new Error(`process.exit(${code})`), { exitCode: code });
    }) as never);

    // Act
    const parsing = program.parseAsync(["node", "cli.js", "iterate", "--bench", "npm run bench"]);

    // Assert
    await expect(parsing).rejects.toHaveProperty("exitCode", 2);
    const stderrText = stderrSpy.mock.calls.map((call) => String(call[0])).join("");
    expect(stderrText).toContain("gymrat start");
  });
});
