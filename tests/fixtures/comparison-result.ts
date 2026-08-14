import type {
  CandidateComparison,
  CandidateMetric,
  ComparisonResult,
} from "../../src/report/types.js";
import type { KindAggregate } from "../../src/verdict/aggregate.js";
import type {
  ApproximateVerdictValue,
  BandVerdict,
  ExactVerdict,
  GeomeanResult,
  SignedRankVerdict,
} from "../../src/verdict/verdict.js";

/** The name-keyed map of all metrics compared in a run. */
export type Metrics = ComparisonResult["metrics"];
/** One metric's comparison data: baseline, per-candidate results, and meta. */
export type MetricEntry = Metrics[string];

/** A `MetricEntry["meta"]` block, defaulting to a lower-is-better, gating, non-exact "other" metric. */
export function metricMeta(
  shortName: string,
  overrides: Partial<MetricEntry["meta"]> = {},
): MetricEntry["meta"] {
  return {
    direction: "lower",
    gating: true,
    exact: false,
    kind: "other",
    shortName,
    ...overrides,
  };
}

/**
 * One candidate's run-level results, judged against the shared baseline.
 *
 * `kinds` defaults to the single-kind run every other default here describes:
 * one `other` kind, no groups, whose section and gated geomeans share the
 * same default aggregate. A multi-kind test passes its own aggregates instead.
 */
export function createCandidate(overrides: Partial<CandidateComparison> = {}): CandidateComparison {
  const { kinds, ...rest } = overrides;
  return {
    label: "perf/faster-decode",
    ...rest,
    kinds: kinds ?? [otherKind(-5.8, 10)],
  };
}

/**
 * A comparison result with a clean baseline-plus-one-candidate run and no metrics.
 *
 * Shared by the renderer tests and the CLI tests so both drive the renderer
 * with the same shape `compare()` returns.
 */
export function createComparisonResult(
  overrides: Partial<ComparisonResult> = {},
): ComparisonResult {
  return {
    baselineLabel: "main",
    candidates: [createCandidate()],
    samples: 10,
    adapter: "mitata",
    metrics: {},
    worktreesRemoved: 0,
    worktreesLeftBehind: [],
    worktreePruneError: undefined,
    ...overrides,
  };
}

/** A two-sided metric whose verdict came from the signed-rank method. */
export function signedRankMetric(options: {
  verdict: ApproximateVerdictValue;
  delta: number;
  baselineMedian?: number;
  baselineSpread?: number;
  candidateMedian?: number;
  candidateSpread?: number;
  p?: number;
  noisePct?: number;
  noiseAbs?: number;
  unit?: "ns" | "bytes";
  gating?: boolean;
  direction?: "lower" | "higher";
  n?: number;
}): MetricEntry {
  const {
    verdict,
    delta,
    baselineMedian = 100,
    baselineSpread = 1,
    candidateMedian = baselineMedian * (1 + delta / 100),
    candidateSpread = 1,
    p = 0.01,
    noisePct = 2.5,
    noiseAbs = 3.5,
    unit,
    gating = true,
    direction = "lower",
    n = 10,
  } = options;
  return {
    baselineMedian,
    baselineSpread,
    candidates: [
      {
        median: candidateMedian,
        spread: candidateSpread,
        verdict: { verdict, method: "signed-rank", delta, n, p, noisePct, noiseAbs },
      },
    ],
    meta: { direction, gating, exact: false, unit, kind: "other", shortName: "time" },
  };
}

/**
 * A two-sided metric whose verdict fell back to the noise band.
 *
 * `n` is the total pair count and `usableN` how many of those pairs survived
 * tie-dropping. `n < 6` means the run was too short for the signed-rank test;
 * `n >= 6` with `usableN < 6` means ties starved it instead.
 */
export function bandMetric(
  options: {
    verdict?: ApproximateVerdictValue;
    delta?: number;
    noisePct?: number;
    n?: number;
    usableN?: number;
    direction?: "lower" | "higher";
    unit?: "ns" | "bytes";
  } = {},
): MetricEntry {
  const {
    verdict = "no-signal",
    delta = -1,
    noisePct = 2.5,
    n = 4,
    usableN = n,
    direction = "lower",
    unit,
  } = options;
  return {
    baselineMedian: 100,
    baselineSpread: 5,
    candidates: [
      {
        median: 100 + delta,
        spread: 4,
        verdict: {
          verdict,
          method: "band",
          delta,
          n,
          usableN,
          band: noisePct,
          noisePct,
          noiseAbs: 3.5,
        },
      },
    ],
    meta: { direction, gating: true, exact: false, unit, kind: "other", shortName: "time" },
  };
}

/**
 * A run of one paired sample, where every verdict rests on a single pair.
 *
 * One pair leaves the band method no spread to measure, so it collapses to the
 * noise floor and reports no signal whatever the deltas were: the ±0.5% beside
 * them is a constant, not an observation.
 */
export function singleSampleResult(): ComparisonResult {
  return createComparisonResult({
    samples: 1,
    metrics: {
      "decode/time": bandMetric({ delta: -0.4, noisePct: 0.5, n: 1, unit: "ns" }),
      "encode/time": bandMetric({ delta: 0.2, noisePct: 0.5, n: 1, unit: "ns" }),
    },
    candidates: [createCandidate({ kinds: [otherKind(-0.1, 2)] })],
  });
}

/** A counted metric, compared exactly rather than statistically. */
export function exactMetric(options: {
  delta: number;
  baselineMedian?: number;
  candidateMedian?: number;
  n?: number;
  unit?: "ns" | "bytes";
}): MetricEntry {
  const {
    delta,
    baselineMedian = 1000,
    candidateMedian = 1000 * (1 + delta / 100),
    n = 10,
    unit = "bytes",
  } = options;
  return {
    baselineMedian,
    candidates: [
      {
        median: candidateMedian,
        verdict: { verdict: delta < 0 ? "improved" : "regressed", method: "exact", delta, n },
      },
    ],
    meta: { direction: "lower", gating: true, exact: true, unit, kind: "other", shortName: "heap" },
  };
}

/** A noise-band verdict, tied pairs and all. */
export function bandVerdict(overrides: Partial<BandVerdict> = {}): BandVerdict {
  return {
    verdict: "no-signal",
    method: "band",
    delta: -0.5,
    n: 10,
    usableN: 3,
    band: 2.5,
    noisePct: 2.5,
    noiseAbs: 2.5,
    ...overrides,
  };
}

/** A verdict the Wilcoxon signed-rank test produced. */
export function signedRankVerdict(overrides: Partial<SignedRankVerdict> = {}): SignedRankVerdict {
  return {
    verdict: "no-signal",
    method: "signed-rank",
    delta: 0.2,
    n: 10,
    p: 0.49,
    noisePct: 2.5,
    noiseAbs: 2.5,
    ...overrides,
  };
}

/** A verdict read straight off a counted metric, with no statistics behind it. */
export function exactVerdict(overrides: Partial<ExactVerdict> = {}): ExactVerdict {
  return { verdict: "no-signal", method: "exact", delta: 0, n: 10, ...overrides };
}

/** One metric judged for several candidates against a single shared baseline. */
export function nWayMetric(
  candidates: readonly { verdict: ApproximateVerdictValue; delta: number; median: number }[],
): MetricEntry {
  const entries: MetricEntry["candidates"] = candidates.map(({ verdict, delta, median }) => ({
    median,
    spread: 1,
    verdict: {
      verdict,
      method: "signed-rank",
      delta,
      n: 10,
      p: 0.01,
      noisePct: 2.5,
      noiseAbs: 3.5,
    },
  }));
  return {
    baselineMedian: 100,
    baselineSpread: 1,
    candidates: entries,
    meta: {
      direction: "lower",
      gating: true,
      exact: false,
      unit: "ns",
      kind: "other",
      shortName: "time",
    },
  };
}

/** A metric of `kind`, displayed under `shortName`, judged by the signed-rank test. */
export function kindMetric(options: {
  kind: string;
  shortName: string;
  verdict: ApproximateVerdictValue;
  delta: number;
  gating?: boolean;
  unit?: "ns" | "bytes";
}): MetricEntry {
  const { kind, shortName, verdict, delta, gating = true, unit = "ns" } = options;
  const metric = signedRankMetric({ verdict, delta, gating, unit });
  return { ...metric, meta: { ...metric.meta, kind, shortName } };
}

/** A metric of `kind` judged once per candidate against the shared baseline. */
export function nWayKindMetric(options: {
  kind: string;
  shortName: string;
  candidates: readonly { verdict: ApproximateVerdictValue; delta: number; median: number }[];
  gating?: boolean;
}): MetricEntry {
  const { kind, shortName, candidates, gating = true } = options;
  const metric = nWayMetric(candidates);
  return { ...metric, meta: { ...metric.meta, kind, shortName, gating } };
}

/** A geomean over `n` metrics, with no exclusions and no band unless overridden. */
export function geomeanOf(
  value: number,
  n: number,
  overrides: Partial<GeomeanResult> = {},
): GeomeanResult {
  return { value, n, excluded: [], band: 0, ...overrides };
}

/**
 * The single-kind `other` aggregate every default here describes: gating,
 * no groups, whose section and gated geomeans are the same value.
 *
 * `geomeanOverrides` applies once to the shared aggregate behind both
 * `geomean` and `gatedGeomean`, so a caller excluding metrics or widening the
 * band writes that only once. `kindOverrides` remains the escape hatch for
 * sites that need `groups` or divergent geomeans instead.
 */
export function otherKind(
  value: number,
  n: number,
  geomeanOverrides: Partial<GeomeanResult> = {},
  kindOverrides: Partial<KindAggregate> = {},
): KindAggregate {
  const geomean = geomeanOf(value, n, geomeanOverrides);
  return {
    kind: "other",
    geomean,
    groups: [],
    gatedGeomean: geomean,
    ...kindOverrides,
  };
}

/**
 * A multi-candidate comparison with one metric ("decode/time") judged per candidate.
 *
 * With 3 candidates (default): candidate-a improved, candidate-b regressed,
 * candidate-c unstable (band method). With 2: the first two only.
 */
export function multiCandidateResult(candidateCount: 2 | 3 = 3): ComparisonResult {
  const candidates: CandidateComparison[] = [
    createCandidate({
      label: "candidate-a",
      kinds: [otherKind(-10, 1)],
    }),
    createCandidate({
      label: "candidate-b",
      kinds: [otherKind(4, 1)],
    }),
  ];

  const metricCandidates: CandidateMetric[] = [
    {
      median: 90,
      spread: 1,
      verdict: {
        verdict: "improved",
        method: "signed-rank",
        delta: -10,
        n: 10,
        p: 0.002,
        noisePct: 2.5,
        noiseAbs: 2.5,
      },
    },
    {
      median: 104,
      spread: 1,
      verdict: {
        verdict: "regressed",
        method: "signed-rank",
        delta: 4,
        n: 10,
        p: 0.002,
        noisePct: 2.5,
        noiseAbs: 2.5,
      },
    },
  ];

  if (candidateCount === 3) {
    candidates.push(
      createCandidate({
        label: "candidate-c",
        kinds: [otherKind(0, 1)],
      }),
    );
    metricCandidates.push({
      median: 150,
      spread: 3,
      verdict: {
        verdict: "unstable",
        method: "band",
        delta: 50,
        n: 10,
        usableN: 3,
        band: 30,
        noisePct: 30,
        noiseAbs: 30,
      },
    });
  }

  return createComparisonResult({
    baselineLabel: "main",
    candidates,
    metrics: {
      "decode/time": {
        baselineMedian: 100,
        baselineSpread: 1,
        candidates: metricCandidates,
        meta: {
          direction: "lower",
          gating: true,
          exact: false,
          unit: "ns",
          kind: "other",
          shortName: "decode/time",
        },
      },
    },
  });
}

/**
 * A `time` kind and a `memory` kind, the second one informational.
 *
 * `time` holds a two-metric `entity` group beside an ungrouped `warmup`, so
 * its rendered output carries both a group block and a bare row; `memory`
 * holds one ungrouped metric, so its rendered output carries no group rows
 * at all.
 */
export function twoKindMetrics(options: { memoryGates?: boolean; timeGates?: boolean } = {}): {
  [name: string]: MetricEntry;
} {
  const { memoryGates = false, timeGates = true } = options;
  return {
    "entity.alive_check/time": kindMetric({
      kind: "time",
      shortName: "entity.alive_check",
      verdict: "improved",
      delta: -10,
      gating: timeGates,
    }),
    "entity.spawn/time": kindMetric({
      kind: "time",
      shortName: "entity.spawn",
      verdict: "regressed",
      delta: 4,
      gating: timeGates,
    }),
    "warmup/time": kindMetric({
      kind: "time",
      shortName: "warmup",
      verdict: "no-signal",
      delta: 0.3,
      gating: timeGates,
    }),
    "encode/heap": kindMetric({
      kind: "memory",
      shortName: "encode",
      verdict: "improved",
      delta: -7,
      gating: memoryGates,
      unit: "bytes",
    }),
  };
}

/**
 * The gating `time` aggregate: a grouped pair and an ungrouped metric.
 *
 * Both its geomeans carry the band propagated from the metrics behind them,
 * and both sit outside it, so a section rendered from this aggregate shows a
 * band beside every figure it prints.
 */
export function timeKind(overrides: Partial<KindAggregate> = {}): KindAggregate {
  const geomean = geomeanOf(-3.2, 3, { band: 2 });
  return {
    kind: "time",
    geomean,
    groups: [{ group: "entity", geomean: geomeanOf(-3.1, 2, { band: 1.5 }) }],
    gatedGeomean: geomean,
    ...overrides,
  };
}

/**
 * The informational `memory` aggregate: one ungrouped metric, nothing gated.
 *
 * Its geomean keeps the default zero band, so a section rendered from this
 * aggregate shows the figure alone.
 */
export function memoryKind(overrides: Partial<KindAggregate> = {}): KindAggregate {
  return {
    kind: "memory",
    geomean: geomeanOf(-7, 1),
    groups: [],
    ...overrides,
  };
}

/** A single-candidate comparison spanning the gating `time` kind and the informational `memory` kind. */
export function twoKindResult(overrides: Partial<ComparisonResult> = {}): ComparisonResult {
  return createComparisonResult({
    metrics: twoKindMetrics(),
    candidates: [createCandidate({ kinds: [timeKind(), memoryKind()] })],
    configKinds: { memory: { gating: false } },
    ...overrides,
  });
}

/**
 * A two-candidate run spanning a grouped `time` kind and a `memory` kind.
 *
 * A run of a single kind renders flat and drops its group rows, so the
 * second kind is what makes the `entity` group render at all.
 */
export function groupedComparison(): ComparisonResult {
  return createComparisonResult({
    metrics: {
      "entity.alive_check/time": nWayKindMetric({
        kind: "time",
        shortName: "entity.alive_check",
        candidates: [
          { verdict: "improved", delta: -10, median: 90 },
          { verdict: "regressed", delta: 4, median: 104 },
        ],
      }),
      "encode/heap": nWayKindMetric({
        kind: "memory",
        shortName: "encode",
        gating: false,
        candidates: [
          { verdict: "improved", delta: -7, median: 93 },
          { verdict: "improved", delta: -2, median: 98 },
        ],
      }),
    },
    candidates: [
      createCandidate({
        label: "candidate-a",
        kinds: [
          timeKind({
            geomean: geomeanOf(-10, 1),
            groups: [{ group: "entity", geomean: geomeanOf(-10, 1) }],
            gatedGeomean: geomeanOf(-10, 1),
          }),
          memoryKind(),
        ],
      }),
      createCandidate({
        label: "candidate-b",
        kinds: [
          timeKind({
            geomean: geomeanOf(4, 1),
            groups: [{ group: "entity", geomean: geomeanOf(4, 1) }],
            gatedGeomean: geomeanOf(4, 1),
          }),
          memoryKind({ geomean: geomeanOf(-2, 1) }),
        ],
      }),
    ],
    configKinds: { memory: { gating: false } },
  });
}
