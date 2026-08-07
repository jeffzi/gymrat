/* eslint-disable typescript/no-unsafe-assignment -- JSON.parse returns any; deep member access is the test's purpose */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
import { afterEach, describe, it, expect, vi } from "vitest";

import { renderJson, renderMeasureJson } from "../../src/report/json.js";
import type { ComparisonResult } from "../../src/report/types.js";
import {
  createCandidate,
  createComparisonResult,
  geomeanOf,
  kindMetric,
  metricMeta,
  otherKind,
  signedRankMetric,
  bandMetric,
  exactMetric,
  nWayMetric,
  singleSampleResult,
} from "../fixtures/comparison-result.js";
import {
  createMeasurementResult,
  measuredMetric,
  twoKindMeasurement,
} from "../fixtures/measurement-result.js";

/**
 * A run spanning a gating `time` kind and an informational `memory` kind.
 *
 * `time` holds a grouped metric and lost two metrics to exclusion rules, so its
 * aggregate exercises groups and exclusions at once; `memory` holds one
 * ungrouped metric and gates nothing, so it pins the no-groups, no-gated case.
 */
function twoKindWithExclusions(): ComparisonResult {
  return createComparisonResult({
    candidates: [
      createCandidate({
        label: "experiment",
        kinds: [
          {
            kind: "time",
            hasGating: true,
            geomean: geomeanOf(-3.2, 2, {
              excluded: [
                { metric: "jittery/time", reason: "unstable" },
                { metric: "broken/ratio", reason: "undefined-ratio" },
              ],
            }),
            groups: [{ group: "entity", geomean: geomeanOf(-3.1, 2) }],
            gatedGeomean: geomeanOf(-3.2, 2),
          },
          { kind: "memory", hasGating: false, geomean: geomeanOf(-7, 1), groups: [] },
        ],
      }),
    ],
    metrics: {
      "entity.alive_check/time": kindMetric({
        kind: "time",
        shortName: "entity.alive_check",
        verdict: "improved",
        delta: -10,
      }),
      "encode/heap": kindMetric({
        kind: "memory",
        shortName: "encode",
        verdict: "improved",
        delta: -7,
        gating: false,
        unit: "bytes",
      }),
    },
  });
}

describe("renderJson", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("schema shape", () => {
    it("produces correct schema shape with schemaVersion 2 for a single-candidate report", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [createCandidate({ label: "experiment" })],
        samples: 10,
        adapter: "mitata",
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -10 }),
        },
      });

      const json = JSON.parse(renderJson(result));

      expect(json.schemaVersion).toBe(2);
      expect(json.baseline).toBe("main");
      expect(json.candidates).toStrictEqual(["experiment"]);
      expect(json.samples).toBe(10);
      expect(json.adapter).toBe("mitata");
      expect(json).toHaveProperty("metrics");
      expect(json).toHaveProperty("perCandidate");
      expect(json).toHaveProperty("worktrees");
    });
  });

  describe("multi-candidate support", () => {
    it("includes all candidates in order", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({ label: "alpha" }),
          createCandidate({ label: "beta" }),
          createCandidate({ label: "gamma" }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "improved", delta: -10, median: 90 },
            { verdict: "regressed", delta: 5, median: 105 },
            { verdict: "no-signal", delta: 0.1, median: 100.1 },
          ]),
        },
      });

      const json = JSON.parse(renderJson(result));

      expect(json.candidates).toStrictEqual(["alpha", "beta", "gamma"]);
      expect(json.perCandidate).toHaveLength(3);
      expect(json.perCandidate[0].label).toBe("alpha");
      expect(json.perCandidate[1].label).toBe("beta");
      expect(json.perCandidate[2].label).toBe("gamma");
    });
  });

  describe("metric verdict methods", () => {
    it("populates p and leaves band as null for signed-rank verdicts", () => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -10, p: 0.003 }),
        },
      });

      const json = JSON.parse(renderJson(result));
      const candidate = json.metrics["decode/time"].candidates[0];

      expect(candidate.method).toBe("signed-rank");
      expect(candidate.p).toBe(0.003);
      expect(candidate.band).toBeNull();
    });

    it("populates band and leaves p as null for band verdicts", () => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": bandMetric({ verdict: "no-signal", delta: -1, noisePct: 3.5 }),
        },
      });

      const json = JSON.parse(renderJson(result));
      const candidate = json.metrics["decode/time"].candidates[0];

      expect(candidate.method).toBe("band");
      expect(candidate.band).toBe(3.5);
      expect(candidate.p).toBeNull();
    });

    // The text report calls a single-pair verdict inconclusive, which is a
    // display decision: what the engine computed is what gets serialized.
    it("stores a single-pair verdict as the no-signal band verdict it is", () => {
      const json = JSON.parse(renderJson(singleSampleResult()));
      const candidate = json.metrics["decode/time"].candidates[0];

      expect.soft(candidate.verdict).toBe("no-signal");
      expect.soft(candidate.method).toBe("band");
      expect.soft(candidate.noisePct).toBe(0.5);
      expect(candidate.band).toBe(0.5);
    });

    it("has noisePct, p, and band all as null for exact verdicts", () => {
      const result = createComparisonResult({
        metrics: {
          "alloc/heap": exactMetric({ delta: -7.9 }),
        },
      });

      const json = JSON.parse(renderJson(result));
      const candidate = json.metrics["alloc/heap"].candidates[0];

      expect(candidate.method).toBe("exact");
      expect(candidate.noisePct).toBeNull();
      expect(candidate.p).toBeNull();
      expect(candidate.band).toBeNull();
    });
  });

  describe("metric metadata", () => {
    it.each([
      { unit: undefined, expected: null },
      { unit: "ns", expected: "ns" },
    ] as const)("serializes unit as $expected when unit is $unit", ({ unit, expected }) => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -5, unit }),
        },
      });

      const json = JSON.parse(renderJson(result));

      expect(json.metrics["decode/time"].unit).toBe(expected);
    });

    it("includes direction and gating from metric meta", () => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -5, gating: false }),
        },
      });

      const json = JSON.parse(renderJson(result));
      const metric = json.metrics["decode/time"];

      expect(metric.direction).toBe("lower");
      expect(metric.gating).toBe(false);
    });

    it("includes baseline median and spread", () => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 200,
            baselineSpread: 3.5,
          }),
        },
      });

      const json = JSON.parse(renderJson(result));
      const metric = json.metrics["decode/time"];

      expect(metric.baseline.median).toBe(200);
      expect(metric.baseline.spreadPct).toBe(3.5);
    });
  });

  describe("when a candidate spans several kinds", () => {
    it("carries one entry per kind, with its groups, section geomean and gated geomean", () => {
      const json = JSON.parse(renderJson(twoKindWithExclusions()));

      expect(json.perCandidate[0].kinds).toStrictEqual([
        {
          kind: "time",
          hasGating: true,
          geomean: {
            value: -3.2,
            n: 2,
            excluded: [
              { metric: "jittery/time", reason: "unstable" },
              { metric: "broken/ratio", reason: "undefined-ratio" },
            ],
            band: 0,
          },
          groups: [{ group: "entity", geomean: { value: -3.1, n: 2, excluded: [], band: 0 } }],
          gatedGeomean: { value: -3.2, n: 2, excluded: [], band: 0 },
        },
        {
          kind: "memory",
          hasGating: false,
          geomean: { value: -7, n: 1, excluded: [], band: 0 },
          groups: [],
          gatedGeomean: null,
        },
      ]);
    });

    it("leaves no blended geomean beside the kinds", () => {
      const json = JSON.parse(renderJson(twoKindWithExclusions()));

      expect(json.perCandidate[0]).not.toHaveProperty("geomean");
    });

    it("serializes a single-kind run through the same shape, with one entry", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({
            label: "experiment",
            kinds: [otherKind(-3.2, 2)],
          }),
        ],
        metrics: { "decode/time": signedRankMetric({ verdict: "improved", delta: -10 }) },
      });

      const json = JSON.parse(renderJson(result));

      expect(json.perCandidate[0].kinds).toStrictEqual([
        {
          kind: "other",
          hasGating: true,
          geomean: { value: -3.2, n: 2, excluded: [], band: 0 },
          groups: [],
          gatedGeomean: { value: -3.2, n: 2, excluded: [], band: 0 },
        },
      ]);
    });
  });

  describe("when reporting a metric's kind and group", () => {
    it.each([
      {
        shape: "a grouped metric",
        metric: "entity.alive_check/time",
        kind: "time",
        group: "entity",
      },
      { shape: "an ungrouped metric", metric: "encode/heap", kind: "memory", group: null },
    ])("reports the kind and group of $shape", ({ metric, kind, group }) => {
      const json = JSON.parse(renderJson(twoKindWithExclusions()));

      expect.soft(json.metrics[metric].kind).toBe(kind);
      expect(json.metrics[metric].group).toBe(group);
    });
  });

  describe("when the ambient environment forces color", () => {
    it("emits no ANSI escape sequences, however the ambient environment forces color", () => {
      vi.stubEnv("FORCE_COLOR", "1");

      expect(renderJson(twoKindWithExclusions())).not.toMatch(/\x1b\[/);
    });
  });

  describe("verdict counts", () => {
    it("tallies verdicts per candidate", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10 }),
          "also-faster/time": signedRankMetric({ verdict: "improved", delta: -5 }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 8 }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: 5 }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.2 }),
        },
      });

      const json = JSON.parse(renderJson(result));
      const counts = json.perCandidate[0].verdictCounts;

      expect(counts).toStrictEqual({
        improved: 2,
        regressed: 1,
        unstable: 1,
        noSignal: 1,
      });
    });
  });

  describe("missing metric data", () => {
    it("renders null for absent baseline and candidate fields", () => {
      const result = createComparisonResult({
        candidates: [createCandidate({ label: "alpha" })],
        metrics: {
          "sparse/time": {
            baselineMedian: undefined,
            baselineSpread: undefined,
            candidates: [
              {
                median: undefined,
                spread: undefined,
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -5,
                  n: 10,
                  p: 0.01,
                  noisePct: 2.5,
                  noiseAbs: 3.5,
                },
              },
            ],
            meta: metricMeta("sparse/time", { unit: "ns" }),
          },
        },
      });

      const json = JSON.parse(renderJson(result));
      const metric = json.metrics["sparse/time"];

      expect.soft(metric.baseline.median).toBeNull();
      expect.soft(metric.baseline.spreadPct).toBeNull();
      expect.soft(metric.candidates[0].median).toBeNull();
      expect.soft(metric.candidates[0].spreadPct).toBeNull();
    });

    it("renders nulls for a candidate with no data for a metric", () => {
      const result = createComparisonResult({
        candidates: [createCandidate({ label: "alpha" }), createCandidate({ label: "beta" })],
        metrics: {
          "decode/time": {
            baselineMedian: 100,
            baselineSpread: 1,
            candidates: [
              {
                median: 90,
                spread: 1,
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -10,
                  n: 10,
                  p: 0.01,
                  noisePct: 2.5,
                  noiseAbs: 3.5,
                },
              },
              {
                // No data for beta
              },
            ],
            meta: metricMeta("decode/time", { unit: "ns" }),
          },
        },
      });

      const json = JSON.parse(renderJson(result));
      const betaCandidate = json.metrics["decode/time"].candidates[1];

      expect(betaCandidate.label).toBe("beta");
      expect(betaCandidate.median).toBeNull();
      expect(betaCandidate.spreadPct).toBeNull();
      expect(betaCandidate.verdict).toBeNull();
      expect(betaCandidate.method).toBeNull();
      expect(betaCandidate.delta).toBeNull();
      expect(betaCandidate.noisePct).toBeNull();
      expect(betaCandidate.p).toBeNull();
      expect(betaCandidate.band).toBeNull();
    });

    it("keeps what a candidate measured when no verdict could be reached", () => {
      const result = createComparisonResult({
        candidates: [createCandidate({ label: "alpha" }), createCandidate({ label: "beta" })],
        metrics: {
          "decode/time": {
            baselineMedian: 100,
            baselineSpread: 1,
            candidates: [
              {
                median: 90,
                spread: 1,
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -10,
                  n: 10,
                  p: 0.01,
                  noisePct: 2.5,
                  noiseAbs: 3.5,
                },
              },
              // Measured on every round, but never paired with the baseline.
              { median: 95, spread: 3 },
            ],
            meta: metricMeta("decode/time", { unit: "ns" }),
          },
        },
      });

      const json = JSON.parse(renderJson(result));
      const betaCandidate = json.metrics["decode/time"].candidates[1];

      expect.soft(betaCandidate.median).toBe(95);
      expect.soft(betaCandidate.spreadPct).toBe(3);
      expect(betaCandidate.verdict).toBeNull();
    });
  });

  describe("worktrees section", () => {
    it("reflects cleanup state with no issues", () => {
      const result = createComparisonResult({
        worktreesRemoved: 2,
        worktreesLeftBehind: [],
        worktreePruneError: undefined,
      });

      const json = JSON.parse(renderJson(result));

      expect(json.worktrees).toStrictEqual({
        removed: 2,
        leftBehind: [],
        pruneError: null,
      });
    });

    it("includes left-behind worktrees and prune errors", () => {
      const result = createComparisonResult({
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "contains modified files" }],
        worktreePruneError: "fatal: prune failed",
      });

      const json = JSON.parse(renderJson(result));

      expect(json.worktrees.removed).toBe(1);
      expect(json.worktrees.leftBehind).toStrictEqual([
        { path: "/tmp/gymrat-abc", reason: "contains modified files" },
      ]);
      expect(json.worktrees.pruneError).toBe("fatal: prune failed");
    });
  });

  describe("JSON validity", () => {
    it("produces output that JSON.parse roundtrips cleanly", () => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "alloc/heap": exactMetric({ delta: -5, unit: "bytes" }),
        },
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-x", error: "locked" }],
        worktreePruneError: "prune failed",
      });

      const output = renderJson(result);
      const parsed = JSON.parse(output);
      const encoded = JSON.stringify(parsed, null, 2);

      expect(encoded).toBe(output);
    });
  });
});

describe("renderMeasureJson", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("schema shape", () => {
    it("names the run under its own schemaVersion, starting at 1", () => {
      const result = createMeasurementResult({
        label: "experiment",
        samples: 10,
        adapter: "mitata",
        metrics: { "decode/time": measuredMetric({ unit: "ns" }) },
      });

      const json = JSON.parse(renderMeasureJson(result));

      expect.soft(json.schemaVersion).toBe(1);
      expect.soft(json.label).toBe("experiment");
      expect.soft(json.samples).toBe(10);
      expect.soft(json.adapter).toBe("mitata");
      expect.soft(json).toHaveProperty("metrics");
      expect(json).toHaveProperty("worktrees");
    });

    it("leaves out everything a comparison alone has to say", () => {
      const json = JSON.parse(renderMeasureJson(twoKindMeasurement()));

      expect.soft(json).not.toHaveProperty("baseline");
      expect.soft(json).not.toHaveProperty("candidates");
      expect(json).not.toHaveProperty("perCandidate");
    });
  });

  describe("metric entries", () => {
    it("carries a grouped metric's measurement beside the metadata behind it", () => {
      const json = JSON.parse(renderMeasureJson(twoKindMeasurement()));

      expect(json.metrics["entity.alive_check/time"]).toStrictEqual({
        median: 100,
        spreadPct: 1,
        unit: "ns",
        direction: "lower",
        gating: true,
        exact: false,
        kind: "time",
        group: "entity",
      });
    });

    it("reports no group for a metric whose name names none", () => {
      const json = JSON.parse(renderMeasureJson(twoKindMeasurement()));

      expect.soft(json.metrics["encode/heap"].group).toBeNull();
      expect(json.metrics["encode/heap"].kind).toBe("memory");
    });

    it.each([
      { field: "median", metric: { median: undefined, spread: 1 } },
      { field: "spreadPct", metric: { median: 100, spread: undefined } },
    ])("renders null for an absent $field", ({ field, metric }) => {
      const result = createMeasurementResult({
        metrics: { "sparse/time": measuredMetric({ ...metric, unit: "ns" }) },
      });

      const json = JSON.parse(renderMeasureJson(result));

      expect(json.metrics["sparse/time"][field]).toBeNull();
    });

    it("renders a metric with no unit as a null unit", () => {
      const result = createMeasurementResult({
        metrics: { "throughput/ops": measuredMetric() },
      });

      const json = JSON.parse(renderMeasureJson(result));

      expect(json.metrics["throughput/ops"].unit).toBeNull();
    });
  });

  describe("worktrees section", () => {
    it("reflects cleanup state with no issues", () => {
      const json = JSON.parse(renderMeasureJson(createMeasurementResult({ worktreesRemoved: 2 })));

      expect(json.worktrees).toStrictEqual({ removed: 2, leftBehind: [], pruneError: null });
    });

    it("includes left-behind worktrees and prune errors", () => {
      const result = createMeasurementResult({
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "contains modified files" }],
        worktreePruneError: "fatal: prune failed",
      });

      const json = JSON.parse(renderMeasureJson(result));

      expect(json.worktrees).toStrictEqual({
        removed: 1,
        leftBehind: [{ path: "/tmp/gymrat-abc", reason: "contains modified files" }],
        pruneError: "fatal: prune failed",
      });
    });
  });

  describe("when the ambient environment forces color", () => {
    it("emits no ANSI escape sequences, however the ambient environment forces color", () => {
      vi.stubEnv("FORCE_COLOR", "1");

      expect(renderMeasureJson(twoKindMeasurement())).not.toMatch(/\x1b\[/);
    });
  });

  describe("JSON validity", () => {
    it("produces output that JSON.parse roundtrips cleanly", () => {
      const result = twoKindMeasurement({
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-x", error: "locked" }],
        worktreePruneError: "prune failed",
      });

      const output = renderMeasureJson(result);
      const encoded = JSON.stringify(JSON.parse(output), null, 2);

      expect(encoded).toBe(output);
    });
  });
});
