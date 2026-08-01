/* eslint-disable typescript/no-unsafe-assignment -- JSON.parse returns any; deep member access is the test's purpose */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
import { describe, it, expect } from "vitest";

import { renderJson } from "../../src/report/json.js";
import {
  createCandidate,
  createComparisonResult,
  signedRankMetric,
  bandMetric,
  exactMetric,
  nWayMetric,
} from "../fixtures/comparison-result.js";

describe("renderJson", () => {
  describe("schema shape", () => {
    it("produces correct schema shape with schemaVersion 1 for a single-candidate report", () => {
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

      expect(json.schemaVersion).toBe(1);
      expect(json.baseline).toBe("main");
      expect(json.candidates).toStrictEqual(["experiment"]);
      expect(json.samples).toBe(10);
      expect(json.adapter).toBe("mitata");
      expect(json).toHaveProperty("metrics");
      expect(json).toHaveProperty("perCandidate");
      expect(json).toHaveProperty("worktrees");
    });

    it("always includes schemaVersion as 1", () => {
      const result = createComparisonResult();

      const json = JSON.parse(renderJson(result));

      expect(json.schemaVersion).toBe(1);
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
    it("includes unit as null when the metric has no unit", () => {
      const result = createComparisonResult({
        metrics: {
          "ops/sec": signedRankMetric({ verdict: "improved", delta: -5 }),
        },
      });

      const json = JSON.parse(renderJson(result));

      expect(json.metrics["ops/sec"].unit).toBeNull();
    });

    it("includes unit when present", () => {
      const result = createComparisonResult({
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -5, unit: "ns" }),
        },
      });

      const json = JSON.parse(renderJson(result));

      expect(json.metrics["decode/time"].unit).toBe("ns");
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

  describe("geomean exclusions", () => {
    it("carries exclusion reasons", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({
            label: "experiment",
            geomean: {
              value: -3.2,
              n: 8,
              excluded: [
                { metric: "jittery/time", reason: "unstable" },
                { metric: "broken/ratio", reason: "undefined-ratio" },
              ],
            },
          }),
        ],
      });

      const json = JSON.parse(renderJson(result));
      const perCandidate = json.perCandidate[0];

      expect(perCandidate.geomean.excluded).toStrictEqual([
        { metric: "jittery/time", reason: "unstable" },
        { metric: "broken/ratio", reason: "undefined-ratio" },
      ]);
      expect(perCandidate.geomean.value).toBe(-3.2);
      expect(perCandidate.geomean.n).toBe(8);
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
                  noiseAbs: 2.5,
                },
              },
            ],
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
                  noiseAbs: 2.5,
                },
              },
              {
                // No data for beta
              },
            ],
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
