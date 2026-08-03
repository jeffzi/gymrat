import { describe, it, expect } from "vitest";

import type { ResolvedMetricMeta } from "../../src/config.js";
import type { KindAggregate } from "../../src/verdict/aggregate.js";
import { computeKindAggregates } from "../../src/verdict/aggregate.js";
import type { MetricVerdict } from "../../src/verdict/verdict.js";
import { metricRecord } from "../fixtures/metrics.js";

/** An exact verdict no exclusion rule drops, so ρ is 1 + delta/100 when lower is better. */
function exactVerdict(delta: number): MetricVerdict {
  return { verdict: delta < 0 ? "improved" : "regressed", method: "exact", delta, n: 1 };
}

/** One metric's contribution to an aggregation run: what it is, and how it moved. */
interface MetricSpec {
  /** Full metric name — the key the exclusion list reports a metric under. */
  name: string;
  shortName: string;
  /** Defaults to "time", so a spec list without kinds describes a single-kind run. */
  kind?: string;
  gating?: boolean;
  /** Percentage delta behind an exact verdict. Ignored when `verdict` is given. */
  delta?: number;
  verdict?: MetricVerdict;
}

/**
 * Verdicts and metadata keyed by metric name, in the order the specs are listed.
 *
 * Both records are built the way the pipeline builds them — without a prototype —
 * so a metric named after an `Object.prototype` member cannot be read off the
 * chain instead of the record.
 */
function buildInputs(specs: readonly MetricSpec[]): {
  verdicts: Record<string, MetricVerdict>;
  metricMeta: Record<string, ResolvedMetricMeta>;
} {
  const verdicts: Record<string, MetricVerdict> = {};
  const metricMeta: Record<string, ResolvedMetricMeta> = {};

  for (const spec of specs) {
    const verdict = spec.verdict ?? exactVerdict(spec.delta ?? 0);
    verdicts[spec.name] = verdict;
    metricMeta[spec.name] = {
      direction: "lower",
      gating: spec.gating ?? true,
      exact: verdict.method === "exact",
      kind: spec.kind ?? "time",
      shortName: spec.shortName,
    };
  }

  return { verdicts: metricRecord(verdicts), metricMeta: metricRecord(metricMeta) };
}

/** The aggregate for `kind`, or a failure naming the kinds that were produced. */
function kindNamed(aggregates: readonly KindAggregate[], kind: string): KindAggregate {
  const found = aggregates.find((aggregate) => aggregate.kind === kind);
  if (!found) {
    throw new Error(
      `no aggregate for kind "${kind}", only: ${aggregates.map((a) => a.kind).join(", ")}`,
    );
  }
  return found;
}

/** The first aggregate, or a failure — most specs describe a single kind. */
function onlyKind(aggregates: readonly KindAggregate[]): KindAggregate {
  const [first] = aggregates;
  if (!first) throw new Error("expected one kind aggregate but got none");
  return first;
}

/**
 * One kind holding one group, whose two metrics differ only in whether they gate.
 *
 * ρ(gating) = 0.9 and ρ(non-gating) = 0.95, so a geomean over both is
 * (0.9 × 0.95)^(1/2) − 1 ≈ −7.54%, and one over the gating metric alone is −10%.
 */
function mixedGatingKind(): ReturnType<typeof buildInputs> {
  return buildInputs([
    { name: "decode-time", shortName: "decode.time", gating: true, delta: -10 },
    { name: "decode-alloc", shortName: "decode.alloc", gating: false, delta: -5 },
  ]);
}

describe("computeKindAggregates", () => {
  describe("aggregate shape", () => {
    it("carries exactly the kind, its gating flag, its geomean and its groups", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "warmup", shortName: "warmup", gating: false, delta: 0 },
      ]);

      const result = computeKindAggregates(verdicts, metricMeta);

      // toStrictEqual also proves the gated geomean is absent when nothing gates.
      expect(result).toStrictEqual([
        {
          kind: "time",
          hasGating: false,
          geomean: { value: 0, n: 1, excluded: [], band: 0 },
          groups: [],
        },
      ]);
    });

    it("returns no aggregates when nothing was measured", () => {
      expect(computeKindAggregates({}, {})).toStrictEqual([]);
    });
  });

  describe("group inference", () => {
    it("groups a kind's metrics by the text before the first dot of their short names", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", shortName: "decode.time", delta: -10 },
        { name: "m2", shortName: "decode.alloc", delta: -5 },
        { name: "m3", shortName: "encode.time", delta: -10 },
      ]);

      const { groups } = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect.soft(groups.map((group) => group.group)).toStrictEqual(["decode", "encode"]);
      expect.soft(groups[0]?.geomean.n).toBe(2);
      expect(groups[1]?.geomean.n).toBe(1);
    });

    it("splits on the first dot only, so a deeper name stays in the outermost group", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", shortName: "decode.utf8.time", delta: -10 },
        { name: "m2", shortName: "decode.time", delta: -10 },
      ]);

      const { groups } = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect.soft(groups.map((group) => group.group)).toStrictEqual(["decode"]);
      expect(groups[0]?.geomean.n).toBe(2);
    });

    it("leaves a metric whose short name has no dot out of every group", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", shortName: "decode.time", delta: -10 },
        { name: "m2", shortName: "warmup", delta: -10 },
      ]);

      const kind = onlyKind(computeKindAggregates(verdicts, metricMeta));

      // The ungrouped metric still counts toward the kind, just not toward a group.
      expect.soft(kind.groups.map((group) => group.group)).toStrictEqual(["decode"]);
      expect.soft(kind.groups[0]?.geomean.n).toBe(1);
      expect(kind.geomean.n).toBe(2);
    });

    it("gives a kind no groups at all when none of its short names is dotted", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", shortName: "alpha", delta: -10 },
        { name: "m2", shortName: "beta", delta: -5 },
      ]);

      const { groups } = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect(groups).toStrictEqual([]);
    });

    it("groups per kind, so a dotted name in one kind leaves another kind flat", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", kind: "time", shortName: "decode.time", delta: -10 },
        { name: "m2", kind: "memory", shortName: "heap", delta: -10 },
      ]);

      const result = computeKindAggregates(verdicts, metricMeta);

      expect
        .soft(kindNamed(result, "time").groups.map((group) => group.group))
        .toStrictEqual(["decode"]);
      expect(kindNamed(result, "memory").groups).toStrictEqual([]);
    });
  });

  describe("ordering", () => {
    it("orders kinds and groups by the first metric that mentions them", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", kind: "time", shortName: "encode.time", delta: -10 },
        { name: "m2", kind: "memory", shortName: "encode.heap", delta: -10 },
        { name: "m3", kind: "time", shortName: "decode.time", delta: -10 },
      ]);

      const result = computeKindAggregates(verdicts, metricMeta);

      expect.soft(result.map((aggregate) => aggregate.kind)).toStrictEqual(["time", "memory"]);
      expect(kindNamed(result, "time").groups.map((group) => group.group)).toStrictEqual([
        "encode",
        "decode",
      ]);
    });
  });

  describe("the kind geomean", () => {
    it("covers the kind's gating and non-gating metrics alike", () => {
      const { verdicts, metricMeta } = mixedGatingKind();

      const { geomean } = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect.soft(geomean.n).toBe(2);
      expect(geomean.value).toBeCloseTo(-7.54, 1);
    });

    it.each([
      {
        reason: "unstable",
        verdict: {
          verdict: "unstable",
          method: "band",
          delta: -50,
          n: 4,
          usableN: 4,
          band: 250,
          noisePct: 250,
          noiseAbs: 25,
        } satisfies MetricVerdict,
      },
      { reason: "undefined-ratio", verdict: exactVerdict(Number.NaN) },
      // ρ = 1 + (−150/100) = −0.5, whose logarithm is not a real number
      { reason: "infinite-rho", verdict: exactVerdict(-150) },
    ])("leaves a $reason metric out, the way computeGeomean does", ({ reason, verdict }) => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "bad", shortName: "bad", verdict },
        { name: "good", shortName: "good", delta: -5 },
      ]);

      const { geomean } = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect.soft(geomean.n).toBe(1);
      expect.soft(geomean.excluded).toStrictEqual([{ metric: "bad", reason }]);
      expect(geomean.value).toBeCloseTo(-5, 5);
    });
  });

  describe("the gated geomean", () => {
    it("covers only the kind's gating metrics", () => {
      const { verdicts, metricMeta } = mixedGatingKind();

      const kind = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect.soft(kind.hasGating).toBe(true);
      expect.soft(kind.gatedGeomean?.n).toBe(1);
      expect(kind.gatedGeomean?.value).toBeCloseTo(-10, 5);
    });

    it("is decided per kind rather than across the run", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "m1", kind: "time", shortName: "time", gating: true, delta: -10 },
        { name: "m2", kind: "memory", shortName: "heap", gating: false, delta: -10 },
      ]);

      const result = computeKindAggregates(verdicts, metricMeta);

      expect.soft(kindNamed(result, "time").hasGating).toBe(true);
      expect.soft(kindNamed(result, "memory").hasGating).toBe(false);
      expect(kindNamed(result, "memory").gatedGeomean).toBeUndefined();
    });
  });

  describe("a group geomean", () => {
    it("covers the group's gating and non-gating metrics alike", () => {
      const { verdicts, metricMeta } = mixedGatingKind();

      const { groups } = onlyKind(computeKindAggregates(verdicts, metricMeta));

      expect.soft(groups[0]?.geomean.n).toBe(2);
      expect(groups[0]?.geomean.value).toBeCloseTo(-7.54, 1);
    });
  });
});
