import { afterEach, describe, expect, it, vi } from "vitest";

import { renderMarkdown } from "../../src/report/markdown.js";
import type { ComparisonResult } from "../../src/report/types.js";
import {
  createCandidate,
  createComparisonResult,
  metricMeta,
  multiCandidateResult,
  signedRankMetric,
  bandMetric,
  exactMetric,
  nWayMetric,
} from "../fixtures/comparison-result.js";

/** Extract lines that look like GFM table rows (start with |). */
function tableRows(output: string): string[] {
  return output.split("\n").filter((line) => line.startsWith("|"));
}

/** Extract the header separator row from a GFM table. */
function headerSeparator(output: string): string | undefined {
  return tableRows(output).find((line) => /^\|[\s:|-]+\|$/.test(line));
}

describe("renderMarkdown", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });
  describe("when rendering a single-candidate report with mixed verdicts", () => {
    function mixedResult(): ComparisonResult {
      return createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({
            label: "perf/faster-decode",
            geomean: { value: -5.8, n: 3, excluded: [] },
          }),
        ],
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.2, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
        },
      });
    }

    it("produces a summary line counting each verdict class", () => {
      const output = renderMarkdown(mixedResult());

      expect(output).toContain("✓ 1 improved");
      expect(output).toContain("✗ 1 regressed");
      expect(output).toContain("~ 1 within noise");
    });

    it("includes a highlights section as a markdown list", () => {
      const output = renderMarkdown(mixedResult());

      // Regressions first, then improvements
      const lines = output.split("\n");
      const regLine = lines.find((l) => l.includes("slower/time"));
      const impLine = lines.find((l) => l.includes("faster/time"));

      expect(regLine).toContain("✗");
      expect(regLine).toContain("+2.2%");
      expect(impLine).toContain("✓");
      expect(impLine).toContain("-17.5%");

      // Regression comes before improvement
      expect(lines.indexOf(regLine!)).toBeLessThan(lines.indexOf(impLine!));
    });

    it("renders a GFM table with pipe-delimited columns and a header separator", () => {
      const output = renderMarkdown(mixedResult());
      const rows = tableRows(output);

      // At least header + separator + metric rows + geomean
      expect(rows.length).toBeGreaterThanOrEqual(5);

      // Header row has expected columns, with every variant name in backticks
      const header = rows[0]!;
      expect(header).toContain("Metric");
      expect(header).toContain("`main`");
      expect(header).toContain("`perf/faster-decode`");
      expect(header).toContain("vs `main`");

      // Separator row exists with alignment markers
      const sep = headerSeparator(output);
      expect(sep).toBeDefined();
    });

    it.each([
      { mode: "plain", options: {} },
      { mode: "verbose", options: { verbose: true } },
    ])("spells out no blockquote legend in $mode mode", ({ options }) => {
      const output = renderMarkdown(mixedResult(), options);

      expect.soft(output).not.toContain("✓ improved · ✗ regressed");
      expect(output).not.toContain("candidates are judged against");
    });

    it("names the signed-rank test in a method line only when verbose", () => {
      expect.soft(renderMarkdown(mixedResult())).not.toContain("Wilcoxon signed-rank");
      expect(renderMarkdown(mixedResult(), { verbose: true })).toContain("Wilcoxon signed-rank");
    });
  });

  describe("when rendering a multi-candidate report", () => {
    it("produces per-candidate summary lines prefixed with the label", () => {
      const output = renderMarkdown(multiCandidateResult(2));

      const lines = output.split("\n");
      const candidateALine = lines.find((l) => l.includes("candidate-a") && l.includes("improved"));
      const candidateBLine = lines.find(
        (l) => l.includes("candidate-b") && l.includes("regressed"),
      );

      expect(candidateALine).toBeDefined();
      expect(candidateBLine).toBeDefined();
    });

    it("groups highlights by candidate under sub-headers", () => {
      const output = renderMarkdown(multiCandidateResult(2));

      expect(output).toContain("**`candidate-a`**");
      expect(output).toContain("**`candidate-b`**");

      // Each candidate's highlight appears
      const lines = output.split("\n");
      const aHeaderIdx = lines.findIndex((l) => l.includes("**`candidate-a`**"));
      const bHeaderIdx = lines.findIndex((l) => l.includes("**`candidate-b`**"));
      expect(aHeaderIdx).toBeLessThan(bHeaderIdx);
    });

    it("heads each candidate column with the label alone, in backticks", () => {
      const header = tableRows(renderMarkdown(multiCandidateResult(2)))[0]!;

      expect.soft(header).toContain("`candidate-a`");
      expect.soft(header).toContain("`candidate-b`");
      expect(header).not.toContain("vs main");
    });

    it("wraps the candidate label in backticks in each summary line", () => {
      const summary = renderMarkdown(multiCandidateResult(2))
        .split("\n")
        .find((line) => line.includes("candidate-a") && line.includes("improved"));

      expect(summary).toContain("`candidate-a`");
    });

    it("renders multi-candidate table with combined value+verdict cells", () => {
      const output = renderMarkdown(multiCandidateResult(2));
      const rows = tableRows(output);

      // Data row should have combined value+verdict cells
      const dataRow = rows.find((r) => r.includes("decode/time"));
      expect(dataRow).toContain("✓");
      expect(dataRow).toContain("-10.0%");
      expect(dataRow).toContain("✗");
      expect(dataRow).toContain("+4.0%");
    });

    it("renders a value-only cell when a candidate has a measurement but no verdict", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: 0, n: 0, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 0, n: 0, excluded: [] } }),
        ],
        metrics: {
          "decode/time": {
            baselineMedian: 1000,
            baselineSpread: 1,
            candidates: [
              {
                median: 900,
                spread: 2,
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
              { median: 950, spread: 3 },
            ],
            meta: metricMeta("decode/time", { unit: "ns" }),
          },
        },
      });

      const output = renderMarkdown(result);
      const dataRow = tableRows(output).find((r) => r.includes("decode/time"));

      const cells = dataRow!.split("|").map((c) => c.trim());
      const candidateBCell = cells[4];

      expect.soft(candidateBCell).toContain("950ns");
      expect.soft(candidateBCell).not.toContain("✓");
      expect.soft(candidateBCell).not.toContain("✗");
      expect(dataRow).toContain("✓");
    });

    it("renders an empty cell when a candidate never reported the metric", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: 0, n: 0, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 0, n: 0, excluded: [] } }),
        ],
        metrics: {
          "decode/time": {
            baselineMedian: 1000,
            baselineSpread: 1,
            candidates: [
              {
                median: 900,
                spread: 2,
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
              {},
            ],
            meta: metricMeta("decode/time", { unit: "ns" }),
          },
        },
      });

      const output = renderMarkdown(result);
      const dataRow = tableRows(output).find((r) => r.includes("decode/time"))!;
      const cells = dataRow.split("|").map((c) => c.trim());
      const candidateBCell = cells[4];

      expect(candidateBCell).toBe("");
    });
  });

  describe("when a variant label overflows the display width", () => {
    it("truncates the label wherever it prints, leaving metric names whole", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [createCandidate({ label: "feature/entity-spawn-fastpath" })],
        metrics: {
          "decode/an-extremely-long-metric-name/time": signedRankMetric({
            verdict: "improved",
            delta: -10,
            unit: "ns",
          }),
        },
      });

      const output = renderMarkdown(result);

      expect.soft(output).not.toContain("feature/entity-spawn-fastpath");
      expect.soft(output).toContain("`feature/en…-fastpath`");
      expect(output).toContain("decode/an-extremely-long-metric-name/time");
    });
  });

  describe("when within-noise rows are present", () => {
    it("places them inside a <details> block", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
        },
        candidates: [createCandidate({ geomean: { value: -5, n: 1, excluded: [] } })],
      });

      const output = renderMarkdown(result);

      expect(output).toContain("<details>");
      expect(output).toContain("</details>");
      expect(output).toContain("<summary>");

      // The within-noise metric row should be inside the details block
      const detailsStart = output.indexOf("<details>");
      const detailsEnd = output.indexOf("</details>");
      const flatIdx = output.indexOf("flat/time", detailsStart);
      expect(flatIdx).toBeGreaterThan(detailsStart);
      expect(flatIdx).toBeLessThan(detailsEnd);
    });

    it("includes the count in the details summary", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "flat2/time": signedRankMetric({ verdict: "no-signal", delta: -0.1, unit: "ns" }),
        },
        candidates: [createCandidate({ geomean: { value: -5, n: 1, excluded: [] } })],
      });

      const output = renderMarkdown(result);

      expect(output).toContain("2 within noise");
    });

    it("counts unstable rows in the details summary alongside within-noise", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50, noisePct: 30 }),
        },
        candidates: [createCandidate({ geomean: { value: -5, n: 1, excluded: [] } })],
      });

      const output = renderMarkdown(result);

      expect(output).toContain("<summary>");
      const summaryMatch = output.match(/<summary>(.*?)<\/summary>/);
      expect(summaryMatch?.[1]).toContain("within noise");
      expect(summaryMatch?.[1]).toContain("unstable");
    });
  });

  describe("when ties starved the signed-rank test", () => {
    /** A run whose `tied/heap` metric moved too little to break any pair apart. */
    function identicalResult(): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 0 }),
        },
        candidates: [createCandidate({ geomean: { value: -5, n: 1, excluded: [] } })],
      });
    }

    it("marks the table cell identical rather than within noise", () => {
      const row = tableRows(renderMarkdown(identicalResult())).find((r) => r.includes("tied/heap"));

      expect(row).toContain("=  -0.5%");
    });

    it("tallies it apart from the metrics that are merely within noise", () => {
      const output = renderMarkdown(identicalResult());

      expect(output).toContain("≈ 0 unstable · = 1 identical · ~ 1 within noise");
    });

    it("counts it on its own in the details summary", () => {
      const output = renderMarkdown(identicalResult());

      expect(output.match(/<summary>(.*?)<\/summary>/)?.[1]).toBe("1 within noise / 1 identical");
    });

    it("collapses it into the details block", () => {
      const output = renderMarkdown(identicalResult());

      const detailsStart = output.indexOf("<details>");
      const tiedIdx = output.indexOf("tied/heap");
      expect.soft(tiedIdx).toBeGreaterThan(detailsStart);
      expect(tiedIdx).toBeLessThan(output.indexOf("</details>"));
    });

    it("leaves it out of the highlights list", () => {
      const bullets = renderMarkdown(identicalResult())
        .split("\n")
        .filter((line) => line.startsWith("- "));

      expect(bullets).toStrictEqual(["- ✓ faster/time  -10.0%"]);
    });

    it("says nothing more about it in the method footer", () => {
      expect(renderMarkdown(identicalResult())).not.toContain("close-to-identical");
    });
  });

  describe("when rendering highlight evidence", () => {
    it("shows (exact) for exact verdicts", () => {
      const result = createComparisonResult({
        metrics: {
          "heap/size": exactMetric({ delta: -7.9 }),
        },
      });

      const output = renderMarkdown(result);
      const highlightLine = output.split("\n").find((l) => l.includes("heap/size"));

      expect(highlightLine).toContain("(exact)");
    });

    it("shows noise band for unstable verdicts", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });

      const output = renderMarkdown(result);
      const highlightLine = output.split("\n").find((l) => l.includes("jittery/time"));

      expect(highlightLine).toContain("unstable");
      expect(highlightLine).toContain("noise");
      expect(highlightLine).toContain("±30.0%");
    });

    it("states the noise in absolute units once it outgrows the median", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/heap": signedRankMetric({
            verdict: "unstable",
            delta: 5,
            baselineMedian: 5,
            noisePct: 7620,
            noiseAbs: 381,
            unit: "bytes",
          }),
        },
      });

      const output = renderMarkdown(result);
      const bullet = output.split("\n").find((line) => line.startsWith("- "));

      expect(bullet).toBe("- ≈ jittery/heap  unstable  ±381B noise on a 5B median");
    });

    it("shows no trailing evidence for signed-rank/band improved or regressed", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
        },
      });

      const output = renderMarkdown(result);
      const highlightLine = output
        .split("\n")
        .find((l) => l.includes("faster/time") && l.includes("✓"));

      expect(highlightLine).toBeDefined();
      expect(highlightLine).toContain("-17.5%");
      expect(highlightLine).not.toContain("(exact)");
      expect(highlightLine).not.toContain("noise");
    });
  });

  describe("when unstable metrics reach the highlights", () => {
    it("closes the list with an italic futility note", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });

      const lines = renderMarkdown(result).split("\n");
      const noteIndex = lines.findIndex((line) => line.includes("won't stabilize"));

      expect.soft(lines[noteIndex]).toBe("_unstable metrics won't stabilize with more samples_");
      expect(lines[noteIndex - 1]).toBe("- ≈ jittery/time  unstable  noise ±30.0%");
    });

    it("states it once for the whole section, however many candidates carry one", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: 0, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 0, n: 1, excluded: [] } }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "unstable", delta: 5, median: 105 },
            { verdict: "unstable", delta: -3, median: 97 },
          ]),
        },
      });

      const notes = renderMarkdown(result)
        .split("\n")
        .filter((line) => line.includes("won't stabilize"));

      expect(notes).toHaveLength(1);
    });

    it("says nothing about stabilizing when no highlight is unstable", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
        },
      });

      expect(renderMarkdown(result)).not.toContain("won't stabilize");
    });

    it("states a table spread wider than the median in absolute units", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/heap": signedRankMetric({
            verdict: "unstable",
            delta: 5,
            baselineMedian: 5,
            baselineSpread: 7620,
            unit: "bytes",
          }),
        },
      });

      const row = tableRows(renderMarkdown(result)).find((r) => r.includes("jittery/heap"));

      expect(row).toContain("5B ± 381B");
    });
  });

  describe("when the metric cells of a column print at different widths", () => {
    it("joins each cell's magnitude and spread with a single space, padding neither", () => {
      const result = createComparisonResult({
        metrics: {
          "first/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 162000,
            baselineSpread: 9,
            unit: "ns",
          }),
          "second/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 29200,
            baselineSpread: 12,
            unit: "ns",
          }),
        },
      });

      const rows = tableRows(renderMarkdown(result));

      expect.soft(rows.find((row) => row.includes("first/metric"))).toContain("| 162.0µs ± 9% |");
      expect(rows.find((row) => row.includes("second/metric"))).toContain("| 29.2µs ± 12% |");
    });
  });

  describe("when rendering a multi-candidate report with quiet rows", () => {
    it("collapses within-noise and unstable rows into a details block", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: -10, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 4, n: 1, excluded: [] } }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "improved", delta: -10, median: 90 },
            { verdict: "regressed", delta: 4, median: 104 },
          ]),
          "flat/time": nWayMetric([
            { verdict: "no-signal", delta: 0.1, median: 100 },
            { verdict: "no-signal", delta: -0.2, median: 100 },
          ]),
          "jittery/time": {
            baselineMedian: 100,
            baselineSpread: 1,
            candidates: [
              {
                median: 105,
                spread: 1,
                verdict: {
                  verdict: "unstable",
                  method: "signed-rank",
                  delta: 5,
                  n: 10,
                  p: 0.4,
                  noisePct: 300,
                  noiseAbs: 300,
                },
              },
              {
                median: 97,
                spread: 1,
                verdict: {
                  verdict: "unstable",
                  method: "signed-rank",
                  delta: -3,
                  n: 10,
                  p: 0.5,
                  noisePct: 300,
                  noiseAbs: 300,
                },
              },
            ],
            meta: metricMeta("jittery/time", { unit: "ns" }),
          },
        },
      });

      const output = renderMarkdown(result);

      expect(output).toContain("<details>");
      const summaryMatch = output.match(/<summary>(.*?)<\/summary>/);
      expect(summaryMatch?.[1]).toContain("within noise");
      expect(summaryMatch?.[1]).toContain("unstable");
      const detailsStart = output.indexOf("<details>");
      const detailsEnd = output.indexOf("</details>");
      const flatIdx = output.indexOf("flat/time", detailsStart);
      expect(flatIdx).toBeGreaterThan(detailsStart);
      expect(flatIdx).toBeLessThan(detailsEnd);
    });

    it("drops the highlights section when no candidate has anything to highlight", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: 0, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 0, n: 1, excluded: [] } }),
        ],
        metrics: {
          "flat/time": nWayMetric([
            { verdict: "no-signal", delta: 0.1, median: 100 },
            { verdict: "no-signal", delta: -0.2, median: 100 },
          ]),
        },
      });

      const output = renderMarkdown(result);

      expect(output).not.toContain("**`candidate-a`**");
      expect(output).not.toContain("**`candidate-b`**");
    });
  });

  describe("when the geomean has exclusions", () => {
    it("leaves the excluded metrics out of the geomean row", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -6 }),
        },
        candidates: [
          createCandidate({
            geomean: {
              value: -5.8,
              n: 3,
              excluded: [
                { metric: "jittery/time", reason: "unstable" },
                { metric: "broken/ratio", reason: "undefined-ratio" },
              ],
            },
          }),
        ],
      });

      const output = renderMarkdown(result);
      const geomeanRow = tableRows(output).find((r) => r.includes("geomean"));

      expect(geomeanRow).not.toContain("excluded");
    });
  });

  describe("when rendering the geomean row", () => {
    it("closes the table with the counted label and the delta alone", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        candidates: [createCandidate({ geomean: { value: -5.8, n: 4, excluded: [] } })],
      });

      const output = renderMarkdown(result);
      const rows = tableRows(output);
      const lastDataRow = rows[rows.length - 1]!;

      expect(lastDataRow.split("|").map((cell) => cell.trim())).toStrictEqual([
        "",
        "geomean (4 stable metrics)",
        "",
        "",
        "-5.8%",
        "",
      ]);
    });

    it("pairs each candidate's figure with its own count when several were compared", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: -10, n: 3, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 4, n: 1, excluded: [] } }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "improved", delta: -10, median: 90 },
            { verdict: "regressed", delta: 4, median: 104 },
          ]),
        },
      });

      const output = renderMarkdown(result);
      const geomeanRow = tableRows(output).find((r) => r.includes("geomean"));

      expect(geomeanRow?.split("|").map((cell) => cell.trim())).toStrictEqual([
        "",
        "geomean",
        "",
        "-10.0% · 3 stable metrics",
        "+4.0% · 1 stable metric",
        "",
      ]);
    });

    it("shows a dash when no stable metrics exist", () => {
      const result = createComparisonResult({
        metrics: { "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50 }) },
        candidates: [
          createCandidate({
            geomean: {
              value: Number.NaN,
              n: 0,
              excluded: [{ metric: "jittery/time", reason: "unstable" }],
            },
          }),
        ],
      });

      const output = renderMarkdown(result);
      const geomeanRow = tableRows(output).find((r) => r.includes("geomean"));

      expect(geomeanRow).toContain("—  no stable metrics");
    });
  });

  describe("when no ANSI codes are present", () => {
    it("emits no ANSI when mixed methods exercise all format helpers with color paths", () => {
      vi.stubEnv("NO_COLOR", "1");

      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.4, unit: "ns" }),
          "flat/time": bandMetric({ verdict: "no-signal", delta: 0.1 }),
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
          "heap/size": exactMetric({ delta: -7.9 }),
        },
        candidates: [
          createCandidate({
            geomean: {
              value: -5.8,
              n: 3,
              excluded: [{ metric: "jittery/time", reason: "unstable" }],
            },
          }),
        ],
      });

      const output = renderMarkdown(result);

      expect(output).not.toContain("\x1b[");
      expect(output).not.toMatch(/\x1b\[\d+m/);
    });
  });

  describe("when the ambient environment forces color", () => {
    it("still emits no ANSI escape sequences", () => {
      vi.stubEnv("FORCE_COLOR", "1");
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.4, unit: "ns" }),
          "flat/time": bandMetric({ verdict: "no-signal", delta: 0.1 }),
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
          "heap/size": exactMetric({ delta: -7.9 }),
        },
        candidates: [createCandidate({ geomean: { value: -5.8, n: 3, excluded: [] } })],
      });

      const output = renderMarkdown(result);

      expect(output).not.toMatch(/\x1b\[/);
    });
  });

  describe("when ordering highlights", () => {
    it("lists regressions first by |delta| desc, then improvements, then unstable", () => {
      const result = createComparisonResult({
        metrics: {
          "small-regression/time": signedRankMetric({ verdict: "regressed", delta: 2.2 }),
          "big-regression/time": signedRankMetric({ verdict: "regressed", delta: 15 }),
          "improvement/time": signedRankMetric({ verdict: "improved", delta: -10 }),
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });

      const output = renderMarkdown(result);
      const lines = output.split("\n");

      const bigRegIdx = lines.findIndex((l) => l.includes("big-regression/time"));
      const smallRegIdx = lines.findIndex((l) => l.includes("small-regression/time"));
      const impIdx = lines.findIndex((l) => l.includes("improvement/time"));
      const unstableIdx = lines.findIndex((l) => l.includes("jittery/time"));

      // All should be present
      expect(bigRegIdx).not.toBe(-1);
      expect(smallRegIdx).not.toBe(-1);
      expect(impIdx).not.toBe(-1);
      expect(unstableIdx).not.toBe(-1);

      // Order: regressions (big first) > improvements > unstable
      expect(bigRegIdx).toBeLessThan(smallRegIdx);
      expect(smallRegIdx).toBeLessThan(impIdx);
      expect(impIdx).toBeLessThan(unstableIdx);
    });
  });

  describe("when rendering GFM table syntax", () => {
    it("produces valid pipe-delimited rows with a header separator containing alignment", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
        },
        candidates: [createCandidate({ geomean: { value: -5, n: 1, excluded: [] } })],
      });

      const output = renderMarkdown(result);
      const rows = tableRows(output);

      // Every row starts and ends with |
      for (const row of rows) {
        expect(row.startsWith("|")).toBe(true);
        expect(row.endsWith("|")).toBe(true);
      }

      // Header separator has alignment markers
      const sep = headerSeparator(output)!;
      expect(sep).toBeDefined();
      // Should have right-alignment for numeric columns (at least some cells with --)
      expect(sep).toMatch(/--/);
    });

    it("has consistent column count across all rows", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "b/time": signedRankMetric({ verdict: "no-signal", delta: 0.2, unit: "ns" }),
        },
        candidates: [createCandidate({ geomean: { value: -5, n: 1, excluded: [] } })],
      });

      const output = renderMarkdown(result);
      // Include table rows from both main table and details block
      const allPipeRows = output.split("\n").filter((l) => l.startsWith("|"));
      const columnCounts = allPipeRows.map((row) => row.split("|").length);

      // All rows should have the same column count
      const expected = columnCounts[0];
      for (const count of columnCounts) {
        expect(count).toBe(expected);
      }
    });
  });

  describe("when a name collides with the table's own syntax", () => {
    /**
     * The cells of a GFM row, split on the pipes that delimit columns.
     *
     * A backslash-escaped pipe is cell content rather than a delimiter, so a row
     * that escapes properly keeps the header's cell count however many pipes its
     * text contains.
     */
    function gfmCells(row: string): string[] {
      return row.split(/(?<!\\)\|/).map((cell) => cell.trim());
    }

    /** A one-metric result whose metric name and candidate label are the test's own. */
    function namedResult(metricName: string, label: string): ComparisonResult {
      return createComparisonResult({
        candidates: [createCandidate({ label })],
        metrics: {
          [metricName]: signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
        },
      });
    }

    it("escapes a pipe in a metric name, so the row keeps the header's columns", () => {
      const rows = tableRows(renderMarkdown(namedResult("decode|encode/time", "candidate")));
      const dataRow = rows.find((row) => row.includes("decode"))!;

      expect.soft(gfmCells(dataRow)).toHaveLength(gfmCells(rows[0]!).length);
      expect(gfmCells(dataRow)[1]).toBe("decode\\|encode/time");
    });

    it("escapes a pipe inside the code span quoting a label", () => {
      const header = tableRows(renderMarkdown(namedResult("decode/time", "perf|lut")))[0]!;

      // Metric, baseline, candidate and delta, between a leading and a trailing pipe.
      expect.soft(gfmCells(header)).toHaveLength(6);
      expect(gfmCells(header)).toContain("`perf\\|lut`");
    });

    it("widens the code-span fence around a label carrying a backtick", () => {
      const header = tableRows(renderMarkdown(namedResult("decode/time", "perf/`lut`")))[0]!;

      // One backtick more than the longest run inside, padded so the label's own
      // trailing backtick cannot close the span.
      expect(header).toContain("`` perf/`lut` ``");
    });
  });

  describe("when rendering the method footer", () => {
    /** A run whose only metric fell back to the band for want of samples. */
    function bandOnlyResult(): ComparisonResult {
      return createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });
    }

    it("names the noise band only when verbose", () => {
      expect.soft(renderMarkdown(bandOnlyResult())).not.toContain("noise band");
      expect(renderMarkdown(bandOnlyResult(), { verbose: true })).toContain("noise band");
    });

    it.each([
      { mode: "plain", options: {} },
      { mode: "verbose", options: { verbose: true } },
    ])("hints at more samples in $mode mode", ({ options }) => {
      const output = renderMarkdown(bandOnlyResult(), options);

      expect(output).toContain("re-run with --samples 6 or more");
    });

    it("drops the hint when the signed-rank test carried the run", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -10 }) },
      });

      const output = renderMarkdown(result);

      expect(output).not.toContain("re-run with --samples");
    });

    it("gives each band fallback the phrasing its own cause earned", () => {
      const result = createComparisonResult({
        metrics: {
          "short/time": bandMetric({ verdict: "no-signal", delta: 1, n: 4 }),
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 3 }),
        },
      });

      const bandLines = renderMarkdown(result, { verbose: true })
        .split("\n")
        .filter((line) => line.startsWith("noise band"));

      expect(bandLines).toStrictEqual([
        "noise band ±(half-range × K) — n=4 below signed-rank floor (6 pairs)",
        "noise band ±(half-range × K) — ties left n=3 usable pairs (6 needed)",
      ]);
    });
  });
});
