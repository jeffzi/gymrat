import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult } from "../../src/report/types.js";
import type { KindAggregate } from "../../src/verdict/aggregate.js";
import {
  createCandidate,
  createComparisonResult,
  geomeanOf,
  groupedComparison,
  memoryKind,
  multiCandidateResult,
  nWayKindMetric,
  nWayMetric,
  otherKind,
  signedRankMetric,
  timeKind,
  twoKindMetrics,
  twoKindResult,
} from "../fixtures/comparison-result.js";

function withoutGatedGeomean(kind: KindAggregate): KindAggregate {
  const { gatedGeomean: _, ...rest } = kind;
  return rest;
}

/** Character offsets of every occurrence of `glyph` in a rendered table line. */
function offsetsOf(line: string, glyph: string): number[] {
  const offsets: number[] = [];
  for (let i = line.indexOf(glyph); i !== -1; i = line.indexOf(glyph, i + 1)) {
    offsets.push(i);
  }
  return offsets;
}

/**
 * Character offsets of every column separator in a rendered table line.
 *
 * Two lines whose separators sit at the same offsets have aligned columns.
 */
function separatorOffsets(line: string): number[] {
  return offsetsOf(line, "│");
}

/** The cells of a rendered table line, padding included. */
function cellsOf(line: string): string[] {
  return line.split("│");
}

/** The single rendered line starting with `prefix`, or a failure naming the report. */
function lineStartingWith(report: string, prefix: string): string {
  const line = report.split("\n").find((candidate) => candidate.startsWith(prefix));
  if (line === undefined) {
    throw new Error(`no line starting with ${prefix} in report:\n${report}`);
  }
  return line;
}

/**
 * The first rendered line containing `needle`, or a failure naming the report.
 *
 * A colored line starts with escape codes rather than its text, so the color
 * tests match on content instead of a prefix.
 */
function lineContaining(report: string, needle: string): string {
  const line = report.split("\n").find((candidate) => candidate.includes(needle));
  if (line === undefined) {
    throw new Error(`no line containing ${needle} in report:\n${report}`);
  }
  return line;
}

/**
 * The part of `cell` ahead of its verdict `glyph` — the value and its padding.
 *
 * A candidate column packs a value and a verdict into one cell, so this is the
 * half that carries no verdict of its own and must stay unstyled. The escape
 * sequences opening the verdict's own style sit right in front of the glyph, so
 * they are trimmed off the tail rather than counted against the value.
 */
function valuePartOf(cell: string, glyph: string): string {
  const index = cell.indexOf(glyph);
  if (index === -1) {
    throw new Error(`no ${glyph} in cell: ${JSON.stringify(cell)}`);
  }
  return cell.slice(0, index).replace(/(?:\x1b\[\d+m)*$/, "");
}

/** Matches a line dimmed end-to-end: opens with SGR 2, closes with SGR 22. */
const DIMMED_LINE = /^\x1b\[2m.*\x1b\[22m$/;

/**
 * The SGR parameters opened immediately before `marker` in `line`.
 *
 * Only the unbroken run of escape sequences touching the marker counts, so a
 * style opened at the start of the line does not leak into the result.
 *
 * Pass `last` to read the trailing occurrence instead of the leading one, for
 * a marker that repeats within the line.
 */
function stylesAt(line: string, marker: string, options: { last?: boolean } = {}): string[] {
  const index = options.last === true ? line.lastIndexOf(marker) : line.indexOf(marker);
  if (index === -1) {
    throw new Error(`no ${marker} in line: ${JSON.stringify(line)}`);
  }
  const opened = /((?:\x1b\[\d+m)*)$/.exec(line.slice(0, index))?.[1] ?? "";
  return [...opened.matchAll(/\x1b\[(\d+)m/g)].map((match) => match[1] ?? "");
}

/** For each SGR parameter that closes a style, the parameters it closes. */
const SGR_CLOSERS: Readonly<Record<string, RegExp>> = {
  "0": /^\d+$/,
  "22": /^[12]$/,
  "23": /^3$/,
  "24": /^4$/,
  "39": /^(?:3[0-7]|9[0-7])$/,
  "49": /^(?:4[0-7]|10[0-7])$/,
};

/**
 * The SGR parameters still open at each column separator of `line`.
 *
 * A separator that inherits its row's style reports that style here; one left
 * in the terminal's default color reports nothing.
 */
function separatorStyles(line: string): string[][] {
  let open: string[] = [];
  const styles: string[][] = [];
  for (const token of line.matchAll(/\x1b\[(\d+)m|│/g)) {
    const parameter = token[1];
    if (parameter === undefined) {
      styles.push(open);
      continue;
    }
    const closes = SGR_CLOSERS[parameter];
    open = closes === undefined ? [...open, parameter] : open.filter((p) => !closes.test(p));
  }
  return styles;
}

/** Every rendered table row of a report, styling stripped, in report order. */
function tableRows(report: string): string[] {
  return report
    .split("\n")
    .map((line) => stripAnsi(line))
    .filter((line) => line.includes("│"));
}

/** The last rendered table row of a report — the row the table closes on. */
function lastTableRow(report: string): string {
  const row = tableRows(report).at(-1);
  if (row === undefined) {
    throw new Error(`no table rows in report:\n${report}`);
  }
  return row;
}

/**
 * One entry per report line, coarse enough to read as a layout.
 *
 * A table row collapses to its first cell, a column rule collapses to a marker,
 * and every other line stays as its plain text. A section's top border joins its
 * columns with top-T junctions rather than the crossings of a rule, so it gets
 * its own marker.
 */
function tableShape(report: string): string[] {
  return report.split("\n").map((line) => {
    const bare = stripAnsi(line);
    if (/^─+┼/.test(bare)) {
      return "<rule>";
    }
    if (/^[─┬]+$/.test(bare)) {
      return "<border>";
    }
    if (!bare.includes("│")) {
      return bare.trimEnd();
    }
    return cellsOf(bare)[0]?.trim() ?? "";
  });
}

/** The table region of a report: everything down to the last table row. */
function tableRegion(report: string): string[] {
  const shape = tableShape(report);
  const lines = report.split("\n");
  const last = lines.reduce(
    (found, line, index) => (stripAnsi(line).includes("│") ? index : found),
    -1,
  );
  if (last === -1) {
    throw new Error(`no table rows in report:\n${report}`);
  }
  return shape.slice(0, last + 1);
}

/** The lines of the `highlights` block, its heading excluded. */
function highlightLines(report: string): string[] {
  const lines = report.split("\n");
  const start = lines.findIndex((line) => stripAnsi(line) === "highlights");
  if (start === -1) {
    return [];
  }
  const rest = lines.slice(start + 1);
  const end = rest.indexOf("");
  return end === -1 ? rest : rest.slice(0, end);
}

beforeEach(() => {
  vi.stubEnv("NO_COLOR", "1");
  vi.stubEnv("FORCE_COLOR", undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("renderReport", () => {
  describe("when ordering the report sections", () => {
    /** A two-metric run whose only footer content is the signed-rank method line. */
    function orderedResult(): ComparisonResult {
      return createComparisonResult({
        baselineLabel: "main",
        metrics: {
          "metric1/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "metric2/time": signedRankMetric({
            verdict: "no-signal",
            delta: 2,
            gating: false,
            unit: "ns",
          }),
        },
        candidates: [
          createCandidate({
            label: "faster",
            kinds: [otherKind(-5, 1)],
          }),
        ],
      });
    }

    it("emits table, summary and highlights in that order, and closes there", () => {
      const lines = renderReport(orderedResult()).split("\n");

      // Each section's content is asserted by its own test; this pins the order.
      expect.soft(lines[0]).toContain("gymrat compare · baseline main ↔ faster");
      expect.soft(lines[1]).toMatch(/^metric\s+│/);
      expect.soft(lines[2]).toMatch(/^─+┼/);
      expect.soft(lines[3]).toContain("metric1/time");
      expect.soft(lines[4]).toContain("metric2/time");
      expect.soft(lines[5]).toMatch(/^─+┼/);
      expect.soft(lines[6]).toContain("geomean");
      expect.soft(lines[7]).toBe("");
      expect.soft(lines[8]).toContain("✓ 1 improved");
      expect.soft(lines[9]).toBe("");
      expect.soft(lines[10]).toBe("highlights");
      expect.soft(lines[11]).toContain("metric1/time");
      expect(lines).toHaveLength(12);
    });

    it("adds the method block below a blank line when verbose", () => {
      const lines = renderReport(orderedResult(), { verbose: true }).split("\n");

      expect.soft(lines[11]).toContain("metric1/time");
      expect.soft(lines[12]).toBe("");
      expect.soft(lines[13]).toContain("Wilcoxon signed-rank");
      expect(lines).toHaveLength(14);
    });
  });

  describe("when rendering more than one candidate", () => {
    it("heads one column per candidate with that candidate's name alone", () => {
      const headerLine = lineStartingWith(renderReport(multiCandidateResult()), "metric");

      expect(cellsOf(headerLine).map((cell) => cell.trim())).toStrictEqual([
        "metric",
        "main",
        "candidate-a",
        "candidate-b",
        "candidate-c",
      ]);
    });

    it("keeps the baseline figure and pairs each candidate's own figure with its verdict", () => {
      const row = lineStartingWith(renderReport(multiCandidateResult()), "decode/time");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
        "150ns ± 3%  ≈  unstable",
      ]);
    });

    it("carries one geomean figure per candidate column, each with its own count", () => {
      const row = lineStartingWith(renderReport(multiCandidateResult()), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean",
        "",
        "-10.0% · 1 stable metric",
        "+4.0% · 1 stable metric",
        "0.0% · 1 stable metric",
      ]);
    });

    it("closes the table on the geomean row", () => {
      const row = lastTableRow(renderReport(multiCandidateResult()));

      expect(cellsOf(row)[0]?.trim()).toBe("geomean");
    });

    it("lines the column separators up across header, metric rows and geomean", () => {
      const report = renderReport(multiCandidateResult());
      const headerOffsets = separatorOffsets(lineStartingWith(report, "metric"));

      expect
        .soft(separatorOffsets(lineStartingWith(report, "decode/time")))
        .toStrictEqual(headerOffsets);
      expect(separatorOffsets(lineStartingWith(report, "geomean"))).toStrictEqual(headerOffsets);
    });

    it("summarizes each candidate on its own line, behind that candidate's label", () => {
      const summaries = renderReport(multiCandidateResult())
        .split("\n")
        .filter((line) => /✓ \d+ improved/.test(line));

      expect(summaries).toStrictEqual([
        "candidate-a  ✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive",
        "candidate-b  ✓ 0 improved   ✗ 1 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive",
        "candidate-c  ✓ 0 improved   ✗ 0 regressed   ≈ 1 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive",
      ]);
    });

    it("groups the highlights into one subsection per candidate", () => {
      const highlights = highlightLines(renderReport(multiCandidateResult()));

      expect(highlights).toStrictEqual([
        "  candidate-a",
        "    ✓ decode/time  -10.0%",
        "  candidate-b",
        "    ✗ decode/time   +4.0%",
        "  candidate-c",
        "    ≈ decode/time  unstable  noise ±30.0%",
        "  unstable metrics won't stabilize with more samples",
      ]);
    });

    it("drops the highlights section when no candidate has anything to highlight", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "no-signal", delta: 0.4, median: 100 },
            { verdict: "no-signal", delta: -0.3, median: 100 },
          ]),
        },
      });

      const report = renderReport(result);

      expect.soft(report).not.toContain("highlights");
      expect(report).toContain("candidate-a  ✓ 0 improved");
    });

    /**
     * A run whose two rows differ in whether any candidate moved at all.
     *
     * `mixed/time` moved for one candidate and stayed flat for the other, which
     * is exactly the row a per-candidate dimming rule would wrongly recede.
     */
    function dimmingResult(): ComparisonResult {
      return createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
        metrics: {
          "flat/time": nWayMetric([
            { verdict: "no-signal", delta: 0.3, median: 100 },
            { verdict: "unstable", delta: -50, median: 50 },
          ]),
          "mixed/time": nWayMetric([
            { verdict: "no-signal", delta: 0.3, median: 100 },
            { verdict: "improved", delta: -17.5, median: 83 },
          ]),
        },
      });
    }

    describe("when color styling is applied", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([{ row: "flat/time" }, { row: "mixed/time" }])(
        "leaves the $row row without an end-to-end dim, quiet cells and all",
        ({ row }) => {
          expect(lineContaining(renderReport(dimmingResult()), row)).not.toMatch(DIMMED_LINE);
        },
      );

      it("styles each cell's verdict by that cell's own verdict on an all-quiet row", () => {
        const row = lineContaining(renderReport(dimmingResult()), "flat/time");

        expect.soft(stylesAt(row, "~")).toContain("2");
        expect(stylesAt(row, "≈")).toContain("33");
      });

      it("leaves the name and the values on an all-quiet row unstyled", () => {
        const cells = cellsOf(lineContaining(renderReport(dimmingResult()), "flat/time"));

        expect.soft(cells.slice(0, 2).join("│")).not.toContain("\x1b[");
        expect.soft(valuePartOf(cells[2] ?? "", "~")).not.toContain("\x1b[");
        expect(valuePartOf(cells[3] ?? "", "≈")).not.toContain("\x1b[");
      });

      it.each([
        { verdict: "improved", column: 2, glyph: "✓" },
        { verdict: "regressed", column: 3, glyph: "✗" },
        { verdict: "unstable", column: 4, glyph: "≈" },
      ])(
        "leaves the $verdict cell's value plain beside its neighbors' verdicts",
        ({ column, glyph }) => {
          const row = lineContaining(renderReport(multiCandidateResult()), "decode/time");

          expect(valuePartOf(cellsOf(row)[column] ?? "", glyph)).not.toContain("\x1b[");
        },
      );

      it("paints an unstable N-way cell's verdict amber, as an unstable row is painted", () => {
        const row = lineContaining(renderReport(multiCandidateResult()), "decode/time");

        expect.soft(stylesAt(row, "≈")).toContain("33");
        expect(stylesAt(row, "unstable")).toContain("33");
      });

      it("pads on the plain text, so the colored columns line up once the styles are stripped", () => {
        const bare = stripAnsi(renderReport(multiCandidateResult()));
        const headerOffsets = separatorOffsets(lineStartingWith(bare, "metric"));

        expect
          .soft(separatorOffsets(lineStartingWith(bare, "decode/time")))
          .toStrictEqual(headerOffsets);
        expect(separatorOffsets(lineStartingWith(bare, "geomean"))).toStrictEqual(headerOffsets);
      });

      it("emboldens the candidate label in N-way summary lines when color is on", () => {
        const report = renderReport(multiCandidateResult());
        const summaries = report.split("\n").filter((line) => /✓ \d+ improved/.test(line));

        expect(summaries).toHaveLength(3);
        const [sa, sb, sc] = summaries;
        if (sa === undefined || sb === undefined || sc === undefined)
          throw new Error("missing summary");
        expect.soft(stylesAt(sa, "candidate-a")).toContain("1");
        expect.soft(stylesAt(sb, "candidate-b")).toContain("1");
        expect(stylesAt(sc, "candidate-c")).toContain("1");
      });

      it("emboldens the candidate sublabels in N-way highlights when color is on", () => {
        const highlights = highlightLines(renderReport(multiCandidateResult()));
        const sublabels = highlights.filter((line) => {
          const stripped = stripAnsi(line).trim();
          return ["candidate-a", "candidate-b", "candidate-c"].includes(stripped);
        });

        expect(sublabels).toHaveLength(3);
        for (const sublabel of sublabels) {
          expect(sublabel).toContain("\x1b[1m");
        }
      });

      it("colors glyph and delta together in N-way improved and regressed cells", () => {
        const report = renderReport(multiCandidateResult());
        const row = lineContaining(report, "decode/time");

        expect.soft(stylesAt(row, "✓")).toContain("32");
        expect.soft(stylesAt(row, "-10.0%")).toContain("32");
        expect.soft(stylesAt(row, "✗")).toContain("31");
        expect(stylesAt(row, "+4.0%")).toContain("31");
      });

      it("dims the quiet candidate segment on a bright N-way row", () => {
        const report = renderReport(dimmingResult());
        const row = lineContaining(report, "mixed/time");

        expect.soft(stylesAt(row, "~")).toContain("2");
        expect(stylesAt(row, "+0.3%")).toContain("2");
      });

      it("dims the provenance in N-way geomean cells", () => {
        const report = renderReport(multiCandidateResult());
        const geomean = lineContaining(report, "geomean");

        expect(stylesAt(geomean, "1 stable metric")).toContain("2");
      });
    });
  });

  describe("when the run spans more than one metric kind", () => {
    it("gives each kind its own titled section, closed by that kind's geomean", () => {
      expect(tableRegion(renderReport(twoKindResult()))).toStrictEqual([
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "",
        "<border>",
        "time",
        "<rule>",
        "entity",
        "alive_check",
        "spawn",
        "geomean · entity (2)",
        "",
        "warmup",
        "<rule>",
        "geomean · time (3)",
        "",
        "informational — gating off (config: kinds.memory.gating = false)",
        "<border>",
        "memory",
        "<rule>",
        "encode",
        "<rule>",
        "geomean · memory (1)",
      ]);
    });

    it("spans a section's top border across the full table width", () => {
      const report = stripAnsi(renderReport(twoKindResult()));
      const lines = report.split("\n");
      const header = lineStartingWith(report, "time");
      const headerIndex = lines.indexOf(header);
      const border = lines[headerIndex - 1];
      const rule = lines[headerIndex + 1];
      if (border === undefined || rule === undefined) {
        throw new Error(`no border or rule around section header in report:\n${report}`);
      }

      expect.soft(border).not.toContain("┼");
      expect(border).toHaveLength(rule.length);
    });

    it("joins a section's top border to the header's columns", () => {
      const report = stripAnsi(renderReport(twoKindResult()));
      const lines = report.split("\n");
      const header = lineStartingWith(report, "time");
      const headerIndex = lines.indexOf(header);
      const border = lines[headerIndex - 1];
      if (border === undefined) {
        throw new Error(`no border above section header in report:\n${report}`);
      }

      expect(offsetsOf(border, "┬")).toStrictEqual(separatorOffsets(header));
    });

    it("lines every section's columns up with the first section's header", () => {
      const report = renderReport(twoKindResult());
      const bare = stripAnsi(report);
      const timeHeader = lineStartingWith(bare, "time");
      const memoryHeader = lineStartingWith(bare, "memory");
      const offsets = separatorOffsets(timeHeader);

      expect.soft(separatorOffsets(memoryHeader)).toStrictEqual(offsets);
      expect
        .soft(separatorOffsets(lineStartingWith(report, "  alive_check")))
        .toStrictEqual(offsets);
      expect.soft(separatorOffsets(lineStartingWith(report, "entity "))).toStrictEqual(offsets);
      expect(separatorOffsets(lineStartingWith(report, "geomean · memory"))).toStrictEqual(offsets);
    });

    it.each([
      {
        source: "the kind-level config entry",
        makeResult: () => twoKindResult({ configKinds: { memory: { gating: false } } }),
        expected: "informational — gating off (config: kinds.memory.gating = false)",
      },
      {
        source: "per-metric overrides alone",
        makeResult: (): ComparisonResult => {
          const { configKinds: _, ...rest } = twoKindResult();
          return rest;
        },
        expected: "informational — gating off",
      },
    ])("credits $source for a non-gating kind's informational tag", ({ makeResult, expected }) => {
      const report = renderReport(makeResult());

      expect(lineContaining(report, "informational")).toBe(expected);
    });

    it.each([
      { placement: "indented under its group, stripped of the group prefix", row: "  alive_check" },
      { placement: "at the margin under its bare short name", row: "warmup" },
    ])("names a metric row $placement", ({ row }) => {
      const line = lineStartingWith(renderReport(twoKindResult()), row);

      expect(cellsOf(line)[0]?.trimEnd()).toBe(row);
    });

    it("names each highlight by kind and short metric, padded to keep the deltas aligned", () => {
      const highlights = highlightLines(renderReport(twoKindResult())).map((line) => line.trim());

      expect(highlights).toStrictEqual([
        "✗ time · entity.spawn         +4.0%",
        "✓ time · entity.alive_check  -10.0%",
        "✓ memory · encode             -7.0%",
      ]);
    });

    it("prefixes the kind inside every candidate's highlight subsection", () => {
      const result = createComparisonResult({
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
            kinds: [timeKind({ geomean: geomeanOf(-10, 1), groups: [] }), memoryKind()],
          }),
          createCandidate({
            label: "candidate-b",
            kinds: [
              timeKind({ geomean: geomeanOf(4, 1), groups: [] }),
              memoryKind({ geomean: geomeanOf(-2, 1) }),
            ],
          }),
        ],
        configKinds: { memory: { gating: false } },
      });

      expect(highlightLines(renderReport(result))).toStrictEqual([
        "  candidate-a",
        "    ✓ time · entity.alive_check  -10.0%",
        "    ✓ memory · encode             -7.0%",
        "  candidate-b",
        "    ✗ time · entity.alive_check   +4.0%",
        "    ✓ memory · encode             -2.0%",
      ]);
    });

    it("counts the excluded metrics into a geomean label's provenance", () => {
      const result = twoKindResult({
        candidates: [
          createCandidate({
            kinds: [
              timeKind({
                geomean: geomeanOf(-3.2, 2, {
                  excluded: [{ metric: "warmup/time", reason: "unstable" }],
                }),
              }),
              memoryKind(),
            ],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean · time");

      expect(cellsOf(row)[0]?.trim()).toBe("geomean · time (2/3)");
    });

    it.each([
      {
        gating: "one kind gates",
        makeResult: (): ComparisonResult => twoKindResult(),
      },
      {
        gating: "several kinds gate",
        makeResult: (): ComparisonResult =>
          createComparisonResult({
            metrics: twoKindMetrics({ memoryGates: true }),
            candidates: [
              createCandidate({
                kinds: [timeKind(), memoryKind({ gatedGeomean: geomeanOf(6.1, 1) })],
              }),
            ],
          }),
      },
      {
        gating: "no kind gates",
        makeResult: (): ComparisonResult =>
          twoKindResult({
            metrics: twoKindMetrics({ timeGates: false }),
            candidates: [
              createCandidate({
                kinds: [withoutGatedGeomean(timeKind()), memoryKind()],
              }),
            ],
          }),
      },
    ])(
      "closes the table on the last kind's geomean, with no gated row, when $gating",
      ({ makeResult }) => {
        const report = renderReport(makeResult());

        expect.soft(tableRegion(report).at(-1)).toBe("geomean · memory (1)");
        expect(report).not.toContain("geomean · gated");
      },
    );

    it("carries one figure per candidate column on every aggregate row", () => {
      const report = renderReport(groupedComparison());
      const cellsAt = (label: string): string[] =>
        cellsOf(lineStartingWith(report, label)).map((cell) => cell.trim());

      expect
        .soft(cellsAt("geomean · entity"))
        .toStrictEqual([
          "geomean · entity",
          "",
          "-10.0% · 1 stable metric",
          "+4.0% · 1 stable metric",
        ]);
      expect
        .soft(cellsAt("geomean · time"))
        .toStrictEqual([
          "geomean · time",
          "",
          "-10.0% · 1 stable metric",
          "+4.0% · 1 stable metric",
        ]);
      expect(tableRegion(report).at(-1)).toBe("geomean · memory");
    });

    describe("when color styling is applied", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([
        { row: "sub-geomean", label: "geomean · entity", value: "-3.1%" },
        { row: "kind geomean", label: "geomean · time", value: "-3.2%" },
      ])("paints an improving $row value green once it clears its band", ({ label, value }) => {
        const line = lineContaining(renderReport(twoKindResult()), label);

        expect(stylesAt(line, value)).toStrictEqual(["1", "32"]);
      });

      it("emboldens the kind name in the section header and dims the informational tag", () => {
        const report = renderReport(twoKindResult());
        const header = report
          .split("\n")
          .find((line) => line.includes("│") && stripAnsi(line).trimStart().startsWith("memory"));
        if (header === undefined) {
          throw new Error("no memory header in report");
        }
        const tag = lineContaining(report, "informational");

        expect.soft(stylesAt(header, "memory")).toStrictEqual(["1"]);
        expect(stylesAt(tag, "informational")).toStrictEqual(["2"]);
      });

      it("leaves every column separator in the default color, whatever style its row carries", () => {
        const rows = renderReport(twoKindResult())
          .split("\n")
          .filter((line) => line.includes("│"));

        const inherited = rows.filter((row) =>
          separatorStyles(row).some((styles) => styles.length > 0),
        );

        expect(inherited).toStrictEqual([]);
      });
    });
  });
});
