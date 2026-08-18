import { afterEach, describe, expect, it, vi } from "vitest";

import {
  computeColumnWidth,
  formatLabel,
  formatTableLine,
  highlightInlineCode,
  shortenLabel,
  styleWithin,
  truncateLabels,
  withColor,
} from "../../src/report/format.js";

/** Stubs the environment so `styleText` emits ANSI codes. */
function forceColor(): void {
  vi.stubEnv("FORCE_COLOR", "1");
}

/** Stubs the environment so `styleText` never emits ANSI codes. */
function suppressColor(): void {
  vi.stubEnv("FORCE_COLOR", undefined);
  vi.stubEnv("NO_COLOR", "1");
}

describe("computeColumnWidth", () => {
  it.each([
    { driver: "the longest cell", header: 6, contents: [11, 24], min: 12, expected: 26 },
    { driver: "the header", header: 24, contents: [11, 9], min: 12, expected: 26 },
    // The gutter is added before the floor applies, so widest+2 wins once it clears the minimum.
    {
      driver: "the widest cell over the minimum",
      header: 10,
      contents: [11, 9],
      min: 12,
      expected: 13,
    },
    { driver: "the minimum, with no rows", header: 6, contents: [], min: 12, expected: 12 },
  ])("sizes the column from $driver", ({ header, contents, min, expected }) => {
    expect(computeColumnWidth(header, contents, min)).toBe(expected);
  });
});

describe("formatTableLine", () => {
  it.each([
    {
      desc: "pads every cell to its column width and separates them with a bar",
      cells: ["metric", "old", "new"],
      widths: [10, 6, 6],
      expected: "metric    │old   │new",
    },
    {
      desc: "pads interior cells that are empty so later columns stay aligned",
      cells: ["geomean", "", "", "-6.0%"],
      widths: [10, 6, 6, 8],
      expected: "geomean   │      │      │-6.0%",
    },
    {
      desc: "trims the padding after a trailing empty cell",
      cells: ["one-sided", "2.048µ ± 2%", ""],
      widths: [12, 14, 10],
      expected: "one-sided   │2.048µ ± 2%   │",
    },
    {
      desc: "treats missing cells as empty when widths outnumber cells",
      cells: ["metric"],
      widths: [10, 6],
      expected: "metric    │",
    },
  ])("$desc", ({ cells, widths, expected }) => {
    expect(formatTableLine(cells, widths)).toBe(expected);
  });
});

describe("shortenLabel", () => {
  const TEXT = "abcdefghijklmnop";

  describe("when the text already fits the width", () => {
    it.each([
      { desc: "is shorter than the width", maxWidth: 20 },
      { desc: "is exactly the width", maxWidth: TEXT.length },
    ])("returns it verbatim when the text $desc", ({ maxWidth }) => {
      expect(shortenLabel(TEXT, maxWidth)).toBe(TEXT);
    });
  });

  describe("when the text overflows the width", () => {
    it.each([
      // cspell:disable-next-line — synthetic test string, not a real word
      { desc: "splits the budget evenly on an odd width", maxWidth: 9, expected: "abcd…mnop" },
      {
        desc: "gives the extra character to the head on an even width",
        maxWidth: 8,
        expected: "abcd…nop",
      },
      {
        desc: "keeps a single leading character once the tail is squeezed out",
        maxWidth: 2,
        expected: "a…",
      },
      { desc: "collapses to the ellipsis alone at a width of one", maxWidth: 1, expected: "…" },
    ])("$desc", ({ maxWidth, expected }) => {
      expect(shortenLabel(TEXT, maxWidth)).toBe(expected);
    });
  });

  describe("when the width leaves no room at all", () => {
    it.each([
      { desc: "zero", maxWidth: 0 },
      { desc: "negative", maxWidth: -5 },
    ])("returns an empty string for a $desc width", ({ maxWidth }) => {
      expect(shortenLabel(TEXT, maxWidth)).toBe("");
    });
  });

  describe("when a cut boundary lands inside a multi-code-point grapheme", () => {
    // Each label places a cluster so that a UTF-16 cut would land inside it:
    // between the members of a ZWJ family, between an emoji and its skin-tone
    // modifier, or between the halves of a surrogate pair. A fragment of a
    // cluster renders as a different glyph — or as a replacement box — so the
    // cluster moves to the kept side whole.
    it.each([
      {
        cluster: "a ZWJ sequence",
        label: "ab👨‍👩‍👧‍👦cdefghij",
        maxWidth: 9,
        expected: "ab👨‍👩‍👧‍👦c…ghij",
      },
      {
        cluster: "an emoji with a skin-tone modifier",
        label: "ab👋🏽cdefghij",
        maxWidth: 9,
        expected: "ab👋🏽c…ghij",
      },
      {
        cluster: "a surrogate pair",
        label: "fix/🚀🚀-hot-path-🎉🎉x",
        maxWidth: 10,
        expected: "fix/🚀…-🎉🎉x",
      },
    ])("keeps $cluster whole", ({ label, maxWidth, expected }) => {
      expect(shortenLabel(label, maxWidth)).toBe(expected);
    });
  });

  describe("when the width counts graphemes rather than code units", () => {
    it("returns a label of clusters verbatim once it fits", () => {
      // Four clusters spanning 22 UTF-16 code units: budgeting by code units
      // would truncate a label the display has room for.
      const label = "👨‍👩‍👧‍👦👋🏽🚀🎉";

      expect(shortenLabel(label, 4)).toBe(label);
    });
  });
});

describe("truncateLabels", () => {
  describe("when every label fits the display width", () => {
    it("returns each one verbatim", () => {
      // "feature/short-branch" is exactly the 20-char display width.
      expect(truncateLabels(["main", "feature/short-branch"])).toStrictEqual([
        "main",
        "feature/short-branch",
      ]);
    });
  });

  describe("when a label overflows the display width", () => {
    it("joins its head and tail with a single ellipsis", () => {
      const truncated = truncateLabels(["feature/entity-spawn-fastpath"]);

      expect.soft(truncated).toStrictEqual(["feature/en…-fastpath"]);
      expect(truncated[0]).toHaveLength(20);
    });
  });

  describe("when two labels are identical once truncated", () => {
    it("extends the kept tail until the displayed labels differ", () => {
      // Both share the same 10-char head and 9-char tail, so the 20-char form
      // would name two different branches identically.
      const truncated = truncateLabels([
        "feature/experiment-one-fastpath",
        "feature/exploration-two-fastpath",
      ]);

      expect(truncated).toStrictEqual(["feature/ex…e-fastpath", "feature/ex…o-fastpath"]);
    });
  });

  describe("when disambiguation widens past a label that already fits the wider budget", () => {
    it("never lengthens that label", () => {
      // The colliding pair forces the budget out to 21 chars; the 21-char label
      // is already within it and must not gain an ellipsis.
      const shortEnough = "release/candidate-2.1";
      expect.soft(shortEnough).toHaveLength(21);

      const truncated = truncateLabels([
        "feature/experiment-one-fastpath",
        "feature/exploration-two-fastpath",
        shortEnough,
      ]);

      expect(truncated).toStrictEqual([
        "feature/ex…e-fastpath",
        "feature/ex…o-fastpath",
        shortEnough,
      ]);
    });
  });
});

describe("formatLabel", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("wraps the label in ANSI codes for the requested styles when color is forced", () => {
    forceColor();

    const styled = formatLabel("Hint:", ["yellow", "underline"]);

    // \x1b[33m = yellow, \x1b[4m = underline
    expect.soft(styled).toContain("\x1b[33m");
    expect.soft(styled).toContain("\x1b[4m");
    expect(styled).toContain("Hint:");
  });

  it("returns the bare label when color is suppressed", () => {
    suppressColor();
    expect(formatLabel("Hint:", ["yellow", "underline"])).toBe("Hint:");
  });
});

describe("styleWithin", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("when the marker occurs more than once", () => {
    it.each([
      {
        desc: "styles the first occurrence by default",
        options: undefined,
        expected: "\x1b[1mvs\x1b[22m vs",
      },
      {
        desc: "styles the first occurrence when last is false",
        options: { last: false },
        expected: "\x1b[1mvs\x1b[22m vs",
      },
      {
        desc: "styles the last occurrence when last is true",
        options: { last: true },
        expected: "vs \x1b[1mvs\x1b[22m",
      },
    ])("$desc", ({ options, expected }) => {
      forceColor();

      expect(styleWithin("vs vs", "vs", ["bold"], options)).toBe(expected);
    });
  });

  describe("when the marker carries a $ replacement pattern", () => {
    // `String.replace` expands these in its replacement argument, which would
    // splice the surrounding cell text in place of the marker.
    it.each([
      { desc: "the whole match", pattern: "$&" },
      { desc: "the text after the match", pattern: "$'" },
      { desc: "the text before the match", pattern: "$`" },
      { desc: "a capture group", pattern: "$1" },
      { desc: "an escaped dollar", pattern: "$$" },
    ])("styles $pattern, which means $desc, as the literal text it is", ({ pattern }) => {
      forceColor();

      expect(styleWithin(`cost ${pattern} up`, pattern, ["bold"])).toBe(
        `cost \x1b[1m${pattern}\x1b[22m up`,
      );
    });
  });

  it("returns the cell unchanged when the marker is absent", () => {
    forceColor();

    expect(styleWithin("vs main", "absent", ["bold"])).toBe("vs main");
  });
});

describe("withColor", () => {
  /** The two env vars that decide whether `styleText` emits ANSI codes. */
  interface ColorEnv {
    FORCE_COLOR: string | undefined;
    NO_COLOR: string | undefined;
  }

  function colorEnv(): ColorEnv {
    return { FORCE_COLOR: process.env.FORCE_COLOR, NO_COLOR: process.env.NO_COLOR };
  }

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("while the callback runs", () => {
    it.each([
      {
        mode: "undefined",
        color: undefined,
        expected: { FORCE_COLOR: "1", NO_COLOR: "0" },
      },
      {
        mode: "false",
        color: false,
        expected: { FORCE_COLOR: undefined, NO_COLOR: "1" },
      },
      {
        mode: "true",
        color: true,
        expected: { FORCE_COLOR: "1", NO_COLOR: undefined },
      },
    ])("hands the callback the environment for color=$mode", ({ color, expected }) => {
      vi.stubEnv("FORCE_COLOR", "1");
      vi.stubEnv("NO_COLOR", "0");

      expect(withColor(color, colorEnv)).toStrictEqual(expected);
    });
  });

  describe("once the callback is done", () => {
    it.each([
      { desc: "puts back the values it found", prior: { FORCE_COLOR: "1", NO_COLOR: "0" } },
      {
        desc: "leaves absent vars absent",
        prior: { FORCE_COLOR: undefined, NO_COLOR: undefined },
      },
    ])("$desc", ({ prior }) => {
      vi.stubEnv("FORCE_COLOR", prior.FORCE_COLOR);
      vi.stubEnv("NO_COLOR", prior.NO_COLOR);

      withColor(false, () => undefined);

      expect(colorEnv()).toStrictEqual(prior);
    });

    it("restores the environment even when the callback throws", () => {
      vi.stubEnv("FORCE_COLOR", "1");
      vi.stubEnv("NO_COLOR", "0");

      expect
        .soft(() =>
          withColor(true, () => {
            throw new Error("render failed");
          }),
        )
        .toThrow("render failed");

      expect(colorEnv()).toStrictEqual({ FORCE_COLOR: "1", NO_COLOR: "0" });
    });
  });
});

describe("highlightInlineCode", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("when the text contains backtick-wrapped spans", () => {
    it("strips the backticks and styles the content yellow when color is forced", () => {
      forceColor();

      const result = highlightInlineCode("Run `gymrat doctor` to verify.");

      // \x1b[33m = yellow
      expect.soft(result).toContain("\x1b[33m");
      expect.soft(result).toContain("gymrat doctor");
      expect(result).not.toContain("`");
    });

    it("strips the backticks and returns the bare content when color is suppressed", () => {
      suppressColor();

      const result = highlightInlineCode("Run `gymrat doctor` to verify.");

      expect.soft(result).toBe("Run gymrat doctor to verify.");
      expect(result).not.toContain("\x1b[");
    });
  });

  describe("when the text contains multiple backtick spans", () => {
    it("styles each span independently", () => {
      forceColor();

      const result = highlightInlineCode("Use `gymrat compare` or `gymrat measure`.");

      expect.soft(result).not.toContain("`");
      expect.soft(result).toContain("gymrat compare");
      expect(result).toContain("gymrat measure");
    });

    it("strips backticks from all spans when color is suppressed", () => {
      suppressColor();

      expect(highlightInlineCode("Use `gymrat compare` or `gymrat measure`.")).toBe(
        "Use gymrat compare or gymrat measure.",
      );
    });
  });

  describe("when the text has no backtick pairs", () => {
    it("returns the text unchanged", () => {
      forceColor();

      expect(highlightInlineCode("No inline code here.")).toBe("No inline code here.");
    });
  });

  it.each([
    { desc: "a command with spaces", text: "`gymrat doctor`", expected: "gymrat doctor" },
    { desc: "a flag", text: "`--bench`", expected: "--bench" },
    { desc: "a path", text: "`gymrat.json`", expected: "gymrat.json" },
    { desc: "a single word", text: "`runbook`", expected: "runbook" },
  ])("handles $desc", ({ text, expected }) => {
    suppressColor();

    expect(highlightInlineCode(text)).toBe(expected);
  });

  it("targets the stream's color detection when a stream is provided", () => {
    forceColor();

    const result = highlightInlineCode("Run `gymrat doctor`.", process.stderr);

    expect.soft(result).toContain("\x1b[33m");
    expect(result).not.toContain("`");
  });
});
