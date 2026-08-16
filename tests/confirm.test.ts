import { Readable } from "node:stream";

import { describe, expect, it } from "vitest";

import { confirmAction } from "../src/confirm.js";

/**
 * Build a readable stream that yields `input` then ends.
 *
 * When `input` is `undefined` the stream ends immediately (EOF with no data),
 * modelling a closed pipe or a non-interactive stdin.
 */
function inputStream(input?: string): Readable {
  return input === undefined ? Readable.from([]) : Readable.from([input]);
}

describe("confirmAction", () => {
  it.each([
    { input: "y\n", desc: "lowercase y", expected: true },
    { input: "Y\n", desc: "uppercase Y", expected: true },
    { input: "n\n", desc: "lowercase n", expected: false },
    { input: "N\n", desc: "uppercase N", expected: false },
    { input: "\n", desc: "empty line (Enter)", expected: false },
    { input: "yes\n", desc: "full word 'yes'", expected: false },
    { input: "nope\n", desc: "random text", expected: false },
  ])("returns $expected for $desc", async ({ input, expected }) => {
    // Act
    const result = await confirmAction("Proceed?", inputStream(input));

    // Assert
    expect(result).toBe(expected);
  });

  it("returns false on EOF (empty stream)", async () => {
    // Act
    const result = await confirmAction("Proceed?", inputStream());

    // Assert
    expect(result).toBe(false);
  });
});
