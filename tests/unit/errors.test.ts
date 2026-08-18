import { describe, expect, it } from "vitest";

import { GymratError, messageOf, assertNever } from "../../src/errors.js";

describe("GymratError", () => {
  it("exposes the message, no hint, and its own name on construction", () => {
    const error = new GymratError("something broke");

    expect.soft(error).toBeInstanceOf(Error);
    expect.soft(error.message).toBe("something broke");
    expect.soft(error.hint).toBeUndefined();
    expect.soft(error.name).toBe("GymratError");
  });

  it("sets .hint when provided as second argument", () => {
    const error = new GymratError("something broke", "try restarting");

    expect(error.hint).toBe("try restarting");
  });

  describe("when subclassed", () => {
    it("sets .name to the subclass name automatically", () => {
      class CustomError extends GymratError {}

      const error = new CustomError("custom issue");

      expect(error.name).toBe("CustomError");
    });
  });
});

describe("messageOf", () => {
  it("returns .message for an Error instance", () => {
    const error = new Error("something failed");

    const result = messageOf(error);

    expect(result).toBe("something failed");
  });

  it.each([
    { description: "a string", value: "boom", expected: "boom" },
    { description: "a number", value: 42, expected: "42" },
  ])("returns String(value) for $description", ({ value, expected }) => {
    const result = messageOf(value);

    expect(result).toBe(expected);
  });
});

describe("assertNever", () => {
  it("throws an Error containing the JSON-stringified value", () => {
    // eslint-disable-next-line typescript/no-unsafe-type-assertion -- exercising the unreachable-value guard requires forcing `never`
    const value = "unexpected" as never;

    expect(() => assertNever(value)).toThrow('Unexpected value: "unexpected"');
  });
});
