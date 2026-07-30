import { describe, expect, it } from "vitest";

import { GymratError, messageOf, assertNever } from "../../src/errors.js";

describe("GymratError", () => {
  it("constructs a base error with message, no hint, and GymratError name", () => {
    // Arrange
    const error = new GymratError("something broke");

    // Assert
    expect.soft(error).toBeInstanceOf(Error);
    expect.soft(error.message).toBe("something broke");
    expect.soft(error.hint).toBeUndefined();
    expect.soft(error.name).toBe("GymratError");
  });

  it("sets .hint when provided as second argument", () => {
    // Arrange & Act
    const error = new GymratError("something broke", "try restarting");

    // Assert
    expect(error.hint).toBe("try restarting");
  });

  describe("when subclassed", () => {
    it("sets .name to the subclass name automatically", () => {
      // Arrange
      class CustomError extends GymratError {}

      // Act
      const error = new CustomError("custom issue");

      // Assert
      expect(error.name).toBe("CustomError");
    });
  });
});

describe("messageOf", () => {
  it("returns .message for an Error instance", () => {
    // Arrange
    const error = new Error("something failed");

    // Act
    const result = messageOf(error);

    // Assert
    expect(result).toBe("something failed");
  });

  it.each([
    { description: "a string", value: "boom", expected: "boom" },
    { description: "a number", value: 42, expected: "42" },
  ])("returns String(value) for $description", ({ value, expected }) => {
    // Act
    const result = messageOf(value);

    // Assert
    expect(result).toBe(expected);
  });
});

describe("assertNever", () => {
  it("throws an Error containing the JSON-stringified value", () => {
    // Arrange
    // eslint-disable-next-line typescript/no-unsafe-type-assertion -- exercising the unreachable-value guard requires forcing `never`
    const value = "unexpected" as never;

    // Act & Assert
    expect(() => assertNever(value)).toThrow('Unexpected value: "unexpected"');
  });
});
