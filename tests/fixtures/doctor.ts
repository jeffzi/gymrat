import type { Check } from "../../src/doctor/checks.js";

/** Build a doctor check with sensible defaults, overridable per field. */
export function createCheck(overrides: Partial<Check> = {}): Check {
  return {
    name: "test-check",
    status: "ok",
    detail: "all good",
    ...overrides,
  };
}
