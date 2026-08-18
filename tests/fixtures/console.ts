import { vi } from "vitest";

/**
 * Run `fn` with `process.stderr.write` spied out and return everything it wrote.
 *
 * Callers must restore the spy from a suite-level `afterEach`
 * (`vi.restoreAllMocks()`), so an early return or throw inside `fn` cannot leak
 * the patched stderr into the next test.
 */
export function captureStderr(fn: () => void): string {
  const writeSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

  fn();

  return writeSpy.mock.calls.map((args) => String(args[0])).join("");
}
