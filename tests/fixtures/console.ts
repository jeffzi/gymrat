import { vi } from "vitest";

/**
 * Run `fn` with `console.warn` spied out and return everything it warned.
 *
 * Callers must restore the spy from a suite-level `afterEach`
 * (`vi.restoreAllMocks()`), so an early return or throw inside `fn` cannot leak
 * the patched console into the next test.
 */
export function captureStderr(fn: () => void): string {
  const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

  fn();

  return warnSpy.mock.calls.map((args) => args.join(" ")).join("");
}
