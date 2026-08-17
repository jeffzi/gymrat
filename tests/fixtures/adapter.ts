import type { Adapter, WarnSink } from "../../src/adapters/types.js";

/** Build a mock adapter with sensible defaults, overridable per field. */
export function createMockAdapter(overrides: Partial<Adapter> = {}): Adapter {
  return {
    name: "metric-lines",
    parse(_stdout: string, _warn?: WarnSink): Record<string, number> {
      return {};
    },
    defaults(_metricName: string) {
      return { direction: "lower" as const };
    },
    ...overrides,
  };
}
