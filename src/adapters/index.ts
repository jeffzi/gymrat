export { AdapterError } from "./types.js";
export type { Adapter, MetricDefaults } from "./types.js";

import metricLinesAdapter from "./metric-lines.js";
import mitataAdapter from "./mitata.js";
import type { Adapter } from "./types.js";

const adapters: ReadonlyMap<string, Adapter> = new Map([
  ["metric-lines", metricLinesAdapter],
  ["mitata", mitataAdapter],
]);

export function getAdapter(name: string): Adapter {
  const adapter = adapters.get(name);
  if (!adapter) {
    const validNames = Array.from(adapters.keys()).toSorted();
    throw new Error(`Unknown adapter: "${name}". Valid adapters: ${validNames.join(", ")}`);
  }

  return adapter;
}
