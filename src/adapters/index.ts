export { AdapterError } from "./types.js";

import { GymratError } from "../errors.js";
import metricLinesAdapter from "./metric-lines.js";
import mitataAdapter from "./mitata.js";
import type { Adapter } from "./types.js";

const adapters: ReadonlyMap<string, Adapter> = new Map([
  ["metric-lines", metricLinesAdapter],
  ["mitata", mitataAdapter],
]);

export const adapterNames: readonly string[] = Array.from(adapters.keys()).toSorted();

/**
 * Look up a built-in adapter by the name used in config and `--adapter`.
 *
 * The registry is closed: adapters are not discovered or registered at runtime,
 * so an unknown name is a user typo rather than a missing plugin. The thrown
 * message lists the valid names sorted, since it doubles as the CLI's only
 * inventory of what is available.
 *
 * @throws when no adapter is registered under `name`.
 */
export function getAdapter(name: string): Adapter {
  const adapter = adapters.get(name);
  if (!adapter) {
    throw new GymratError(
      `Unknown adapter: "${name}".`,
      `valid adapters are: ${adapterNames.join(", ")}`,
    );
  }

  return adapter;
}
