import { GymratError } from "../errors.js";

/**
 * Where an adapter sends a complaint about output it could not read.
 *
 * The caller owns the destination so a warning can be interleaved with whatever
 * else is on the terminal — the CLI's progress line, for one — instead of
 * landing on stderr wherever the cursor happens to be.
 */
export type WarnSink = (message: string) => void;

/**
 * Turns one benchmark harness's stdout into the metric map the rest of gymrat works with.
 *
 * `parse` receives a bench script's full stdout and must throw {@link AdapterError}
 * when it yields no usable metric — returning an empty map instead would let a
 * silently broken bench script read as a run with nothing to compare. Complaints
 * about individual unreadable lines go to `warn`, which defaults to stderr so a
 * direct caller need not supply one.
 *
 * `defaults` is consulted once per metric name by `resolveMetricMeta`, and only
 * for fields the user's config does not override.
 */
export interface Adapter {
  readonly name: string;
  parse(stdout: string, warn?: WarnSink): Record<string, number>;
  defaults(metricName: string): MetricDefaults;
}

/**
 * What an adapter knows about a metric from its name alone.
 *
 * `unit` is omitted when the adapter cannot tell, in which case the report
 * prints the raw value rather than scaling it.
 *
 * `kind` groups metrics an adapter emits for the same benchmark under one label
 * (mitata emits both a `time` and a `memory` metric per benchmark), and
 * `shortName` is that benchmark's name with the kind suffix stripped. Both are
 * omitted when the adapter cannot tell, leaving the full metric name as the only
 * thing the report can show.
 */
export interface MetricDefaults {
  direction: "lower" | "higher";
  unit?: "ns" | "bytes";
  kind?: string;
  shortName?: string;
}

/**
 * A bench script produced output the adapter could not read.
 *
 * The class itself is the signal: `formatCliError` matches on it to prefix the
 * message with the class name, which is what tells the user the fault is in
 * their bench script's output rather than in gymrat's git or config handling.
 * Adapters should raise this and nothing else for unparseable output.
 */
export class AdapterError extends GymratError {}
