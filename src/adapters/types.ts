/**
 * Turns one benchmark harness's stdout into the metric map the rest of gymrat works with.
 *
 * `parse` receives a bench script's full stdout and must throw {@link AdapterError}
 * when it yields no usable metric — returning an empty map instead would let a
 * silently broken bench script read as a run with nothing to compare.
 *
 * `defaults` is consulted once per metric name by `resolveMetricMeta`, and only
 * for fields the user's config does not override.
 */
export interface Adapter {
  readonly name: string;
  parse(stdout: string): Record<string, number>;
  defaults(metricName: string): MetricDefaults;
}

/**
 * What an adapter knows about a metric from its name alone.
 *
 * `unit` is omitted when the adapter cannot tell, in which case the report
 * prints the raw value rather than scaling it.
 */
export interface MetricDefaults {
  direction: "lower" | "higher";
  unit?: "ns" | "bytes";
}

/**
 * A bench script produced output the adapter could not read.
 *
 * The class itself is the signal: `formatCliError` matches on it to prefix the
 * message with the class name, which is what tells the user the fault is in
 * their bench script's output rather than in gymrat's git or config handling.
 * Adapters should raise this and nothing else for unparseable output.
 *
 * The explicit `setPrototypeOf` call keeps `instanceof` working when the class
 * is transpiled to a target that does not natively subclass `Error`.
 */
export class AdapterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdapterError";
    Object.setPrototypeOf(this, AdapterError.prototype);
  }
}
