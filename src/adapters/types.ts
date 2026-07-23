export interface Adapter {
  readonly name: string;
  parse(stdout: string): Record<string, number>;
  defaults(metricName: string): MetricDefaults;
}

export interface MetricDefaults {
  direction: "lower" | "higher";
  unit?: "ns" | "bytes";
}

export class AdapterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdapterError";
    Object.setPrototypeOf(this, AdapterError.prototype);
  }
}
