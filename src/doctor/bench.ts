import { getAdapter } from "../adapters/index.js";
import type { Adapter } from "../adapters/types.js";
import { AdapterError } from "../adapters/types.js";
import {
  DEFAULT_METRIC_KIND,
  GEOMEAN_PRIMARY,
  type ConfigKinds,
  type ConfigMetrics,
} from "../config.js";
import { hintOf, messageOf } from "../errors.js";
import { exec } from "../exec.js";
import type { Check, CheckSection } from "./checks.js";

const MAX_STDERR_EXCERPT_LINES = 5;
const MAX_METRIC_NAMES_SHOWN = 5;

/** Inputs for building the benchmark section of the doctor report. */
export interface BenchSectionInput {
  bench: string | undefined;
  adapter: string;
  timeoutSeconds: number;
  primary: string;
  metrics?: ConfigMetrics;
  kinds?: ConfigKinds;
  repoRoot: string;
  noBench?: boolean;
  configFailed?: boolean;
}

function crossCheckMetrics(
  input: BenchSectionInput,
  metricNames: string[],
  adapter: Adapter,
): Check[] {
  const checks: Check[] = [];

  if (input.primary !== GEOMEAN_PRIMARY && !metricNames.includes(input.primary)) {
    checks.push({
      name: "primary",
      status: "fail",
      detail: `primary "${input.primary}" was not found in parsed metrics`,
    });
  }

  if (input.metrics !== undefined) {
    const missing = Object.keys(input.metrics).filter((m) => !metricNames.includes(m));
    if (missing.length > 0) {
      checks.push({
        name: "metrics",
        status: "warn",
        detail: `Config metrics not found in bench output: ${missing.join(", ")}`,
      });
    }
  }

  if (input.kinds !== undefined) {
    const parsedKinds = new Set(
      metricNames.map((name) => adapter.defaults(name).kind ?? DEFAULT_METRIC_KIND),
    );
    const missing = Object.keys(input.kinds).filter((k) => !parsedKinds.has(k));
    if (missing.length > 0) {
      checks.push({
        name: "kinds",
        status: "warn",
        detail: `Config kinds not matched by any parsed metric: ${missing.join(", ")}`,
      });
    }
  }

  return checks;
}

function benchSection(checks: Check[]): CheckSection {
  return { title: "Bench", checks };
}

async function runAndParseBench(
  bench: string,
  adapter: Adapter,
  input: BenchSectionInput,
): Promise<Check[]> {
  const timeoutMs = input.timeoutSeconds * 1000;
  const result = await exec(bench, { cwd: input.repoRoot, timeoutMs });

  if ("kind" in result) {
    return [
      {
        name: "bench run",
        status: "fail",
        detail: `Bench command timed out after ${String(input.timeoutSeconds)}s`,
        hint: 'Raise the limit with --timeout or the "timeoutSeconds" config key',
      },
    ];
  }

  if (result.exitCode !== 0) {
    const excerpt = result.stderr.trim().split("\n").slice(0, MAX_STDERR_EXCERPT_LINES).join("\n");
    return [
      {
        name: "bench run",
        status: "fail",
        detail: `Bench command exited with code ${String(result.exitCode)}: ${excerpt}`,
      },
    ];
  }

  const warnings: string[] = [];
  let parsed: Record<string, number>;
  try {
    parsed = adapter.parse(result.stdout, (msg) => warnings.push(msg));
  } catch (error: unknown) {
    if (error instanceof AdapterError) {
      return [{ name: "parse", status: "fail", detail: error.message }];
    }
    throw error;
  }

  const metricNames = Object.keys(parsed);
  const nameList =
    metricNames.length <= MAX_METRIC_NAMES_SHOWN
      ? metricNames.join(", ")
      : `${metricNames.slice(0, MAX_METRIC_NAMES_SHOWN).join(", ")} … (${String(metricNames.length)} total)`;
  let detail = `${String(metricNames.length)} metric(s): ${nameList}`;
  if (warnings.length > 0) {
    detail += `\n${warnings.join("\n")}`;
  }

  return [
    { name: "bench run", status: "ok", detail },
    ...crossCheckMetrics(input, metricNames, adapter),
  ];
}

/**
 * Build the "Bench" section of the doctor report by running the bench command
 * once and validating the adapter can parse its output.
 *
 * The section is skipped entirely when `--no-bench` was passed or the config
 * section already failed — a smoke run against broken config proves nothing.
 */
export async function buildBenchSection(input: BenchSectionInput): Promise<CheckSection> {
  let adapter;
  try {
    adapter = getAdapter(input.adapter);
  } catch (error: unknown) {
    const hint = hintOf(error);
    return benchSection([
      {
        name: "adapter",
        status: "fail",
        detail: messageOf(error),
        ...(hint !== undefined ? { hint } : undefined),
      },
    ]);
  }

  if (input.noBench) {
    return benchSection([
      { name: "bench", status: "ok", detail: "Bench smoke run skipped (--no-bench)" },
    ]);
  }

  if (input.configFailed) {
    return benchSection([
      { name: "bench", status: "ok", detail: "Bench smoke run skipped — fix config errors first" },
    ]);
  }

  if (input.bench === undefined) {
    return benchSection([
      {
        name: "bench",
        status: "fail",
        detail: "No bench command resolved",
        hint: 'Set the bench command with --bench or the "bench" config key',
      },
    ]);
  }

  const checks = await runAndParseBench(input.bench, adapter, input);
  return benchSection(checks);
}
