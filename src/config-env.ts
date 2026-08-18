import { MAX_TIMEOUT_SECONDS } from "./timer-limits.js";

/**
 * Check a `GYMRAT_*` string env var, returning the value or a problem string.
 *
 * Returns an empty object when the variable is unset so the next source in the
 * precedence chain (config file, then built-in default) can supply the value.
 */
export function envStringResult(envVar: string): { value?: string; problem?: string } {
  const raw = process.env[envVar];
  if (raw === undefined) return {};
  if (raw === "") {
    return { problem: `Invalid value for ${envVar}: expected a non-empty string, got ""` };
  }
  return { value: raw };
}

/**
 * Check a `GYMRAT_*` numeric env var, returning the parsed value or a problem string.
 *
 * When `max` is supplied the cap is included in the error phrase so the user
 * sees the allowed range in a single message.
 */
export function envPositiveIntResult(
  envVar: string,
  max?: number,
): { value?: number; problem?: string } {
  const raw = process.env[envVar];
  if (raw === undefined) return {};
  const n = Number(raw);
  const phrase =
    max !== undefined ? `a positive integer no greater than ${max}` : "a positive integer";
  if (!Number.isInteger(n) || n < 1 || (max !== undefined && n > max)) {
    return {
      problem: `Invalid value for ${envVar}: expected ${phrase}, got ${JSON.stringify(raw)}`,
    };
  }
  return { value: n };
}

/** One `GYMRAT_*` string field's association between its `CliFlags` key, env var name, and reader. */
export const STRING_ENV_FIELDS: readonly {
  field: "bench" | "prepare" | "adapter";
  envVar: string;
  reader: (envVar: string) => { value?: string; problem?: string };
}[] = [
  { field: "bench", envVar: "GYMRAT_BENCH", reader: envStringResult },
  { field: "prepare", envVar: "GYMRAT_PREPARE", reader: envStringResult },
  { field: "adapter", envVar: "GYMRAT_ADAPTER", reader: envStringResult },
];

/** One `GYMRAT_*` numeric field's association between its `CliFlags` key, env var name, and reader. */
export const NUMBER_ENV_FIELDS: readonly {
  field: "samples" | "timeout";
  envVar: string;
  reader: (envVar: string) => { value?: number; problem?: string };
}[] = [
  { field: "samples", envVar: "GYMRAT_SAMPLES", reader: envPositiveIntResult },
  {
    field: "timeout",
    envVar: "GYMRAT_TIMEOUT",
    reader: (n) => envPositiveIntResult(n, MAX_TIMEOUT_SECONDS),
  },
];
