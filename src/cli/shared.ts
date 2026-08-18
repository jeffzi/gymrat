import { Command, InvalidArgumentError, Option } from "commander";

import { AdapterError } from "../adapters/index.js";
import type { TargetSpec } from "../compare.js";
import type { CliFlags, ResolvedConfig } from "../config.js";
import { assertNever, GymratError, hintOf, messageOf } from "../errors.js";
import { NotAGitRepositoryError, tryGit } from "../git.js";
import { formatHintLabel, formatLabel, highlightInlineCode } from "../report/format.js";
import type { FailOnCondition, ReportOptions } from "../report/types.js";
import type { RunOptions } from "../sampling.js";
import { acquireLock, type ReleaseLock } from "../session/lock.js";
import { lockfilePath, repoRoot } from "../session/paths.js";
import { installTerminationCleanup } from "../signals.js";
import { MAX_TIMEOUT_SECONDS } from "../timer-limits.js";

// ---------------------------------------------------------------------------
// URL constants
// ---------------------------------------------------------------------------

const BUGS_URL = "https://github.com/jeffzi/gymrat/issues";

// ---------------------------------------------------------------------------
// Exit codes
// ---------------------------------------------------------------------------

/** The exit code of a gate trip: a run that did what it was asked and said no. */
export const GATE_EXIT_CODE = 1;

/** The exit code of a tool failure, the convention every unhandled error exits on. */
export const TOOL_FAILURE_EXIT_CODE = 2;

// ---------------------------------------------------------------------------
// Debug mode
// ---------------------------------------------------------------------------

/** Whether the global `--debug` flag was passed on the command line. */
let debugMode = false;

/** Set the global debug flag read by {@link debugMode} to control verbose output. */
export function setDebugMode(value: boolean): void {
  debugMode = value;
}

// ---------------------------------------------------------------------------
// Stream helpers
// ---------------------------------------------------------------------------

/**
 * `@types/node` declares `isTTY` as `boolean`, but Node leaves it `undefined`
 * on non-TTY streams — the comparison is what makes the boolean honest.
 */
export function isTTY(stream: { isTTY?: boolean }): boolean {
  return stream.isTTY === true;
}

/**
 * Write to a stream and resolve once the chunk has been handed on.
 *
 * `process.exit` drops whatever is still queued, so an immediate exit truncates
 * a report the stream has taken but not yet written out. A `true` return does
 * not mean it has — only that the chunk fit under the high-water mark — so
 * completion is taken from the callback `write` reports it on.
 *
 * The `error` listener stays attached on the failing path: the stream emits
 * `error` right after handing the same error to the callback, and with nothing
 * listening that lands as an unhandled `error` event, which tears the process
 * down before the rejection can be acted on.
 */
export function writeAndFlush(stream: NodeJS.WriteStream, data: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const onError = (error: Error): void => {
      reject(error);
    };
    stream.once("error", onError);

    stream.write(data, (error) => {
      if (error) {
        reject(error);
        return;
      }
      stream.removeListener("error", onError);
      resolve();
    });
  });
}

// ---------------------------------------------------------------------------
// Color control
// ---------------------------------------------------------------------------

/**
 * Veto color unconditionally: `styleText` resolves `FORCE_COLOR` before `NO_COLOR`, so a
 * `FORCE_COLOR` left over from the caller's shell would otherwise defeat `NO_COLOR` and leak
 * ANSI escapes into non-interactive output.
 */
export function suppressColor(): void {
  delete process.env.FORCE_COLOR;
  process.env.NO_COLOR = "1";
}

/**
 * Translate Commander's `--no-color` flag into the color override the report
 * renderer reads: `false` vetoes color, `undefined` defers to auto-detection.
 */
export function colorOverrideOf(options: { color: boolean }): false | undefined {
  return options.color ? undefined : false;
}

// ---------------------------------------------------------------------------
// Error formatting
// ---------------------------------------------------------------------------

/**
 * Render an error for stderr with a red `Error:` label, the message body,
 * and optional diagnostic sections in order:
 *
 * 1. **Error: label** — always present, styled red via `formatLabel`.
 * 2. **Body** — `AdapterError` keeps its class-name prefix after the label
 *    (`Error: AdapterError: <message>`); everything else is the bare message.
 * 3. **Stack trace** — included when `options.debug` is true and the error
 *    carries a stack.
 * 4. **Hint** — appended for `GymratError` instances that carry a hint.
 * 5. **Bug-report footer** — appended for errors that are NOT `GymratError`
 *    (unexpected errors the user should report).
 *
 * Color is governed by `styleText` auto-detection: `NO_COLOR` / `FORCE_COLOR`
 * env vars and the stream's TTY status.
 */
export function formatCliError(error: unknown, options: { debug?: boolean } = {}): string {
  const errorLabel = `${formatLabel("Error", "red", process.stderr)}: `;

  let body: string;
  if (error instanceof AdapterError) {
    body = `${error.name}: ${error.message}`;
  } else {
    body = messageOf(error);
  }

  let output = `${errorLabel}${body}`;

  if (options.debug === true && error instanceof Error && error.stack !== undefined) {
    output += `\n${error.stack}`;
  }

  const hint = hintOf(error);
  if (hint !== undefined) {
    output += `\n${formatHintLabel(process.stderr)} ${highlightInlineCode(hint, process.stderr)}`;
  }

  if (!(error instanceof GymratError)) {
    output += highlightInlineCode(
      `\nRun with \`gymrat --debug\` for details. If this is a bug, please report it at\n${BUGS_URL}`,
      process.stderr,
    );
  }

  return output;
}

/** Print a formatted error to stderr and exit on `code`. */
export async function exitWithError(error: unknown, code = TOOL_FAILURE_EXIT_CODE): Promise<never> {
  try {
    await writeAndFlush(process.stderr, `${formatCliError(error, { debug: debugMode })}\n`);
  } catch {
    /*
     * stderr is the reporting channel, so a write failure (closed pipe, EPIPE)
     * has nowhere to be reported. Swallow it rather than let it escape as an
     * unhandled rejection, which would exit 1 — the code reserved for a gate
     * trip — instead of the code this exit was asked for.
     */
  }
  process.exit(code);
}

// ---------------------------------------------------------------------------
// Flag parsers
// ---------------------------------------------------------------------------

/**
 * Parse the `label=target` syntax of a positional argument, baseline or candidate.
 *
 * Only the first `=` splits, so a target containing its own `=` survives intact —
 * `a=b=c` parses to label `a`, target `b=c`.
 */
export function parsePositional(positional: string): TargetSpec {
  const eqIndex = positional.indexOf("=");
  const label = eqIndex === -1 ? undefined : positional.slice(0, eqIndex);
  const target = eqIndex === -1 ? positional : positional.slice(eqIndex + 1);

  /*
   * An empty half is always a typo, and both halves fail silently rather than
   * loudly: an empty target resolves to the working directory, so `old=` would
   * benchmark whatever the user happens to have checked out, and an empty label
   * leaves the report's header cells holding nothing but ANSI escapes.
   */
  if (label === "") {
    throw new InvalidArgumentError(
      'the label before "=" is empty; write the positional as "label=<ref|dir>" or drop the "=".',
    );
  }
  if (target === "") {
    throw new InvalidArgumentError(
      'the target is empty; write the positional as "[label=]<ref|dir>".',
    );
  }

  return { ...(label !== undefined && { label }), target };
}

/** Accumulate parsed candidate positionals as Commander walks the variadic argument. */
export function collectPositional(value: string, previous: readonly TargetSpec[]): TargetSpec[] {
  return [...previous, parsePositional(value)];
}

/**
 * Build a coercer for a numeric flag value, rejecting anything that is not a
 * positive integer at or below `max`.
 */
export function parsePositiveIntegerUpTo(max: number): (value: string) => number {
  return (value: string): number => {
    const parsed = Number(value);
    if (!/^\d+$/.test(value) || parsed <= 0) {
      throw new InvalidArgumentError("must be a positive integer.");
    }
    if (parsed > max) {
      throw new InvalidArgumentError(`must be a positive integer no greater than ${max}.`);
    }
    return parsed;
  };
}

/** Accept a complete finite decimal — reject trailing garbage, Infinity, and NaN. */
export function parseStopTargetValue(value: string): number {
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(value)) {
    throw new InvalidArgumentError("must be a finite number.");
  }
  return Number(value);
}

/** Parse a strictly positive finite decimal — rejects negatives, zero, Infinity, NaN, and trailing garbage. */
export function parsePositiveNumber(value: string): number {
  if (!/^\d+(\.\d+)?$/.test(value)) {
    throw new InvalidArgumentError("must be a positive number.");
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new InvalidArgumentError("must be a positive number.");
  }
  return parsed;
}

/** `parsePositiveNumber` with an upper bound derived from the 32-bit timer ceiling. */
export function parseMaxMinutes(value: string): number {
  const parsed = parsePositiveNumber(value);
  const maxMinutes = Math.floor(MAX_TIMEOUT_SECONDS / 60);
  if (parsed > maxMinutes) {
    throw new InvalidArgumentError(`must be at most ${String(maxMinutes)} minutes.`);
  }
  return parsed;
}

/**
 * Matches the percentage of a `geomean:<pct>` condition.
 */
const GEOMEAN_CONDITION_RE = /^geomean:(-?\d+(?:\.\d+)?)$/;

/**
 * Accepts `regressed` or `geomean:<number>`. Throws `InvalidArgumentError`
 * for anything else so Commander renders the allowed grammar in the usage error.
 */
export function parseFailOn(
  value: string,
  previous: readonly FailOnCondition[],
): FailOnCondition[] {
  if (value === "regressed") {
    return [...previous, { kind: "regressed" }];
  }

  const match = GEOMEAN_CONDITION_RE.exec(value);
  const pct = match ? Number(match[1]) : NaN;
  if (Number.isFinite(pct)) {
    return [...previous, { kind: "geomean", pct }];
  }

  throw new InvalidArgumentError(
    'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).',
  );
}

// ---------------------------------------------------------------------------
// Option builders
// ---------------------------------------------------------------------------

/**
 * Declare the flags `resolveConfig` reads, in one place.
 */
export function addConfigOptions(command: Command): Command {
  return command
    .option("-b, --bench <cmd>", "bench command")
    .option("-p, --prepare <script>", "preparation script to run before each revision")
    .option("-a, --adapter <type>", "adapter type for parsing benchmark output")
    .option(
      "-s, --samples <number>",
      "paired samples per target",
      parsePositiveIntegerUpTo(Number.MAX_SAFE_INTEGER),
    )
    .option(
      "-t, --timeout <number>",
      "timeout in seconds",
      parsePositiveIntegerUpTo(MAX_TIMEOUT_SECONDS),
    )
    .option("-c, --config <file>", "configuration file path");
}

/**
 * Declare the flags a benchmarking command carries: the configuration set plus
 * the two that govern how its report prints.
 */
export function addSharedOptions(command: Command): Command {
  return addConfigOptions(command)
    .option("--no-color", "print the report without ANSI styles")
    .addOption(
      new Option("--format <value>", "output format").choices(["text", "json"]).default("text"),
    );
}

// ---------------------------------------------------------------------------
// Flag types
// ---------------------------------------------------------------------------

import type { StrictOmit } from "../errors.js";

/** The flags every command carries: everything `resolveConfig` reads, plus how to print. */
export interface SharedFlags extends CliFlags {
  color: boolean;
  format: "text" | "json";
}

/** The status command's flags: the shared set minus the format choice status does not offer. */
export type StatusFlags = StrictOmit<SharedFlags, "format">;

/** The keep command's flags: the configuration set plus the message the commit carries. */
export interface KeepFlags extends CliFlags {
  message?: string;
}

/**
 * The finalize command's flags — the only loop command whose flags are all its
 * own, because closing a session reads no configuration.
 */
export interface FinalizeFlags {
  message?: string;
  branch?: string;
}

/** The measure command's flags: the shared set plus whether the run becomes history. */
export interface MeasureFlags extends SharedFlags {
  record: boolean;
}

/** The compare command's flags: the shared set plus the two only a verdict can answer. */
export interface CompareFlags extends SharedFlags {
  failOn: FailOnCondition[];
  verbose: boolean;
}

/**
 * The doctor command's flags: config options plus format, color, and the bench skip.
 */
export interface DoctorFlags extends StrictOmit<SharedFlags, "bench"> {
  bench: string | boolean | undefined;
}

/** The init command's flags: wizard inputs, not config overrides. */
export interface InitFlags {
  bench?: string;
  adapter?: string;
  checks?: string;
  stopTarget?: number;
  stopMaxIterations?: number;
  primary?: string;
  runbook?: string | boolean;
  skill?: boolean;
  yes: boolean;
}

/** The supervise command's flags: session caps plus where to write the event log. */
export interface SuperviseFlags {
  maxMinutes: number;
  maxUsd?: number;
  log?: string;
  model?: string;
  allowDirty: boolean;
}

// ---------------------------------------------------------------------------
// Locking
// ---------------------------------------------------------------------------

/**
 * The repository a run must serialize against, or `undefined` when there is none.
 */
function lockableRepoRoot(): string | undefined {
  try {
    return repoRoot();
  } catch (error) {
    if (error instanceof NotAGitRepositoryError) {
      return undefined;
    }
    throw error;
  }
}

/**
 * Run `execute`, and route a thrown error through {@link exitWithError}.
 */
export async function runOrExit<T>(
  execute: () => Promise<T>,
  exitCodeOf: (error: unknown) => number = () => TOOL_FAILURE_EXIT_CODE,
): Promise<T> {
  try {
    return await execute();
  } catch (error) {
    return exitWithError(error, exitCodeOf(error));
  }
}

/**
 * Hold the repository's single-flight lock for the length of `run`, routing a
 * thrown error through {@link exitWithError}.
 */
export async function withRepoLock<T>(
  command: string,
  run: () => Promise<T>,
  exitCodeOf: (error: unknown) => number = () => TOOL_FAILURE_EXIT_CODE,
): Promise<T> {
  let root: string | undefined;
  try {
    root = lockableRepoRoot();
  } catch (error) {
    return exitWithError(error);
  }
  if (root === undefined) {
    return await runOrExit(run, exitCodeOf);
  }

  let release: ReleaseLock;
  try {
    release = acquireLock(lockfilePath(root), command);
  } catch (error) {
    return exitWithError(error);
  }

  process.once("exit", release);

  try {
    return await runOrExit(run, exitCodeOf);
  } finally {
    process.removeListener("exit", release);
    release();
  }
}

// ---------------------------------------------------------------------------
// Run infrastructure
// ---------------------------------------------------------------------------

/** Wire `config`'s run settings and `progress`'s callbacks into the shared run fields. */
export function runOptionsOf(
  config: ResolvedConfig,
  progress: ProgressReporter,
): RunOptions & Required<Pick<RunOptions, "onProgress" | "warn">> {
  return {
    bench: config.bench,
    prepare: config.prepare,
    adapter: config.adapter,
    samples: config.samples,
    timeoutSeconds: config.timeoutSeconds,
    configMetrics: config.metrics,
    configKinds: config.kinds,
    onProgress: (step) => {
      progress.emit(step);
    },
    warn: (message) => {
      progress.warn(message);
    },
  };
}

/**
 * Suppress color per `--no-color`, then build the reporter the run streams its
 * progress through for `targetCount` targets.
 */
export function beginRun(options: SharedFlags, targetCount: number): ProgressReporter {
  if (!options.color) {
    suppressColor();
  }
  const tty = isTTY(process.stderr);
  const colorSuppressed = process.env.NO_COLOR !== undefined;
  return createProgressReporter(tty && !colorSuppressed, tty, targetCount);
}

/**
 * Run `execute`, guarded by `progress`: stop the reporter once the run settles.
 */
export async function runGuarded<T>(
  progress: ProgressReporter,
  execute: () => Promise<T>,
): Promise<T> {
  try {
    return await execute();
  } finally {
    progress.stop();
  }
}

/**
 * Run `execute` with an abort signal a termination signal trips, then exit with
 * the conventional `128 + signum`.
 */
export async function runInterruptibly<T>(
  execute: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const run = new AbortController();
  const uninstallTerminationCleanup = installTerminationCleanup(() => {
    run.abort();
  });

  try {
    return await execute(run.signal);
  } finally {
    uninstallTerminationCleanup();
  }
}

/**
 * Render `result` per `options.format` and write it to stdout.
 */
export async function emitReport<T>(
  result: T,
  options: SharedFlags,
  render: { json: (result: T) => string; text: (result: T, textOptions: ReportOptions) => string },
  textOptions: ReportOptions,
): Promise<void> {
  let output: string;
  switch (options.format) {
    case "json":
      output = render.json(result);
      break;
    case "text":
      output = render.text(result, textOptions);
      break;
    default:
      assertNever(options.format);
  }

  await writeAndFlush(process.stdout, output + "\n");
}

// ---------------------------------------------------------------------------
// Git environment
// ---------------------------------------------------------------------------

export interface GitEnvironment {
  gitAvailable: boolean;
  insideGitRepo: boolean;
  repoRootDir?: string;
  gitError?: string;
}

/** Probe git's availability and repository status from `cwd`, without throwing. */
export function detectGitEnvironment(cwd: string): GitEnvironment {
  const gitAvailable = tryGit(["--version"], cwd) === undefined;
  if (!gitAvailable) {
    return { gitAvailable: false, insideGitRepo: false };
  }

  try {
    return { gitAvailable: true, insideGitRepo: true, repoRootDir: repoRoot(cwd) };
  } catch (error) {
    if (error instanceof NotAGitRepositoryError) {
      return { gitAvailable: true, insideGitRepo: false };
    }
    const message = error instanceof Error ? error.message : String(error);
    return { gitAvailable: true, insideGitRepo: true, gitError: message };
  }
}

import { createProgressReporter, type ProgressReporter } from "./progress.js";
export { createProgressReporter, type ProgressReporter } from "./progress.js";
