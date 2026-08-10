#!/usr/bin/env node

import { readFileSync, realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { createHelpConfig } from "@jeffzi/epaulettes";
import { Command, CommanderError, InvalidArgumentError, Option } from "commander";
import yoctoSpinner from "yocto-spinner";

import { AdapterError } from "./adapters/index.js";
import { compare } from "./compare.js";
import type { CompareOptions, ProgressStep, TargetSpec } from "./compare.js";
import { resolveConfig, type CliFlags, type ResolvedConfig } from "./config.js";
import { assertNever, GymratError, messageOf } from "./errors.js";
import { EtaTracker, formatEta } from "./eta.js";
import { iterateSession, LoopStopError } from "./loop/iterate.js";
import { discardSession, keepSession } from "./loop/settle.js";
import { startSession, type StartResult } from "./loop/start.js";
import { statusSession } from "./loop/status.js";
import { measure } from "./measure.js";
import type { MeasureOptions } from "./measure.js";
import { metricRecord } from "./metric-record.js";
import { countVerdicts, formatHintLabel, formatLabel, shortenLabel } from "./report/format.js";
import { renderJson, renderMeasureJson } from "./report/json.js";
import { formatBaselineRef } from "./report/loop.js";
import { pluralize, renderMeasureReport, renderReport } from "./report/text.js";
import type {
  CandidateComparison,
  ComparisonResult,
  FailOnCondition,
  MeasurementResult,
  MetricComparisons,
  ReportOptions,
} from "./report/types.js";
import type { RunOptions } from "./sampling.js";
import { acquireLock, type ReleaseLock } from "./session/lock.js";
import { lockfilePath, repoRoot, sessionJsonlPath } from "./session/paths.js";
import type { BaselineRecord } from "./session/records.js";
import { appendRecord, foldSession, readRecords } from "./session/store.js";
import { MAX_TIMEOUT_SECONDS } from "./timer-limits.js";
import type { GeomeanResult } from "./verdict/verdict.js";

/**
 * Parse the `label=target` syntax of a positional argument, baseline or candidate.
 *
 * Only the first `=` splits, so a target containing its own `=` survives intact —
 * `a=b=c` parses to label `a`, target `b=c`.
 */
function parsePositional(positional: string): TargetSpec {
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

  return { label, target };
}

/** Accumulate parsed candidate positionals as Commander walks the variadic argument. */
function collectPositional(value: string, previous: readonly TargetSpec[]): TargetSpec[] {
  return [...previous, parsePositional(value)];
}

/**
 * Build a coercer for a numeric flag value, rejecting anything that is not a
 * positive integer at or below `max`.
 *
 * `parseInt` is too permissive for flag values: it returns `NaN` for
 * non-numeric input and silently truncates trailing garbage (`"10abc"` → `10`).
 * `NaN` is not nullish, so it survives the `??` default chain in
 * `resolveConfig` and only surfaces much later as a confusing benchmark error.
 *
 * Throwing `InvalidArgumentError` lets Commander render a usage error naming
 * the offending flag and value.
 */
function parsePositiveIntegerUpTo(max: number): (value: string) => number {
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

/**
 * Matches the percentage of a `geomean:<pct>` condition.
 *
 * The percentage is pinned to plain decimal notation because `Number` is far
 * looser than the grammar this flag documents: it reads `""` and `" "` as 0 and
 * `"0x10"` as 16, so a truncated or mistyped threshold would silently become a
 * gate the user never asked for.
 */
const GEOMEAN_CONDITION_RE = /^geomean:(-?\d+(?:\.\d+)?)$/;

/**
 * Accepts `regressed` or `geomean:<number>`. Throws `InvalidArgumentError`
 * for anything else so Commander renders the allowed grammar in the usage error.
 */
function parseFailOn(value: string, previous: readonly FailOnCondition[]): FailOnCondition[] {
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

/**
 * Non-gating metrics never participate in gate evaluation — their verdicts
 * are informational and must not trip an exit-code gate.
 */
function gatingMetrics(metrics: MetricComparisons): MetricComparisons {
  return metricRecord(Object.entries(metrics).filter(([, metric]) => metric.meta.gating));
}

/**
 * The gated geomean of every kind that gates, one entry per such kind.
 *
 * Each kind is judged independently so the gate never mixes units across kinds
 * and an informational kind cannot move a number the user asked to be judged on.
 */
function gatedGeomeansOf(candidate: CandidateComparison): readonly GeomeanResult[] {
  return candidate.kinds.flatMap((kind) => (kind.gatedGeomean ? [kind.gatedGeomean] : []));
}

/**
 * Returns `true` when any condition trips — meaning the process should exit 1.
 *
 * A geomean threshold is inclusive: `geomean:2` trips at exactly +2.0%, matching
 * the "at or worse" contract the README states. It is evaluated per gating kind,
 * so any one kind crossing the threshold trips the gate.
 *
 * A gated geomean that aggregated nothing (`n === 0`) can never trip. Its `value`
 * is a placeholder 0, so comparing it against a threshold of 0 or less would
 * report a regression nobody measured; {@link warnEmptyGeomeanGates} tells the
 * user the gate was vacuous instead.
 */
function shouldFailGate(conditions: readonly FailOnCondition[], result: ComparisonResult): boolean {
  if (conditions.length === 0) return false;

  const gating = gatingMetrics(result.metrics);

  return conditions.some((condition) => {
    switch (condition.kind) {
      case "regressed":
        return result.candidates.some((_, i) => countVerdicts(gating, i).regressed > 0);
      case "geomean":
        return result.candidates.some((candidate) =>
          gatedGeomeansOf(candidate).some(
            (geomean) => geomean.n > 0 && geomean.value >= condition.pct,
          ),
        );
      default:
        return assertNever(condition);
    }
  });
}

/**
 * Warn once per candidate whose geomean gate has nothing to judge.
 *
 * Every gating metric being excluded — all unstable, or all turned off in the
 * config — leaves a geomean gate that can never trip. Passing silently would
 * read as "the threshold held"; the warning distinguishes that from a gate that
 * was never evaluated.
 *
 * A candidate with no gating kind at all is vacuous by the same measure: nothing
 * it ran is judged, so `every` over the empty list warns.
 */
function warnEmptyGeomeanGates(
  conditions: readonly FailOnCondition[],
  result: ComparisonResult,
): void {
  if (!conditions.some((condition) => condition.kind === "geomean")) return;

  for (const candidate of result.candidates) {
    if (gatedGeomeansOf(candidate).every((geomean) => geomean.n === 0)) {
      process.stderr.write(
        `warning: geomean gate for "${candidate.label}" had no stable gating metrics to measure\n`,
      );
    }
  }
}

/**
 * `@types/node` declares `isTTY` as boolean, but node leaves it `undefined` when
 * the stream is not a TTY. Naming the real type in one place keeps every caller's
 * declared `boolean` honest instead of quietly handing back `undefined`.
 */
function isTerminal(stream: NodeJS.WriteStream): boolean {
  return (stream.isTTY as boolean | undefined) === true;
}

/**
 * Veto color unconditionally: `styleText` resolves `FORCE_COLOR` before `NO_COLOR`, so a
 * `FORCE_COLOR` left over from the caller's shell would otherwise defeat `NO_COLOR` and leak
 * ANSI escapes into non-interactive output.
 */
function suppressColor(): void {
  delete process.env.FORCE_COLOR;
  process.env.NO_COLOR = "1";
}

/**
 * Render an error for stderr: either an adapter failure labelled with its
 * class name, or any other error with a styled hint line appended when it
 * carries one. The two are mutually exclusive — the adapter branch returns
 * before the hint block runs, so an adapter error never gets a hint.
 *
 * An adapter message states what could not be parsed ("No valid METRIC lines
 * found") without saying which layer produced it, so the class name is what
 * tells the user the bench script's output — rather than gymrat's git or config
 * handling — is at fault. Errors raised elsewhere already name their own
 * subsystem, so prefixing them would only add noise.
 *
 * Color is governed by `styleText` auto-detection: `NO_COLOR` / `FORCE_COLOR`
 * env vars and the stream's TTY status.
 */
export function formatCliError(error: unknown): string {
  if (error instanceof AdapterError) {
    return `${error.name}: ${error.message}`;
  }

  let output = messageOf(error);

  if (error instanceof GymratError && error.hint !== undefined) {
    output += `\n${formatHintLabel(process.stderr)} ${error.hint}`;
  }

  return output;
}

/**
 * Write to a stream and resolve once the data has left the internal buffer.
 *
 * `process.exit` drops whatever is still queued, so a report larger than the
 * pipe buffer (64 KiB on most systems) is truncated when an immediate exit
 * follows the write. A `false` return means the data was buffered; `drain`
 * fires once it has been flushed.
 */
function writeAndDrain(stream: NodeJS.WriteStream, data: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const cleanup = (): void => {
      stream.removeListener("error", onError);
      stream.removeListener("drain", onDrain);
    };
    const onError = (error: Error): void => {
      cleanup();
      reject(error);
    };
    const onDrain = (): void => {
      cleanup();
      resolve();
    };

    stream.once("error", onError);

    if (stream.write(data)) {
      cleanup();
      resolve();
      return;
    }
    stream.once("drain", onDrain);
  });
}

/** The exit code of a gate trip: a run that did what it was asked and said no. */
const GATE_EXIT_CODE = 1;

/** The exit code of a tool failure, the convention every unhandled error exits on. */
const TOOL_FAILURE_EXIT_CODE = 2;

/** Print a formatted error to stderr and exit on `code`. */
async function exitWithError(error: unknown, code = TOOL_FAILURE_EXIT_CODE): Promise<never> {
  try {
    await writeAndDrain(process.stderr, `${formatCliError(error)}\n`);
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

/** Shown after a sample step until enough gaps have been measured for an ETA. */
const ETA_PENDING_LABEL = "estimating time left…";

/** Structured segments shared by {@link formatProgressLine} and {@link styleProgressLine}. */
interface ProgressLineParts {
  readonly stepWord: string;
  readonly counter: string | undefined;
  readonly label: string;
  readonly etaSuffix: string | undefined;
}

/**
 * Derive the step word, counter, label, and ETA suffix for a progress line,
 * leaving presentation (plain text vs. ANSI styling) to the caller.
 */
function buildProgressLineParts(step: ProgressStep, etaMs?: number): ProgressLineParts {
  let etaSuffix: string | undefined;
  if (etaMs !== undefined) {
    etaSuffix = formatEta(etaMs);
  } else if (step.kind === "sample") {
    etaSuffix = ETA_PENDING_LABEL;
  }

  switch (step.kind) {
    case "prepare":
      return { stepWord: "prepare", counter: undefined, label: step.label, etaSuffix };
    case "sample":
      return {
        stepWord: "sample",
        counter: `${step.index}/${step.total}`,
        label: step.label,
        etaSuffix,
      };
    default:
      return assertNever(step);
  }
}

function joinStepLine(stepWord: string, counter: string | undefined, label: string): string {
  return counter === undefined ? `${stepWord} · ${label}` : `${stepWord} ${counter} · ${label}`;
}

/** Per-field presentation applied by {@link renderProgressLine}: identity for plain text, ANSI styling for the spinner. */
interface ProgressLineStyle {
  readonly label: (text: string) => string;
  readonly counter: (text: string) => string;
  readonly eta: (text: string) => string;
}

/**
 * Assemble a progress line from its parts, applying `style` to the label,
 * counter, and ETA suffix. `formatProgressLine` and `styleProgressLine` share
 * this so their output only differs in presentation, never in structure.
 */
function renderProgressLine(
  step: ProgressStep,
  etaMs: number | undefined,
  style: ProgressLineStyle,
): string {
  const { stepWord, counter, label, etaSuffix } = buildProgressLineParts(step, etaMs);
  const styledCounter = counter === undefined ? undefined : style.counter(counter);
  let line = joinStepLine(stepWord, styledCounter, style.label(label));
  if (etaSuffix !== undefined) {
    line += style.eta(` · ${etaSuffix}`);
  }
  return line;
}

const IDENTITY_PROGRESS_LINE_STYLE: ProgressLineStyle = {
  label: (text) => text,
  counter: (text) => text,
  eta: (text) => text,
};

/**
 * Each line names the target so the user can tell which one is running;
 * samples also show their position in the total.
 * When an ETA estimate is available, it is appended as plain text.
 */
function formatProgressLine(step: ProgressStep, etaMs?: number): string {
  return renderProgressLine(step, etaMs, IDENTITY_PROGRESS_LINE_STYLE);
}

/**
 * Apply ANSI styling to the progress line for spinner display:
 * step word in default foreground, counter bold (sample only), label cyan.
 * When an ETA estimate is available, it is appended as a dim segment.
 * When no ETA is available for a sample step, a dim placeholder is shown.
 */
function styleProgressLine(step: ProgressStep, etaMs?: number): string {
  return renderProgressLine(step, etaMs, {
    label: (text) => formatLabel(text, "cyan", process.stderr),
    counter: (text) => formatLabel(text, "bold", process.stderr),
    eta: (text) => formatLabel(text, "dim", process.stderr),
  });
}

/** Carriage-return + clear-to-EOL: rewinds the cursor and erases the line it lands on. */
const CLEAR_LINE = "\r\x1b[K";

/**
 * Cut `line` down to a single terminal row.
 *
 * {@link CLEAR_LINE} erases the row the cursor lands on and nothing else, so a
 * line that wrapped leaves its first rows on screen as stale fragments. Stopping
 * one column short of the width keeps terminals that wrap as soon as the last
 * column is written from spilling onto a second row.
 *
 * The cut takes the middle so both ends of the line survive a narrow terminal.
 *
 * `@types/node` declares `columns` as a number, but node defines it only on a
 * TTY; elsewhere it is `undefined` — and a non-TTY stream has no width to fit.
 */
function fitToTerminalWidth(line: string): string {
  const columns = process.stderr.columns as number | undefined;
  if (columns === undefined) {
    return line;
  }
  return shortenLabel(line, columns - 1);
}

/**
 * Non-TTY output must stay free of ANSI escapes; TTY output overwrites in
 * place using {@link CLEAR_LINE} so only one progress line is ever visible.
 */
function writeProgress(line: string, tty: boolean): void {
  process.stderr.write(tty ? `${CLEAR_LINE}${fitToTerminalWidth(line)}` : `${line}\n`);
}

/**
 * Erase the last in-place progress line so the report starts on a clean row.
 *
 * Only meaningful in TTY mode — non-TTY lines are already newline-terminated
 * and cannot be erased.
 */
function clearProgress(tty: boolean): void {
  if (tty) {
    process.stderr.write(CLEAR_LINE);
  }
}

/**
 * One warning line on stderr.
 *
 * Written through the same stream as the progress line rather than `console`, so
 * the clear/warn/redraw sequence cannot be reordered by a different writer.
 */
function writeWarning(message: string): void {
  process.stderr.write(`${message}\n`);
}

/** Single-use: `stop()` must be called exactly once, after the run completes or fails. */
interface ProgressReporter {
  emit(step: ProgressStep): void;
  /**
   * Print `message` on a row of its own, giving the progress line back afterwards.
   *
   * A warning written while an in-place progress line is on screen would
   * otherwise be spliced into it, leaving both unreadable.
   */
  warn(message: string): void;
  /** Stop the spinner or erase the in-place line so the next output starts clean. */
  stop(): void;
}

/** How often the spinner's ETA countdown ticks down while waiting for the next step. */
const COUNTDOWN_TICK_MS = 1000;

/**
 * TTY + color allowed: use yocto-spinner (yellow glyph on stderr).
 * TTY + color vetoed: fall back to \r\x1b[K overwrite with plain text —
 *   yocto-spinner's frame color cannot be disabled through its API.
 * Non-TTY: one newline-terminated line per step, no ANSI.
 */
function createProgressReporter(
  colorAllowed: boolean,
  tty: boolean,
  targetCount: number,
): ProgressReporter {
  const spinner = colorAllowed
    ? yoctoSpinner({ color: "yellow", stream: process.stderr })
    : undefined;
  const eta = new EtaTracker(targetCount);
  let countdownInterval: ReturnType<typeof setInterval> | undefined;
  /** The step the in-place line currently shows, so `warn` can put it back. */
  let drawnStep: { step: ProgressStep; etaMs?: number } | undefined;

  function clearCountdown(): void {
    if (countdownInterval !== undefined) {
      clearInterval(countdownInterval);
      countdownInterval = undefined;
    }
  }

  /** Tick the spinner text down toward zero every second until the next `emit`. */
  function startCountdown(
    activeSpinner: ReturnType<typeof yoctoSpinner>,
    step: ProgressStep,
    etaMs: number,
  ): void {
    const emitTime = Date.now();
    countdownInterval = setInterval(() => {
      const remaining = Math.max(0, etaMs - (Date.now() - emitTime));
      activeSpinner.text = styleProgressLine(step, remaining);
    }, COUNTDOWN_TICK_MS);
  }

  return {
    emit(step: ProgressStep): void {
      const etaMs = eta.record(step);
      if (!spinner) {
        drawnStep = tty ? { step, etaMs } : undefined;
        writeProgress(formatProgressLine(step, etaMs), tty);
        return;
      }

      spinner.text = styleProgressLine(step, etaMs);
      if (!spinner.isSpinning) {
        spinner.start();
      }
      clearCountdown();
      if (etaMs !== undefined) {
        startCountdown(spinner, step, etaMs);
      }
    },
    warn(message: string): void {
      if (spinner) {
        // The spinner redraws itself on its next frame, so erasing the current
        // one is all it takes to hand the row over.
        spinner.clear();
        writeWarning(message);
        return;
      }

      if (!drawnStep) {
        writeWarning(message);
        return;
      }

      clearProgress(tty);
      writeWarning(message);
      writeProgress(formatProgressLine(drawnStep.step, drawnStep.etaMs), tty);
    },
    stop(): void {
      clearCountdown();
      drawnStep = undefined;
      if (spinner) {
        spinner.stop();
      } else {
        clearProgress(tty);
      }
    },
  };
}

/** The flags every command carries: everything `resolveConfig` reads, plus how to print. */
interface SharedFlags extends CliFlags {
  /** Commander's `--no-color` counterpart: true unless the flag was passed. */
  color: boolean;
  /** Output format — `text` produces the ANSI-styled table, `json` produces plain output. */
  format: "text" | "json";
}

/** The status command's flags: the configuration set plus whether its report may be styled. */
interface StatusFlags extends CliFlags {
  /** Commander's `--no-color` counterpart: true unless the flag was passed. */
  color: boolean;
}

/** The keep command's flags: the configuration set plus the message the commit carries. */
interface KeepFlags extends CliFlags {
  /** Commit message the agent supplied; absent, keep generates one from the iteration. */
  message?: string;
}

/** The measure command's flags: the shared set plus whether the run becomes history. */
interface MeasureFlags extends SharedFlags {
  /** Append the run to the session log. Recording is opt-in, never the default. */
  record: boolean;
}

/** The compare command's flags: the shared set plus the two only a verdict can answer. */
interface CompareFlags extends SharedFlags {
  /** Gate conditions that cause exit 1 when any trips. Empty when `--fail-on` is not used. */
  failOn: FailOnCondition[];
  /** Name the statistical method behind each verdict in the report footer. */
  verbose: boolean;
}

/**
 * Declare the flags `resolveConfig` reads, in one place.
 *
 * Every command settles its configuration through the same resolver, so a flag
 * added to one and forgotten on another is a bug the user meets as an "unknown
 * option". Declaring them here makes that impossible: the option sets cannot
 * drift because there is only one definition.
 */
function addConfigOptions(command: Command): Command {
  return command
    .option("--bench <cmd>", "bench command")
    .option("--prepare <script>", "preparation script to run before each revision")
    .option("--adapter <type>", "adapter type for parsing benchmark output")
    .option(
      "--samples <number>",
      "paired samples per target",
      parsePositiveIntegerUpTo(Number.MAX_SAFE_INTEGER),
    )
    .option(
      "--timeout <number>",
      "timeout in seconds",
      parsePositiveIntegerUpTo(MAX_TIMEOUT_SECONDS),
    )
    .option("--config <file>", "configuration file path");
}

/**
 * Declare the flags a benchmarking command carries: the configuration set plus
 * the two that govern how its report prints.
 */
function addSharedOptions(command: Command): Command {
  return addConfigOptions(command)
    .option("--no-color", "print the report without ANSI styles")
    .addOption(
      new Option("--format <value>", "output format").choices(["text", "json"]).default("text"),
    );
}

/** The `resolveConfig` view of a parsed flag set: the run settings, without the presentation ones. */
function configFlagsOf(options: SharedFlags): CliFlags {
  return {
    bench: options.bench,
    prepare: options.prepare,
    adapter: options.adapter,
    samples: options.samples,
    timeout: options.timeout,
    config: options.config,
  };
}

/** Wire `config`'s run settings and `progress`'s callbacks into the shared `CompareOptions`/`MeasureOptions` fields. */
function runOptionsOf(
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
 *
 * The veto must run first: the reporter reads `NO_COLOR` rather than the parsed
 * flag, so it only sees the veto once `suppressColor()` has applied it — whether
 * `NO_COLOR` came from `--no-color`, the user's environment, or `suppressColor`
 * itself.
 */
function beginRun(options: SharedFlags, targetCount: number): ProgressReporter {
  if (!options.color) {
    suppressColor();
  }
  const tty = isTerminal(process.stderr);
  const colorSuppressed = process.env.NO_COLOR !== undefined;
  return createProgressReporter(tty && !colorSuppressed, tty, targetCount);
}

/**
 * `--no-color` is a veto, never a force: left unset, each renderer keeps its own
 * detection rather than being told what the stream supports.
 */
function noColorOverride(colorFlag: boolean): boolean | undefined {
  return colorFlag ? undefined : false;
}

/**
 * Run `execute`, guarded by `progress`: stop the reporter once the run settles,
 * whether it succeeds or throws. `compare` and `measure` share this so neither
 * command's action can forget to stop the progress reporter before a failure
 * reaches `withRepoLock`, which is what routes a thrown error to
 * {@link exitWithError} — `runGuarded` re-throws rather than exiting itself, so
 * that single path is the only place either command's errors are reported.
 */
async function runGuarded<T>(progress: ProgressReporter, execute: () => Promise<T>): Promise<T> {
  try {
    const result = await execute();
    progress.stop();
    return result;
  } catch (error) {
    progress.stop();
    throw error;
  }
}

/**
 * The repository a run must serialize against, or `undefined` when there is none.
 *
 * Both commands accept plain directories, so standing outside a git repository
 * is a supported way to run gymrat rather than a failure: it just leaves nothing
 * to take a lock on.
 */
function lockableRepoRoot(): string | undefined {
  try {
    return repoRoot();
  } catch (error) {
    if (error instanceof GymratError) {
      return undefined;
    }
    throw error;
  }
}

/** `withRepoLock`'s default: every uncaught error is a tool failure. */
function toolFailure(): number {
  return TOOL_FAILURE_EXIT_CODE;
}

/**
 * Run `execute`, and route a thrown error through {@link exitWithError} instead
 * of letting it escape — `exitCodeOf` picks the exit code the error deserves,
 * defaulting to a tool failure.
 *
 * `withRepoLock` calls this for both the locked and unlocked paths; `status`,
 * the one loop command that takes no lock, calls it directly so its error
 * handling still goes through the same path.
 */
async function runOrExit<T>(
  execute: () => Promise<T>,
  exitCodeOf: (error: unknown) => number = toolFailure,
): Promise<T> {
  try {
    return await execute();
  } catch (error) {
    return exitWithError(error, exitCodeOf(error));
  }
}

/**
 * Hold the repository's single-flight lock for the length of `run`, routing a
 * thrown error through {@link exitWithError} — `exitCodeOf` picks the exit code,
 * defaulting to a tool failure.
 *
 * The lock is taken before anything the run would execute, so a repository
 * another gymrat process is already benchmarking is refused before a prepare or
 * bench command can perturb its measurements.
 *
 * Release is wired twice because the two ways a command finishes need different
 * hooks: `finally` covers a run that returns or throws, and the `exit` listener
 * covers `process.exit`, which unwinds nothing — the error path and the
 * `--fail-on` gate trip both leave that way. `runOrExit`'s catch runs before
 * that `finally`, while the `exit` listener is still registered, so the
 * `process.exit` inside `exitWithError` still triggers it and releases the lock.
 */
async function withRepoLock<T>(
  command: string,
  run: () => Promise<T>,
  exitCodeOf: (error: unknown) => number = toolFailure,
): Promise<T> {
  const root = lockableRepoRoot();
  if (root === undefined) {
    return await runOrExit(run, exitCodeOf);
  }

  let release: ReleaseLock;
  try {
    release = acquireLock(lockfilePath(root), command);
  } catch (error) {
    return exitWithError(error);
  }

  const releaseOnExit = (): void => {
    release();
  };
  process.once("exit", releaseOnExit);

  try {
    return await runOrExit(run, exitCodeOf);
  } finally {
    process.removeListener("exit", releaseOnExit);
    release();
  }
}

/**
 * Render `result` per `options.format` and write it to stdout with a trailing
 * newline. `compare` and `measure` share this so the format switch — and the
 * `assertNever` exhaustiveness guard on it — exists once.
 */
async function emitReport<T>(
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

  await writeAndDrain(process.stdout, output + "\n");
}

/** Where a recorded run writes, and which session it becomes part of. */
interface RecordingTarget {
  jsonlPath: string;
  sessionId: string;
}

/**
 * The open session `--record` appends to.
 *
 * @throws GymratError when the repository has no session — there is nowhere to
 *   record, and only `gymrat start` can change that.
 */
function recordingTarget(root: string): RecordingTarget {
  const jsonlPath = sessionJsonlPath(root);
  const { session } = foldSession(readRecords(jsonlPath));
  if (session === undefined) {
    throw new GymratError(
      `No session in ${root}`,
      "Run gymrat start to open one before recording a measurement.",
    );
  }
  return { jsonlPath, sessionId: session.sessionId };
}

/** The history entry a recorded run leaves behind: what it measured, round by round. */
function baselineRecordOf(result: MeasurementResult): BaselineRecord {
  return {
    type: "baseline",
    at: new Date().toISOString(),
    label: result.label,
    samples: [...result.rounds],
  };
}

/**
 * The summary `start` prints: where the session's work happens, and — when the
 * session was already there — how far it has got.
 *
 * Both worktree paths are named because the agent driving the loop edits in one
 * of them and must never touch the other.
 */
function formatStartSummary(result: StartResult): string {
  const { session, state } = result;
  const headline = result.resumed
    ? `Resumed session ${session.sessionId} — ${pluralize(state.iterationCount, "iteration")}, ${pluralize(state.keepCount, "keep")}`
    : `Started session ${session.sessionId}`;

  const rows: readonly (readonly [string, string])[] = [
    ["branch", session.branch],
    ["baseline", formatBaselineRef(session.baseline)],
    ["experiment worktree", session.worktrees.experiment],
    ["baseline worktree", session.worktrees.baseline],
  ];
  const labelWidth = Math.max(...rows.map(([label]) => label.length)) + 1;

  return [
    headline,
    ...rows.map(([label, value]) => `  ${`${label}:`.padEnd(labelWidth)} ${value}`),
  ].join("\n");
}

/**
 * `import ... with { type: "json" }` would be tidier, but package.json sits
 * outside `rootDir`, so importing it would pull an extra directory level into
 * `dist/` and break the `dist/cli.js` bin path. Reading at runtime keeps the
 * emitted layout flat and keeps the reported version in lockstep with the
 * manifest instead of a literal that has to be bumped by hand.
 *
 * The relative path resolves to the package root from both `src/cli.ts` (tests)
 * and `dist/cli.js` (published), and npm always ships package.json regardless
 * of the `files` allowlist.
 */
function readPackageVersion(): string {
  const manifest: unknown = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );

  if (
    typeof manifest !== "object" ||
    manifest === null ||
    !("version" in manifest) ||
    typeof manifest.version !== "string"
  ) {
    throw new GymratError("package.json has no string version field");
  }

  return manifest.version;
}

/**
 * Build the `gymrat` program: fully wired but inert until the caller invokes
 * `parseAsync` on it.
 *
 * `readPackageVersion()` runs during construction, so building a program reads
 * `package.json` off disk even before any command executes.
 */
export function createProgram(): Command {
  const noTargets: TargetSpec[] = [];
  const noFailOnConditions: FailOnCondition[] = [];
  /** What `measure` benches when given no target: the tree the user is standing in. */
  const currentDirectory: TargetSpec = { target: "." };

  const program = new Command();

  program
    .name("gymrat")
    .description("Performance comparison tool for benchmarks")
    .version(readPackageVersion());

  const compareCmd = addSharedOptions(
    program
      .command("compare")
      .description("Compare one baseline revision against one or more candidates")
      .argument("<baseline>", "[label=]<ref|dir> to measure against", parsePositional)
      .argument(
        "<candidates...>",
        "[label=]<ref|dir>, each judged against the baseline",
        collectPositional,
        noTargets,
      ),
  )
    .option("--verbose", "name the statistical method behind each verdict", false)
    .option(
      "--fail-on <condition>",
      'exit 1 when a condition trips (repeatable: "regressed", "geomean:<pct>")',
      parseFailOn,
      noFailOnConditions,
    )
    .configureHelp(createHelpConfig())
    .action(async (baseline: TargetSpec, candidates: TargetSpec[], options: CompareFlags) => {
      /*
       * The gate check runs after the lock is released, as `keep`'s does: it
       * only reads the result the lock already protected, and process.exit
       * must not be reachable from inside withRepoLock's own error handling.
       */
      const result = await withRepoLock("compare", async () => {
        const progress = beginRun(options, 1 + candidates.length);

        const compareResult = await runGuarded(progress, async () => {
          const config = resolveConfig(configFlagsOf(options));

          const compareOptions: CompareOptions = {
            baseline,
            candidates,
            ...runOptionsOf(config, progress),
            unstableNoisePct: config.unstableNoisePct,
          };

          return compare(compareOptions);
        });

        await emitReport(
          compareResult,
          options,
          { json: renderJson, text: renderReport },
          {
            verbose: options.verbose,
            color: noColorOverride(options.color),
            failOn: options.failOn,
          },
        );

        warnEmptyGeomeanGates(options.failOn, compareResult);

        return compareResult;
      });

      if (shouldFailGate(options.failOn, result)) {
        process.exit(GATE_EXIT_CODE);
      }
    });

  const measureCmd = addSharedOptions(
    program
      .command("measure")
      .description("Measure one revision or directory on its own, with nothing to compare it to")
      .argument(
        "[target]",
        "[label=]<ref|dir> to measure; defaults to the current directory",
        parsePositional,
        currentDirectory,
      ),
  )
    .option("-r, --record", "append the run to the session log as a baseline", false)
    .configureHelp(createHelpConfig())
    .action(async (target: TargetSpec, options: MeasureFlags) => {
      await withRepoLock("measure", async () => {
        // One target, so every sample step the run reports is a step of the whole run.
        const progress = beginRun(options, 1);

        const run = await runGuarded(progress, async () => {
          const config = resolveConfig(configFlagsOf(options));

          /*
           * The session is resolved before a single sample is taken: a run that
           * benches for ten minutes and only then discovers it has nowhere to
           * write has thrown the whole measurement away.
           */
          const recording = options.record ? recordingTarget(repoRoot()) : undefined;

          const measureOptions: MeasureOptions = {
            target,
            ...runOptionsOf(config, progress),
          };

          return { result: await measure(measureOptions), recording };
        });

        await emitReport(
          run.result,
          options,
          { json: renderMeasureJson, text: renderMeasureReport },
          { color: noColorOverride(options.color) },
        );

        if (run.recording !== undefined) {
          appendRecord(run.recording.jsonlPath, baselineRecordOf(run.result));
          // The note is prose, so it goes to stderr whenever stdout is carrying
          // JSON a consumer has to parse.
          await writeAndDrain(
            options.format === "json" ? process.stderr : process.stdout,
            `baseline "${run.result.label}" recorded to session ${run.recording.sessionId}\n`,
          );
        }
      });
    });

  const startCmd = addConfigOptions(
    program
      .command("start")
      .description("Create or resume this repository's optimization session")
      .argument("[ref]", "ref the baseline is pinned to; defaults to HEAD"),
  )
    .configureHelp(createHelpConfig())
    .action(async (ref: string | undefined, options: CliFlags) => {
      /*
       * The summary prints after the lock is released: it describes a workspace
       * that is already on disk, so holding the repository against every other
       * gymrat process while stdout drains would serialize runs for nothing.
       */
      const result = await withRepoLock("start", () =>
        Promise.resolve(startSession(repoRoot(), ref, resolveConfig(options))),
      );

      await writeAndDrain(process.stdout, `${formatStartSummary(result)}\n`);
    });

  const iterateCmd = addConfigOptions(
    program
      .command("iterate")
      .description("Measure the session's experiment worktree against its baseline"),
  )
    .configureHelp(createHelpConfig())
    .action(async (options: CliFlags) => {
      /*
       * The report prints after the lock is released, as `start`'s summary does:
       * the measurement is over by then, and holding the repository while stdout
       * drains would serialize runs for nothing.
       */
      const result = await withRepoLock(
        "iterate",
        async () => iterateSession(repoRoot(), resolveConfig(options)),
        /*
         * A met stop condition is a gate trip, not a tool failure: the loop
         * reached the end it was configured for, so it exits the way a keep
         * the checks refused does.
         */
        (error) => (error instanceof LoopStopError ? GATE_EXIT_CODE : TOOL_FAILURE_EXIT_CODE),
      );

      await writeAndDrain(process.stdout, `${result.report}\n`);
    });

  const keepCmd = addConfigOptions(
    program
      .command("keep")
      .description("Commit the session's measured edit once its checks pass")
      .option("-m, --message <text>", "commit message for the kept edit"),
  )
    .configureHelp(createHelpConfig())
    .action(async (options: KeepFlags) => {
      const result = await withRepoLock("keep", async () =>
        keepSession(repoRoot(), resolveConfig(options), { message: options.message }),
      );

      await writeAndDrain(process.stdout, `${result.report}\n`);

      // A refused keep is a gate trip, not a tool failure: the record is written
      // and reported, and only the exit code tells the agent it did not land.
      if (result.record.status === "blocked") {
        process.exit(GATE_EXIT_CODE);
      }
    });

  /*
   * The only settle command that reads no configuration: a revert is git alone,
   * so `discard` carries none of the config flags its siblings do.
   */
  const discardCmd = program
    .command("discard")
    .description("Revert the session's experiment worktree to its last commit")
    .configureHelp(createHelpConfig())
    .action(async () => {
      const result = await withRepoLock("discard", () =>
        Promise.resolve(discardSession(repoRoot())),
      );

      await writeAndDrain(process.stdout, `${result.report}\n`);
    });

  /*
   * The one loop command that takes no repository lock: it reads the session
   * log and writes nothing, so serializing it against a running iterate would
   * only make the agent wait to be told what is already on disk.
   */
  const statusCmd = addConfigOptions(
    program
      .command("status")
      .description("Show this repository's session history, read from its log")
      .option("--no-color", "print the report without ANSI styles"),
  )
    .configureHelp(createHelpConfig())
    .action(async (options: StatusFlags) => {
      // Suppressed before rendering, not after: the lines style themselves as
      // they are built, and `styleText` reads the environment on every call.
      if (!options.color) {
        suppressColor();
      }

      const report = await runOrExit(() =>
        Promise.resolve(statusSession(repoRoot(), resolveConfig(options))),
      );

      await writeAndDrain(process.stdout, `${report}\n`);
    });

  /*
   * Commander exits 1 for usage errors by default. Override so all Commander
   * errors (unknown option, missing argument, invalid choice) surface as exit
   * code 2, keeping exit 1 reserved for gate trips. Exit code 0 (--help,
   * --version) passes through unchanged.
   */
  for (const command of [
    program,
    compareCmd,
    measureCmd,
    startCmd,
    iterateCmd,
    keepCmd,
    discardCmd,
    statusCmd,
  ]) {
    command.exitOverride((err) => {
      throw new CommanderError(err.exitCode === 0 ? 0 : 2, err.code, err.message);
    });
  }

  return program;
}

/**
 * Resolve argv[1] to a canonical file URL, or `undefined` when it names nothing.
 *
 * Importing this module must never throw on the way in, and argv[1] is not
 * guaranteed to be a live path: `node -e` and the REPL leave it unset, and it
 * can point at a file that no longer exists. Neither case is this module being
 * the entry point, so both resolve to `undefined` rather than propagating.
 */
function resolveEntryUrl(entryPath: string | undefined): string | undefined {
  if (entryPath === undefined) {
    return undefined;
  }
  try {
    return pathToFileURL(realpathSync(entryPath)).href;
  } catch {
    return undefined;
  }
}

const isEntryPoint = resolveEntryUrl(process.argv[1]) === import.meta.url;

/* v8 ignore next 13 -- entry point. The "executes CLI when invoked through symlink"
   test in tests/cli.test.ts does run this block, but it spawns a child process, and
   in-process v8 coverage cannot attribute execution that happens outside the worker. */
if (isEntryPoint) {
  try {
    const program = createProgram();
    await program.parseAsync(process.argv);
    process.exit(0);
  } catch (error) {
    /* Commander already wrote help, the version, or the usage error to its stream. */
    if (error instanceof CommanderError) {
      process.exit(error.exitCode);
    }
    await exitWithError(error);
  }
}
