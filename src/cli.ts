#!/usr/bin/env node

import { readFileSync, realpathSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { createHelpConfig } from "@jeffzi/epaulettes";
import { Command, CommanderError, InvalidArgumentError, Option } from "commander";
import yoctoSpinner from "yocto-spinner";

import { AdapterError } from "./adapters/index.js";
import { compare } from "./compare.js";
import type { CompareOptions, ProgressStep, TargetSpec } from "./compare.js";
import {
  resolveBenchlessConfig,
  resolveConfig,
  type CliFlags,
  type ResolvedConfig,
} from "./config.js";
import { confirmAction } from "./confirm.js";
import { assertNever, GymratError, messageOf, type StrictOmit } from "./errors.js";
import { EtaTracker, formatEta } from "./eta.js";
import { NotAGitRepositoryError, runGit } from "./git.js";
import { finalizeSession } from "./loop/finalize.js";
import { iterateSession, LoopStopError } from "./loop/iterate.js";
import { discardSession, keepSession } from "./loop/settle.js";
import { startSession, type StartResult } from "./loop/start.js";
import { statusSession } from "./loop/status.js";
import { measure } from "./measure.js";
import type { MeasureOptions } from "./measure.js";
import { metricRecord } from "./metric-record.js";
import {
  countVerdicts,
  formatHintLabel,
  formatLabel,
  pluralize,
  shortenLabel,
} from "./report/format.js";
import { renderJson, renderMeasureJson } from "./report/json.js";
import { formatBaselineRef } from "./report/loop.js";
import { renderMeasureReport, renderReport } from "./report/text.js";
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
import { lockfilePath, repoRoot, superviseLockfilePath } from "./session/paths.js";
import type { BaselineRecord } from "./session/records.js";
import { appendRecord, requireOpenSession } from "./session/store.js";
import { ensureGitExclude } from "./session/workspace.js";
import { installTerminationCleanup } from "./signals.js";
import { createClaudeDriver } from "./supervisor/claude.js";
import type { LaunchEvent } from "./supervisor/events.js";
import { summarize } from "./supervisor/events.js";
import { composeKickoff } from "./supervisor/kickoff.js";
import { supervise } from "./supervisor/supervise.js";
import type { SupervisionResult } from "./supervisor/supervise.js";
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

function parsePositiveNumber(value: string): number {
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
function parseMaxMinutes(value: string): number {
  const parsed = parsePositiveNumber(value);
  const maxMinutes = Math.floor(MAX_TIMEOUT_SECONDS / 60);
  if (parsed > maxMinutes) {
    throw new InvalidArgumentError(`must be at most ${String(maxMinutes)} minutes.`);
  }
  return parsed;
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
async function warnEmptyGeomeanGates(
  conditions: readonly FailOnCondition[],
  result: ComparisonResult,
): Promise<void> {
  if (!conditions.some((condition) => condition.kind === "geomean")) return;

  for (const candidate of result.candidates) {
    if (gatedGeomeansOf(candidate).every((geomean) => geomean.n === 0)) {
      // Drained rather than fired and forgotten: a gate trip exits the process
      // right after this, and `process.exit` drops whatever is still buffered.
      await writeAndFlush(
        process.stderr,
        `warning: geomean gate for "${candidate.label}" had no stable gating metrics to measure\n`,
      );
    }
  }
}

/**
 * `@types/node` declares `isTTY` as `boolean`, but Node leaves it `undefined`
 * on non-TTY streams — the comparison is what makes the boolean honest.
 */
function isTTY(stream: { isTTY?: boolean }): boolean {
  return stream.isTTY === true;
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

  if (error instanceof GymratError && error.hint !== undefined) {
    output += `\n${formatHintLabel(process.stderr)} ${error.hint}`;
  }

  if (!(error instanceof GymratError)) {
    const debugCmd = formatLabel("gymrat --debug", "yellow", process.stderr);
    output += `\nRun with \`${debugCmd}\` for details. If this is a bug, please report it at\n${BUGS_URL}`;
  }

  return output;
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
function writeAndFlush(stream: NodeJS.WriteStream, data: string): Promise<void> {
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

/** The exit code of a gate trip: a run that did what it was asked and said no. */
const GATE_EXIT_CODE = 1;

/** The exit code of a tool failure, the convention every unhandled error exits on. */
const TOOL_FAILURE_EXIT_CODE = 2;

/** Whether the global `--debug` flag was passed on the command line. */
let debugMode = false;

/** Print a formatted error to stderr and exit on `code`. */
async function exitWithError(error: unknown, code = TOOL_FAILURE_EXIT_CODE): Promise<never> {
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

/** Shown after a sample step until enough gaps have been measured for an ETA. */
const ETA_PENDING_LABEL = "estimating time left…";

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
  let etaSuffix: string | undefined;
  if (etaMs !== undefined) {
    etaSuffix = formatEta(etaMs);
  } else if (step.kind === "sample") {
    etaSuffix = ETA_PENDING_LABEL;
  }

  const label = style.label(step.label);
  let line: string;
  switch (step.kind) {
    case "prepare":
      line = `prepare · ${label}`;
      break;
    case "sample":
      line = `sample ${style.counter(`${step.index}/${step.total}`)} · ${label}`;
      break;
    default:
      return assertNever(step);
  }

  return etaSuffix === undefined ? line : line + style.eta(` · ${etaSuffix}`);
}

/**
 * Each line names the target so the user can tell which one is running;
 * samples also show their position in the total.
 * When an ETA estimate is available, it is appended as plain text.
 */
function formatProgressLine(step: ProgressStep, etaMs?: number): string {
  return renderProgressLine(step, etaMs, {
    label: (text) => text,
    counter: (text) => text,
    eta: (text) => text,
  });
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

/**
 * Translate Commander's `--no-color` flag into the color override the report
 * renderer reads: `false` vetoes color, `undefined` defers to auto-detection.
 */
function colorOverrideOf(options: { color: boolean }): false | undefined {
  return options.color ? undefined : false;
}

/** The flags every command carries: everything `resolveConfig` reads, plus how to print. */
interface SharedFlags extends CliFlags {
  /** Commander's `--no-color` counterpart: true unless the flag was passed. */
  color: boolean;
  /** Output format — `text` produces the ANSI-styled table, `json` produces plain output. */
  format: "text" | "json";
}

/** The status command's flags: the shared set minus the format choice status does not offer. */
type StatusFlags = StrictOmit<SharedFlags, "format">;

/** The keep command's flags: the configuration set plus the message the commit carries. */
interface KeepFlags extends CliFlags {
  /** Commit message the agent supplied; absent, keep generates one from the iteration. */
  message?: string;
}

/**
 * The finalize command's flags — the only loop command whose flags are all its
 * own, because closing a session reads no configuration.
 */
interface FinalizeFlags {
  /** Message the squash commit carries; absent, finalize generates one from the kept commits. */
  message?: string;
  /** Branch to point at the squash commit; absent, the session branch's `-final`. */
  branch?: string;
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

/** The supervise command's flags: session caps plus where to write the event log. */
interface SuperviseFlags {
  maxMinutes: number;
  maxUsd?: number;
  /** Path to the JSONL event log; absent, defaults to `.gymrat/supervisor-<timestamp>.jsonl`. */
  log?: string;
  model?: string;
  allowDirty: boolean;
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
function addSharedOptions(command: Command): Command {
  return addConfigOptions(command)
    .option("--no-color", "print the report without ANSI styles")
    .addOption(
      new Option("--format <value>", "output format").choices(["text", "json"]).default("text"),
    );
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
  const tty = isTTY(process.stderr);
  const colorSuppressed = process.env.NO_COLOR !== undefined;
  return createProgressReporter(tty && !colorSuppressed, tty, targetCount);
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
    return await execute();
  } finally {
    progress.stop();
  }
}

function getDirtyFileCount(root: string): number {
  const output = runGit(["status", "--porcelain", "--untracked-files=all"], root).trim();
  if (output === "") return 0;
  return output.split("\n").length;
}

function resolveLogPath(root: string, explicitPath: string | undefined): string {
  if (explicitPath !== undefined) return explicitPath;
  ensureGitExclude(root);
  return join(root, ".gymrat", `supervisor-${Date.now()}.jsonl`);
}

/**
 * The repository a run must serialize against, or `undefined` when there is none.
 *
 * Both commands accept plain directories, so standing outside a git repository
 * is a supported way to run gymrat rather than a failure: it just leaves nothing
 * to take a lock on. Git declining to answer is not that answer — an untrusted
 * or unreadable repository still holds a repository, and a run that read the
 * failure as "no repository here" would bench it unlocked alongside whatever
 * else is running there. That error propagates instead.
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

/**
 * Run `execute` with an abort signal a termination signal trips, then exit with
 * the conventional `128 + signum`.
 *
 * The bench runs in a detached process group, so gymrat's own Ctrl-C never
 * reaches it: without this, a killed `iterate` leaves a bench group running in —
 * and writing into — the session's worktrees. Aborting is the whole cleanup.
 * Those worktrees belong to the session and outlive any single run, unlike the
 * throwaway ones a comparison creates and sweeps on its way out.
 */
async function runInterruptibly<T>(execute: (signal: AbortSignal) => Promise<T>): Promise<T> {
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

  await writeAndFlush(process.stdout, output + "\n");
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
 * of them and must never touch the other. A closed session that was moved aside
 * is named too: the log it left is still on disk, and this is the only place the
 * agent is told so.
 */
function formatStartSummary(result: StartResult, runbook?: string): string {
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
    ...(runbook === undefined ? [] : [`  runbook: ${runbook} — read it before your first edit`]),
    ...(result.archivedPath === undefined
      ? []
      : [`  archived the finalized session ${result.archived} to ${result.archivedPath}`]),
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

const DOCS_URL = "https://github.com/jeffzi/gymrat#readme";
const BUGS_URL = "https://github.com/jeffzi/gymrat/issues";

const ROOT_EPILOGUE = `
Examples:

  • gymrat compare main my-branch --bench "npm run bench"

  • gymrat compare old=main new=perf/decode --bench "npm run bench" --fail-on regressed

  • gymrat measure --bench "npm run bench"

Docs: ${DOCS_URL}
Bugs: ${BUGS_URL}
`;

const COMPARE_EPILOGUE = `
Examples:

  • gymrat compare main perf/faster-decode --bench "npm run bench"

  • gymrat compare main perf/simd perf/lookup-table --bench "npm run bench"

  • gymrat compare old=main new=perf/faster-decode --bench "npm run bench"

  • gymrat compare main my-branch --bench "npm run bench" --fail-on geomean:2 --format json
`;

const MEASURE_EPILOGUE = `
Examples:

  • gymrat measure --bench "npm run bench"

  • gymrat measure release=v2.0.0 --bench "npm run bench" --adapter mitata

  • gymrat measure --bench "npm run bench" --record
`;

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
    .version(readPackageVersion())
    .option("-d, --debug", "show stack traces on errors", false)
    .configureHelp(createHelpConfig())
    .exitOverride((err) => {
      throw new CommanderError(
        err.exitCode === 0 ? 0 : TOOL_FAILURE_EXIT_CODE,
        err.code,
        err.message,
      );
    })
    .hook("preSubcommand", (thisCommand) => {
      debugMode = thisCommand.opts<{ debug: boolean }>().debug;
    });

  addSharedOptions(
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
    .action(async (baseline: TargetSpec, candidates: TargetSpec[], options: CompareFlags) => {
      const colorOverride = colorOverrideOf(options);

      /*
       * Everything after the lock only reads the result the lock already
       * protected: the report describes a finished measurement, so holding the
       * repository against every other gymrat process while stdout drains would
       * serialize runs for nothing. The gate check has a second reason to sit
       * here — process.exit must not be reachable from inside withRepoLock's own
       * error handling.
       */
      const result = await withRepoLock("compare", async () => {
        const progress = beginRun(options, 1 + candidates.length);

        return runGuarded(progress, async () => {
          const config = resolveConfig(options);

          const compareOptions: CompareOptions = {
            baseline,
            candidates,
            ...runOptionsOf(config, progress),
            unstableNoisePct: config.unstableNoisePct,
          };

          return compare(compareOptions);
        });
      });

      await emitReport(
        result,
        options,
        { json: renderJson, text: renderReport },
        {
          verbose: options.verbose,
          color: colorOverride,
          failOn: options.failOn,
        },
      );

      await warnEmptyGeomeanGates(options.failOn, result);

      if (shouldFailGate(options.failOn, result)) {
        process.exit(GATE_EXIT_CODE);
      }
    });

  addSharedOptions(
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
    .action(async (target: TargetSpec, options: MeasureFlags) => {
      const colorOverride = colorOverrideOf(options);

      /*
       * The lock covers the measurement and the append it produces — every
       * repository write — and nothing beyond. The report describes a run that
       * is already over, so holding the repository while stdout drains would
       * serialize runs for nothing.
       */
      const run = await withRepoLock("measure", async () => {
        // One target, so every sample step the run reports is a step of the whole run.
        const progress = beginRun(options, 1);

        const measured = await runGuarded(progress, async () => {
          const config = resolveConfig(options);

          /*
           * The session is resolved before a single sample is taken: a run that
           * benches for ten minutes and only then discovers it has nowhere to
           * write has thrown the whole measurement away.
           */
          const recording = options.record
            ? requireOpenSession(repoRoot(), "recording a measurement")
            : undefined;

          const measureOptions: MeasureOptions = {
            target,
            ...runOptionsOf(config, progress),
          };

          return { result: await measure(measureOptions), recording };
        });

        if (measured.recording !== undefined) {
          appendRecord(measured.recording.jsonlPath, baselineRecordOf(measured.result));
        }

        return measured;
      });

      await emitReport(
        run.result,
        options,
        { json: renderMeasureJson, text: renderMeasureReport },
        { color: colorOverride },
      );

      if (run.recording !== undefined) {
        // The note is prose, so it goes to stderr whenever stdout is carrying
        // JSON a consumer has to parse.
        await writeAndFlush(
          options.format === "json" ? process.stderr : process.stdout,
          `baseline "${run.result.label}" recorded to session ${run.recording.session.sessionId}\n`,
        );
      }
    });

  addConfigOptions(
    program
      .command("start")
      .description("Create or resume this repository's optimization session")
      .argument("[ref]", "ref the baseline is pinned to; defaults to HEAD"),
  ).action(async (ref: string | undefined, options: CliFlags) => {
    /*
     * The summary prints after the lock is released: it describes a workspace
     * that is already on disk, so holding the repository against every other
     * gymrat process while stdout drains would serialize runs for nothing.
     */
    const started = await withRepoLock("start", () => {
      const root = repoRoot();
      const config = resolveConfig(options, root);
      return Promise.resolve({
        result: startSession(root, ref, config),
        runbook: config.runbook,
      });
    });

    await writeAndFlush(process.stdout, `${formatStartSummary(started.result, started.runbook)}\n`);
  });

  addConfigOptions(
    program
      .command("iterate")
      .description("Measure the session's experiment worktree against its baseline"),
  ).action(async (options: CliFlags) => {
    /*
     * The report prints after the lock is released, as `start`'s summary does:
     * the measurement is over by then, and holding the repository while stdout
     * drains would serialize runs for nothing.
     */
    const result = await withRepoLock(
      "iterate",
      async () => {
        const root = repoRoot();
        return runInterruptibly((signal) =>
          iterateSession(root, resolveConfig(options, root), { signal }),
        );
      },
      /*
       * A met stop condition is a gate trip, not a tool failure: the loop
       * reached the end it was configured for, so it exits the way a keep
       * the checks refused does.
       */
      (error) => (error instanceof LoopStopError ? GATE_EXIT_CODE : TOOL_FAILURE_EXIT_CODE),
    );

    await writeAndFlush(process.stdout, `${result.report}\n`);
  });

  addConfigOptions(
    program
      .command("keep")
      .description("Commit the session's measured edit once its checks pass")
      .option("-m, --message <text>", "commit message for the kept edit"),
  ).action(async (options: KeepFlags) => {
    const result = await withRepoLock("keep", async () => {
      const root = repoRoot();
      return keepSession(root, resolveBenchlessConfig(options, root), {
        message: options.message,
      });
    });

    await writeAndFlush(process.stdout, `${result.report}\n`);

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
  program
    .command("discard")
    .description("Revert the session's experiment worktree to its last commit")
    .option("-f, --force", "skip the confirmation prompt")
    .action(async (options: { force?: boolean }) => {
      const root = repoRoot();

      if (isTTY(process.stdin) && options.force !== true) {
        const { session } = requireOpenSession(root, "discard");
        const confirmed = await confirmAction(
          `discard will revert uncommitted changes in ${session.worktrees.experiment}.\nProceed?`,
          process.stdin,
        );
        if (!confirmed) {
          await writeAndFlush(process.stderr, "discard cancelled\n");
          process.exit(GATE_EXIT_CODE);
        }
      }

      const result = await withRepoLock("discard", () => Promise.resolve(discardSession(root)));

      await writeAndFlush(process.stdout, `${result.report}\n`);
    });

  /*
   * Reads no configuration, for `discard`'s reason: a squash is git alone, so
   * `finalize` carries none of the config flags its siblings do.
   */
  program
    .command("finalize")
    .description("Collapse the session's kept iterations into one commit and close it")
    .option("-m, --message <text>", "message for the squash commit")
    .option("--branch <name>", "branch to point at the squash commit (default: <branch>-final)")
    .action(async (options: FinalizeFlags) => {
      const result = await withRepoLock("finalize", () =>
        Promise.resolve(
          finalizeSession(repoRoot(), { message: options.message, branch: options.branch }),
        ),
      );

      await writeAndFlush(process.stdout, `${result.report}\n`);
    });

  /*
   * The one loop command that takes no repository lock: it reads the session
   * log and writes nothing, so serializing it against a running iterate would
   * only make the agent wait to be told what is already on disk.
   */
  addConfigOptions(
    program
      .command("status")
      .description("Show this repository's session history, read from its log")
      .option("--no-color", "print the report without ANSI styles"),
  ).action(async (options: StatusFlags) => {
    // Suppressed before rendering, not after: the lines style themselves as
    // they are built, and `styleText` reads the environment on every call.
    if (!options.color) {
      suppressColor();
    }

    const report = await runOrExit(() => {
      const root = repoRoot();
      return Promise.resolve(statusSession(root, resolveBenchlessConfig(options, root)));
    });

    await writeAndFlush(process.stdout, `${report}\n`);
  });

  // ---------------------------------------------------------------------------
  // supervise helpers
  // ---------------------------------------------------------------------------

  async function validateWorkingTree(root: string, allowDirty: boolean): Promise<number> {
    const dirtyFileCount = getDirtyFileCount(root);
    if (dirtyFileCount > 0 && !allowDirty) {
      await exitWithError(
        new GymratError(
          `Working tree has ${pluralize(dirtyFileCount, "uncommitted file")}.`,
          "Commit or stash your changes, or pass --allow-dirty to proceed anyway.",
        ),
      );
    }
    if (dirtyFileCount > 0) {
      await writeAndFlush(
        process.stderr,
        `warning: working tree has ${pluralize(dirtyFileCount, "dirty file")} — proceeding because --allow-dirty was set\n`,
      );
    }
    return dirtyFileCount;
  }

  async function reportSupervisionResult(
    result: SupervisionResult,
    logPath: string,
  ): Promise<void> {
    const durationSec = Math.round(result.durationMs / 1000);
    const minutes = Math.floor(durationSec / 60);
    const seconds = durationSec % 60;
    const summary = [
      `outcome: ${result.outcome.reason}`,
      `ended by: ${result.endedBy}`,
      `duration: ${String(minutes)}m ${String(seconds)}s`,
      `cost: $${result.costUsd.toFixed(2)}`,
      `log: ${logPath}`,
    ].join("\n");

    await writeAndFlush(process.stdout, `${summary}\n`);

    if (result.outcome.reason === "error") {
      if (result.outcome.message) {
        await exitWithError(new GymratError(result.outcome.message));
      }
      process.exit(TOOL_FAILURE_EXIT_CODE);
    }

    if (result.endedBy !== "session") {
      process.exit(GATE_EXIT_CODE);
    }
  }

  // ---------------------------------------------------------------------------
  // supervise
  // ---------------------------------------------------------------------------

  program
    .command("supervise")
    .description("Run a supervised agent session with wall-clock and spend caps")
    .argument("[prompt]", "optimization prompt for the agent")
    .requiredOption("--max-minutes <number>", "wall-clock cap in minutes", parseMaxMinutes)
    .option("--max-usd <number>", "spend cap in USD", parsePositiveNumber)
    .option("--log <path>", "path for the JSONL event log")
    .option("--model <name>", "model to use for the agent session")
    .option("--allow-dirty", "allow launching with uncommitted changes", false)
    .action(async (prompt: string | undefined, options: SuperviseFlags) => {
      const root = await runOrExit(() => Promise.resolve(repoRoot()));
      const dirtyFileCount = await validateWorkingTree(root, options.allowDirty);

      const release = await runOrExit(() =>
        Promise.resolve(acquireLock(superviseLockfilePath(root), "supervise")),
      );

      process.once("exit", release);

      try {
        const logPath = resolveLogPath(root, options.log);

        const config = resolveBenchlessConfig({}, root);
        const kickoff = composeKickoff(config, import.meta.url, prompt);

        const headSha = runGit(["rev-parse", "HEAD"], root).trim();

        const launch: LaunchEvent = {
          type: "launch",
          timestamp: Date.now(),
          headSha,
          dirty: dirtyFileCount > 0 ? { fileCount: dirtyFileCount } : false,
          maxMinutes: options.maxMinutes,
          maxUsd: options.maxUsd,
          model: options.model,
          runbookPath: config.runbook ?? "",
          kickoffSummary: summarize(kickoff.kickoff),
        };

        const driver = createClaudeDriver();

        await writeAndFlush(process.stderr, `log: ${logPath}\n`);

        const result: SupervisionResult = await runOrExit(() =>
          supervise({
            driver,
            prompt: {
              kickoff: kickoff.kickoff,
              systemPromptAppend: kickoff.systemPromptAppend,
              cwd: root,
              model: options.model,
            },
            maxMinutes: options.maxMinutes,
            maxUsd: options.maxUsd,
            logPath,
            launch,
          }),
        );

        await reportSupervisionResult(result, logPath);
      } finally {
        process.removeListener("exit", release);
        release();
      }
    });

  program.addHelpText("after", ROOT_EPILOGUE);
  const subcommandEpilogues: Record<string, string> = {
    compare: COMPARE_EPILOGUE,
    measure: MEASURE_EPILOGUE,
  };
  for (const sub of program.commands) {
    const epilogue = subcommandEpilogues[sub.name()];
    if (epilogue !== undefined) {
      sub.addHelpText("after", epilogue);
    }
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

/* istanbul ignore next -- entry point. The "executes CLI when invoked through symlink"
   test in tests/cli.test.ts does run this block, but it spawns a child process, and
   in-process coverage cannot attribute execution that happens outside the worker. */
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
