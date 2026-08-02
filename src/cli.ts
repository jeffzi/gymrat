#!/usr/bin/env node

import { readFileSync, realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { createHelpConfig } from "@jeffzi/epaulettes";
import { Command, CommanderError, InvalidArgumentError, Option } from "commander";
import yoctoSpinner from "yocto-spinner";

import { AdapterError } from "./adapters/index.js";
import { compare } from "./compare.js";
import type { CompareOptions, ProgressStep, TargetSpec } from "./compare.js";
import { resolveConfig, type CliFlags } from "./config.js";
import { assertNever, GymratError } from "./errors.js";
import { EtaTracker, formatEta } from "./eta.js";
import { countVerdicts, formatHintLabel, formatLabel } from "./report/format.js";
import { renderJson } from "./report/json.js";
import { renderMarkdown } from "./report/markdown.js";
import { renderReport } from "./report/text.js";
import type { ComparisonResult, MetricComparisons } from "./report/types.js";

/**
 * Parse the `label=target` syntax of a positional argument, baseline or candidate.
 *
 * Only the first `=` splits, so a target containing its own `=` survives intact —
 * `a=b=c` parses to label `a`, target `b=c`.
 */
function parsePositional(positional: string): TargetSpec {
  const eqIndex = positional.indexOf("=");
  if (eqIndex === -1) {
    return { label: undefined, target: positional };
  }
  return { label: positional.slice(0, eqIndex), target: positional.slice(eqIndex + 1) };
}

/**
 * Coerce a numeric flag value to a positive integer, rejecting anything else.
 *
 * `parseInt` is too permissive for flag values: it returns `NaN` for
 * non-numeric input and silently truncates trailing garbage (`"10abc"` → `10`).
 * `NaN` is not nullish, so it survives the `??` default chain in
 * `resolveConfig` and only surfaces much later as a confusing benchmark error.
 *
 * Throwing `InvalidArgumentError` lets Commander render a usage error naming
 * the offending flag and value.
 */
function parsePositiveInteger(value: string): number {
  const parsed = Number(value);
  if (!/^\d+$/.test(value) || parsed <= 0) {
    throw new InvalidArgumentError("must be a positive integer.");
  }
  return parsed;
}

/** A parsed `--fail-on` condition: either a `regressed` check or a geomean threshold. */
type FailOnCondition = { kind: "regressed" } | { kind: "geomean"; pct: number };

/**
 * Accepts `regressed` or `geomean:<number>`. Throws `InvalidArgumentError`
 * for anything else so Commander renders the allowed grammar in the usage error.
 */
function parseFailOn(value: string, previous: FailOnCondition[]): FailOnCondition[] {
  if (value === "regressed") {
    previous.push({ kind: "regressed" });
    return previous;
  }

  const match = /^geomean:(.+)$/.exec(value);
  const pct = match ? Number(match[1]) : NaN;
  if (Number.isFinite(pct)) {
    previous.push({ kind: "geomean", pct });
    return previous;
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
  return Object.fromEntries(Object.entries(metrics).filter(([, metric]) => metric.meta.gating));
}

/** Returns `true` when any condition trips — meaning the process should exit 1. */
function shouldFailGate(conditions: readonly FailOnCondition[], result: ComparisonResult): boolean {
  if (conditions.length === 0) return false;

  const gating = gatingMetrics(result.metrics);

  return conditions.some((condition) => {
    switch (condition.kind) {
      case "regressed":
        return result.candidates.some((_, i) => countVerdicts(gating, i).regressed > 0);
      case "geomean":
        return result.candidates.some((candidate) => candidate.geomean.value > condition.pct);
      default:
        return assertNever(condition);
    }
  });
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

  let output = error instanceof Error ? error.message : String(error);

  if (error instanceof GymratError && error.hint !== undefined) {
    output += `\n${formatHintLabel(process.stderr)} ${error.hint}`;
  }

  return output;
}

/** Print a formatted error to stderr and exit 2, the convention for tool failures. */
function exitWithError(error: unknown): never {
  process.stderr.write(`${formatCliError(error)}\n`);
  process.exit(2);
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

/** Join a step word with its optional counter and label, `·`-separated. */
function joinStepLine(stepWord: string, counter: string | undefined, label: string): string {
  return counter === undefined ? `${stepWord} · ${label}` : `${stepWord} ${counter} · ${label}`;
}

/**
 * Each line names the target so the user can tell which one is running;
 * samples also show their position in the total.
 * When an ETA estimate is available, it is appended as plain text.
 */
function formatProgressLine(step: ProgressStep, etaMs?: number): string {
  const { stepWord, counter, label, etaSuffix } = buildProgressLineParts(step, etaMs);
  let line = joinStepLine(stepWord, counter, label);
  if (etaSuffix !== undefined) {
    line += ` · ${etaSuffix}`;
  }
  return line;
}

/**
 * Apply ANSI styling to the progress line for spinner display:
 * step word in default foreground, counter bold (sample only), label cyan.
 * When an ETA estimate is available, it is appended as a dim segment.
 * When no ETA is available for a sample step, a dim placeholder is shown.
 */
function styleProgressLine(step: ProgressStep, etaMs?: number): string {
  const { stepWord, counter, label, etaSuffix } = buildProgressLineParts(step, etaMs);
  const styledLabel = formatLabel(label, "cyan", process.stderr);
  const styledCounter =
    counter === undefined ? undefined : formatLabel(counter, "bold", process.stderr);
  let line = joinStepLine(stepWord, styledCounter, styledLabel);
  if (etaSuffix !== undefined) {
    line += formatLabel(` · ${etaSuffix}`, "dim", process.stderr);
  }
  return line;
}

/** Carriage-return + clear-to-EOL: rewinds the cursor and erases the line it lands on. */
const CLEAR_LINE = "\r\x1b[K";

/**
 * Non-TTY output must stay free of ANSI escapes; TTY output overwrites in
 * place using {@link CLEAR_LINE} so only one progress line is ever visible.
 */
function writeProgress(line: string, tty: boolean): void {
  process.stderr.write(tty ? `${CLEAR_LINE}${line}` : `${line}\n`);
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

/** Single-use: `stop()` must be called exactly once, after the run completes or fails. */
interface ProgressReporter {
  emit(step: ProgressStep): void;
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
    stop(): void {
      clearCountdown();
      if (spinner) {
        spinner.stop();
      } else {
        clearProgress(tty);
      }
    },
  };
}

/** The compare command's flags: everything `resolveConfig` reads, plus how to print. */
interface CompareFlags extends CliFlags {
  /** Commander's `--no-color` counterpart: true unless the flag was passed. */
  color: boolean;
  /** Output format — `text` produces the ANSI-styled table, others produce plain output. */
  format: "text" | "markdown" | "json";
  /** Gate conditions that cause exit 1 when any trips. Empty when `--fail-on` is not used. */
  failOn: FailOnCondition[];
  /** Name the statistical method behind each verdict in the report footer. */
  verbose: boolean;
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
    throw new Error("package.json has no string version field");
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
  const program = new Command();

  program
    .name("gymrat")
    .description("Performance comparison tool for benchmarks")
    .version(readPackageVersion());

  const compareCmd = program
    .command("compare")
    .description("Compare one baseline revision against one or more candidates")
    .argument("<baseline>", "[label=]<ref|dir> to measure against")
    .argument("<candidates...>", "[label=]<ref|dir>, each judged against the baseline")
    .option("--bench <cmd>", "bench command")
    .option("--prepare <script>", "preparation script to run before each revision")
    .option("--adapter <type>", "adapter type for parsing benchmark output")
    .option("--samples <number>", "paired samples per target", parsePositiveInteger)
    .option("--timeout <number>", "timeout in seconds", parsePositiveInteger)
    .option("--config <file>", "configuration file path")
    .option("--no-color", "print the report without ANSI styles")
    .option("--verbose", "name the statistical method behind each verdict", false)
    .addOption(
      new Option("--format <value>", "output format")
        .choices(["text", "markdown", "json"])
        .default("text"),
    )
    .option(
      "--fail-on <condition>",
      'exit 1 when a condition trips (repeatable: "regressed", "geomean:<pct>")',
      parseFailOn,
      [] as FailOnCondition[],
    )
    .configureHelp(createHelpConfig())
    .action(async (baselineArg: string, candidateArgs: string[], options: CompareFlags) => {
      let result: ComparisonResult;

      if (!options.color) {
        suppressColor();
      }

      const tty = isTerminal(process.stderr);
      const colorSuppressed = process.env.NO_COLOR !== undefined;
      const targetCount = 1 + candidateArgs.length;
      const progress = createProgressReporter(tty && !colorSuppressed, tty, targetCount);

      try {
        const config = resolveConfig({
          bench: options.bench,
          prepare: options.prepare,
          adapter: options.adapter,
          samples: options.samples,
          timeout: options.timeout,
          config: options.config,
        });

        const compareOptions: CompareOptions = {
          baseline: parsePositional(baselineArg),
          candidates: candidateArgs.map((arg) => parsePositional(arg)),
          bench: config.bench,
          prepare: config.prepare,
          adapter: config.adapter,
          samples: config.samples,
          timeoutSeconds: config.timeoutSeconds,
          unstableNoisePct: config.unstableNoisePct,
          configMetrics: config.metrics,
          onProgress: (step) => {
            progress.emit(step);
          },
        };

        result = await compare(compareOptions);
        progress.stop();
      } catch (error) {
        progress.stop();
        exitWithError(error);
      }

      // `--no-color` is a veto, never a force: left unset, each renderer keeps
      // its own detection rather than being told what the stream supports.
      const color = options.color ? undefined : false;

      let output: string;
      switch (options.format) {
        case "markdown":
          output = renderMarkdown(result, { verbose: options.verbose, color });
          break;
        case "json":
          // Machine-readable output stays byte-identical whatever --verbose says.
          output = renderJson(result);
          break;
        case "text":
          output = renderReport(result, { verbose: options.verbose, color });
          break;
        default:
          assertNever(options.format);
      }

      process.stdout.write(output + "\n");

      if (shouldFailGate(options.failOn, result)) {
        process.exit(1);
      }
    });

  /*
   * Commander exits 1 for usage errors by default. Override so all Commander
   * errors (unknown option, missing argument, invalid choice) surface as exit
   * code 2, keeping exit 1 reserved for gate trips. Exit code 0 (--help,
   * --version) passes through unchanged.
   */
  for (const command of [program, compareCmd]) {
    command.exitOverride((err) => {
      throw new CommanderError(err.exitCode === 0 ? 0 : 2, err.code, err.message);
    });
  }

  return program;
}

/* v8 ignore next 20 -- entry point. The "executes CLI when invoked through symlink"
   test in tests/cli.test.ts does run this block, but it spawns a child process, and
   in-process v8 coverage cannot attribute execution that happens outside the worker. */
if (import.meta.url === pathToFileURL(realpathSync(process.argv[1]!)).href) {
  try {
    const program = createProgram();
    await program.parseAsync(process.argv);
    process.exit(0);
  } catch (error) {
    /* --help and --version throw CommanderError with exitCode 0 after output */
    if (error instanceof CommanderError && error.exitCode === 0) {
      process.exit(0);
    }
    exitWithError(error);
  }
}
