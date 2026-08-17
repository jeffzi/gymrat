import { createInterface } from "node:readline/promises";
import type { Readable, Writable } from "node:stream";

import { adapterNames, getAdapter } from "../adapters/index.js";
import { CONFIG_DEFAULTS, GEOMEAN_PRIMARY } from "../config.js";
import { GymratError } from "../errors.js";

/** The runbook filename scaffolded when the wizard creates one without a user-supplied path. */
export const DEFAULT_RUNBOOK_PATH = "gymrat-runbook.md";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Pre-filled answers and I/O streams for the init wizard.
 *
 * Any field left `undefined` is prompted for interactively (unless `yes` is set, in which case
 * a default is applied instead). `runbook` is tri-state: `false` skips the runbook, `true` writes
 * it to {@link DEFAULT_RUNBOOK_PATH}, and a string writes it to that path.
 *
 * `input`/`output` are the streams the wizard reads prompts from and writes to — pass
 * `process.stdin`/`process.stdout` in normal operation, or in-memory streams in tests.
 */
export interface WizardOptions {
  bench?: string | undefined;
  adapter?: string | undefined;
  checks?: string | undefined;
  stopTarget?: number | undefined;
  stopMaxIterations?: number | undefined;
  primary?: string | undefined;
  runbook?: string | boolean | undefined;
  skill?: boolean | undefined;
  yes?: boolean | undefined;
  input: Readable;
  output: Writable;
}

/**
 * The settled wizard output, consumed by {@link scaffold} to write config and runbook files.
 *
 * Unlike {@link WizardOptions}, every field here is resolved: `runbook` is either `false` (skip)
 * or the concrete `{ path }` to write, with no remaining tri-state ambiguity.
 */
export interface WizardResult {
  bench: string;
  adapter?: string;
  checks?: string;
  stopTarget?: number;
  stopMaxIterations?: number;
  primary?: string;
  runbook: { path: string } | false;
  installSkill: boolean;
}

// ---------------------------------------------------------------------------
// Line reader — buffers lines from readline so none are lost between prompts
// ---------------------------------------------------------------------------

interface LineReader {
  nextLine(this: void): Promise<string | undefined>;
  close(): void;
  /** True when the stream has closed AND no buffered lines remain. */
  readonly atEof: boolean;
}

/**
 * Wrap a readable stream in a line-buffered reader.
 *
 * Node's readline emits all available 'line' events in one microtask batch
 * when data is pre-buffered (common in tests with PassThrough streams).
 * Sequential `rl.question()` calls miss lines that fire between awaits. This
 * reader captures every 'line' event eagerly and delivers them one at a time
 * via {@link LineReader.nextLine}.
 */
function createLineReader(input: Readable): LineReader {
  const rl = createInterface({ input, terminal: false });

  const buffer: string[] = [];
  const waiters: Array<(line: string | undefined) => void> = [];
  let closed = false;

  rl.on("line", (line: string) => {
    const waiter = waiters.shift();
    if (waiter) {
      waiter(line);
    } else {
      buffer.push(line);
    }
  });

  rl.once("close", () => {
    closed = true;
    for (const waiter of waiters) waiter(undefined);
    waiters.length = 0;
  });

  return {
    nextLine(): Promise<string | undefined> {
      const buffered = buffer.shift();
      if (buffered !== undefined) return Promise.resolve(buffered);
      if (closed) return Promise.resolve(undefined);
      return new Promise<string | undefined>((resolve) => {
        waiters.push(resolve);
      });
    },
    close() {
      rl.close();
    },
    get atEof() {
      return closed && buffer.length === 0;
    },
  };
}

// ---------------------------------------------------------------------------
// askPrompt — core prompt loop over a line reader
// ---------------------------------------------------------------------------

interface AskInternalOptions {
  default?: string;
  validate?: (value: string) => string | undefined;
}

/**
 * Write a prompt to `output`, read the next line via `nextLine`, and
 * validate. Reprompts on invalid input. Returns the default (or `undefined`)
 * on empty input or EOF.
 */
async function askPrompt(
  question: string,
  output: Writable,
  nextLine: () => Promise<string | undefined>,
  opts?: AskInternalOptions,
): Promise<string | undefined> {
  const defaultSuffix = opts?.default !== undefined ? ` [${opts.default}]` : "";
  const prompt = `${question}${defaultSuffix} `;

  for (;;) {
    output.write(prompt);
    const answer = await nextLine();

    if (answer === undefined || answer === "") {
      return opts?.default;
    }

    if (opts?.validate) {
      const error = opts.validate(answer);
      if (error !== undefined) {
        output.write(`${error}\n`);
        continue;
      }
    }

    return answer;
  }
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

function validateAdapter(name: string): string | undefined {
  try {
    getAdapter(name);
    return undefined;
  } catch (err) {
    if (!(err instanceof GymratError)) throw err;
    return [err.message, err.hint].filter(Boolean).join(" ");
  }
}

function validateStopTarget(value: string): string | undefined {
  return Number.isFinite(Number(value)) ? undefined : "Must be a finite number.";
}

function validateMaxIterations(value: string): string | undefined {
  const n = Number(value);
  return !Number.isInteger(n) || n < 1 ? "Must be an integer >= 1." : undefined;
}

function validatePrimary(value: string): string | undefined {
  return value === GEOMEAN_PRIMARY ? `Primary metric cannot be "${GEOMEAN_PRIMARY}".` : undefined;
}

// ---------------------------------------------------------------------------
// Settlement helpers
// ---------------------------------------------------------------------------

function isInteractive(options: WizardOptions): boolean {
  if (options.yes) return false;
  return "isTTY" in options.input && options.input.isTTY === true;
}

function settleRunbookFromFlag(options: WizardOptions): { path: string } | false | undefined {
  if (options.runbook === false) return false;
  if (typeof options.runbook === "string") return { path: options.runbook };
  if (options.runbook === true) return { path: DEFAULT_RUNBOOK_PATH };
  return undefined;
}

// ---------------------------------------------------------------------------
// Per-field settlement
// ---------------------------------------------------------------------------

type PromptFn = (question: string, opts?: AskInternalOptions) => Promise<string | undefined>;

interface SettleContext {
  options: WizardOptions;
  interactive: boolean;
  advancedGate: boolean;
  prompt: PromptFn;
  /** True when the input stream has closed (EOF). */
  inputClosed(): boolean;
}

/**
 * Prompt on a loop until an answer is given or the input stream reaches EOF.
 */
async function promptUntilAnswered(
  ctx: SettleContext,
  question: string,
  opts?: AskInternalOptions,
): Promise<string | undefined> {
  let answer: string | undefined;
  do {
    answer = await ctx.prompt(question, opts);
  } while (!answer && !ctx.inputClosed());
  return answer;
}

/** Flag → interactive reprompt → throw. */
async function settleBench(ctx: SettleContext): Promise<string> {
  let bench = ctx.options.bench;
  if (!bench && ctx.interactive) {
    bench = await promptUntilAnswered(ctx, "Bench command:");
  }
  if (!bench) throw new GymratError("Missing --bench flag.");
  return bench;
}

async function settleAdvancedGate(ctx: SettleContext): Promise<boolean> {
  if (!ctx.interactive) return false;
  const answer = await ctx.prompt("Configure advanced settings? (y/N)");
  return answer === "y" || answer === "Y";
}

/** Flag (validated) → interactive prompt → omit. */
async function settleAdapter(ctx: SettleContext): Promise<string | undefined> {
  let adapter = ctx.options.adapter;
  if (adapter !== undefined) {
    getAdapter(adapter);
  } else if (ctx.advancedGate) {
    adapter = await ctx.prompt(`Adapter (${adapterNames.join(", ")}):`, {
      default: CONFIG_DEFAULTS.adapter,
      validate: validateAdapter,
    });
  }
  return adapter;
}

/** Flag → interactive prompt → omit. */
async function settleChecks(ctx: SettleContext): Promise<string | undefined> {
  return (
    ctx.options.checks ??
    (ctx.advancedGate ? await ctx.prompt("Checks command (optional):") : undefined)
  );
}

/** Flag (validated) → interactive prompt → omit. */
async function settleStopTarget(ctx: SettleContext): Promise<number | undefined> {
  if (ctx.options.stopTarget !== undefined) {
    if (Number.isNaN(ctx.options.stopTarget)) {
      throw new GymratError("Invalid --stop-target: not a number.");
    }
    return ctx.options.stopTarget;
  }
  if (ctx.advancedGate) {
    const { output } = ctx.options;
    output.write("Stop the loop when the primary metric reaches this threshold.\n");
    const raw = await ctx.prompt("Stop target (optional):", {
      validate: validateStopTarget,
    });
    if (raw) return Number(raw);
  }
  return undefined;
}

/**
 * Required when a stop target is set.
 * Flag → interactive reprompt → throw.
 */
async function settlePrimary(
  ctx: SettleContext,
  stopTarget: number | undefined,
): Promise<string | undefined> {
  let primary = ctx.options.primary;
  if (stopTarget !== undefined && !primary) {
    if (ctx.advancedGate) {
      primary = await promptUntilAnswered(ctx, "Primary metric:", { validate: validatePrimary });
    }
    if (!primary) {
      throw new GymratError(
        "Missing --primary flag.",
        "--stop-target requires --primary to name a metric.",
      );
    }
  }
  return primary;
}

/** Flag (validated) → interactive prompt → omit. */
async function settleMaxIterations(ctx: SettleContext): Promise<number | undefined> {
  if (ctx.options.stopMaxIterations !== undefined) {
    const n = ctx.options.stopMaxIterations;
    if (!Number.isInteger(n) || n < 1) {
      throw new GymratError("Invalid --stop-max-iterations: must be an integer >= 1.");
    }
    return n;
  }
  if (ctx.advancedGate) {
    const raw = await ctx.prompt("Max iterations (optional):", { validate: validateMaxIterations });
    if (raw) return Number(raw);
  }
  return undefined;
}

/** Flag → interactive prompt → default (create with default path). */
async function settleRunbook(ctx: SettleContext): Promise<{ path: string } | false> {
  const flagRunbook = settleRunbookFromFlag(ctx.options);
  if (flagRunbook !== undefined) return flagRunbook;
  if (ctx.interactive) {
    const createAnswer = await ctx.prompt("Create runbook? (y/N)");
    if (createAnswer === "y" || createAnswer === "Y") {
      const path = await ctx.prompt("Runbook path:", {
        default: DEFAULT_RUNBOOK_PATH,
      });
      return { path: path ?? DEFAULT_RUNBOOK_PATH };
    }
    return false;
  }
  return { path: DEFAULT_RUNBOOK_PATH };
}

/** Flag → interactive prompt → default (install). */
async function settleInstallSkill(ctx: SettleContext): Promise<boolean> {
  if (ctx.options.skill === false) return false;
  if (ctx.options.skill !== undefined) return ctx.options.skill;
  if (ctx.interactive) {
    const answer = await ctx.prompt("Install skill? (y/N)");
    return answer === "y" || answer === "Y";
  }
  return true;
}

// ---------------------------------------------------------------------------
// runWizard — unified settlement
// ---------------------------------------------------------------------------

/**
 * Settle each wizard answer from its flag, from a prompt in interactive mode,
 * or from its default under `--yes` / non-TTY.
 *
 * Non-interactive mode (`yes` or stdin not a TTY) asks nothing; missing
 * required answers throw a {@link GymratError} naming the flag.
 */
export async function runWizard(options: WizardOptions): Promise<WizardResult> {
  const interactive = isInteractive(options);
  const reader = interactive ? createLineReader(options.input) : undefined;
  const { output } = options;

  async function prompt(question: string, opts?: AskInternalOptions): Promise<string | undefined> {
    if (!reader) return undefined;
    return askPrompt(question, output, reader.nextLine, opts);
  }

  try {
    const baseCtx: SettleContext = {
      options,
      interactive,
      advancedGate: false,
      prompt,
      inputClosed: () => reader?.atEof ?? true,
    };
    const bench = await settleBench(baseCtx);
    const advancedGate = await settleAdvancedGate(baseCtx);
    const ctx: SettleContext = { ...baseCtx, advancedGate };

    const adapter = await settleAdapter(ctx);
    const checks = await settleChecks(ctx);
    const stopTarget = await settleStopTarget(ctx);
    const primary = await settlePrimary(ctx, stopTarget);
    const stopMaxIterations = await settleMaxIterations(ctx);
    const runbook = await settleRunbook(ctx);
    const installSkill = await settleInstallSkill(ctx);

    const result: WizardResult = { bench, installSkill, runbook };
    if (adapter && adapter !== CONFIG_DEFAULTS.adapter) result.adapter = adapter;
    if (checks) result.checks = checks;
    if (stopTarget !== undefined) result.stopTarget = stopTarget;
    if (primary) result.primary = primary;
    if (stopMaxIterations !== undefined) result.stopMaxIterations = stopMaxIterations;
    return result;
  } finally {
    reader?.close();
  }
}
