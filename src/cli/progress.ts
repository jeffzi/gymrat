import yoctoSpinner from "yocto-spinner";

import type { ProgressStep } from "../compare.js";
import { assertNever } from "../errors.js";
import { EtaTracker, formatEta } from "../eta.js";
import { formatLabel, shortenLabel } from "../report/format.js";

/** Shown after a sample step until enough gaps have been measured for an ETA. */
const ETA_PENDING_LABEL = "estimating time left…";

/** Per-field presentation applied by {@link renderProgressLine}. */
interface ProgressLineStyle {
  readonly label: (text: string) => string;
  readonly counter: (text: string) => string;
  readonly eta: (text: string) => string;
}

/**
 * Assemble a progress line from its parts, applying `style` to the label,
 * counter, and ETA suffix.
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

function formatProgressLine(step: ProgressStep, etaMs?: number): string {
  return renderProgressLine(step, etaMs, {
    label: (text) => text,
    counter: (text) => text,
    eta: (text) => text,
  });
}

function styleProgressLine(step: ProgressStep, etaMs?: number): string {
  return renderProgressLine(step, etaMs, {
    label: (text) => formatLabel(text, "cyan", process.stderr),
    counter: (text) => formatLabel(text, "bold", process.stderr),
    eta: (text) => formatLabel(text, "dim", process.stderr),
  });
}

/** Carriage-return + clear-to-EOL. */
const CLEAR_LINE = "\r\x1b[K";

/**
 * Cut `line` down to a single terminal row.
 */
function fitToTerminalWidth(line: string): string {
  const columns = process.stderr.columns as number | undefined;
  if (columns === undefined) {
    return line;
  }
  return shortenLabel(line, columns - 1);
}

function writeProgress(line: string, tty: boolean): void {
  process.stderr.write(tty ? `${CLEAR_LINE}${fitToTerminalWidth(line)}` : `${line}\n`);
}

function clearProgress(tty: boolean): void {
  if (tty) {
    process.stderr.write(CLEAR_LINE);
  }
}

function writeWarning(message: string): void {
  process.stderr.write(`${message}\n`);
}

/** Single-use: `stop()` must be called exactly once, after the run completes or fails. */
export interface ProgressReporter {
  emit(step: ProgressStep): void;
  warn(message: string): void;
  stop(): void;
}

/** How often the spinner's ETA countdown ticks down while waiting for the next step. */
const COUNTDOWN_TICK_MS = 1000;

function emitToSpinner(
  spinner: ReturnType<typeof yoctoSpinner>,
  step: ProgressStep,
  etaMs: number | undefined,
  countdownState: { interval: ReturnType<typeof setInterval> | undefined },
): void {
  spinner.text = styleProgressLine(step, etaMs);
  if (!spinner.isSpinning) {
    spinner.start();
  }
  clearCountdownInterval(countdownState);
  if (etaMs !== undefined) {
    const emitTime = Date.now();
    countdownState.interval = setInterval(() => {
      const remaining = Math.max(0, etaMs - (Date.now() - emitTime));
      spinner.text = styleProgressLine(step, remaining);
    }, COUNTDOWN_TICK_MS);
  }
}

function clearCountdownInterval(state: {
  interval: ReturnType<typeof setInterval> | undefined;
}): void {
  if (state.interval !== undefined) {
    clearInterval(state.interval);
    state.interval = undefined;
  }
}

/**
 * TTY + color allowed: use yocto-spinner (yellow glyph on stderr).
 * TTY + color vetoed: fall back to \r\x1b[K overwrite with plain text.
 * Non-TTY: one newline-terminated line per step, no ANSI.
 */
export function createProgressReporter(
  colorAllowed: boolean,
  tty: boolean,
  targetCount: number,
): ProgressReporter {
  const spinner = colorAllowed
    ? yoctoSpinner({ color: "yellow", stream: process.stderr })
    : undefined;
  const eta = new EtaTracker(targetCount);
  const countdownState: { interval: ReturnType<typeof setInterval> | undefined } = {
    interval: undefined,
  };
  let drawnStep: { step: ProgressStep; etaMs?: number } | undefined;

  return {
    emit(step: ProgressStep): void {
      const etaMs = eta.record(step);
      if (spinner) {
        emitToSpinner(spinner, step, etaMs, countdownState);
        return;
      }
      drawnStep = tty ? { step, ...(etaMs !== undefined && { etaMs }) } : undefined;
      writeProgress(formatProgressLine(step, etaMs), tty);
    },
    warn(message: string): void {
      if (spinner) {
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
      clearCountdownInterval(countdownState);
      drawnStep = undefined;
      if (spinner) {
        spinner.stop();
      } else {
        clearProgress(tty);
      }
    },
  };
}
