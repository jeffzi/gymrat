import type { ProgressStep } from "./compare.js";

/**
 * Tracks wall-clock gaps between progress steps and estimates the remaining
 * time for a round-robin benchmark run.
 *
 * The estimate equals `meanSampleDuration * remainingSteps`, where
 * `remainingSteps = total * targetCount - completedSampleSteps`.
 *
 * Gaps that follow a `prepare` step are excluded from the mean — prepare
 * durations are unpredictable and would skew the estimate.
 */
export class EtaTracker {
  readonly #targetCount: number;
  readonly #clock: () => number;

  /** Durations of completed sample-to-sample gaps (milliseconds). */
  #durations: number[] = [];

  /** Whether the previous step was a `prepare` step (gap must be excluded). */
  #prevWasPrepare = false;

  #prevTime: number | undefined;

  #completedSamples = 0;

  constructor(targetCount: number, clock: () => number = Date.now) {
    this.#targetCount = targetCount;
    this.#clock = clock;
  }

  /**
   * Returns undefined for `prepare` steps and when no sample-to-sample gap
   * has been measured yet; otherwise returns the estimated remaining
   * wall-clock time in milliseconds.
   */
  record(step: ProgressStep): number | undefined {
    const now = this.#clock();
    const isPrepare = step.kind === "prepare";

    if (this.#prevTime !== undefined && !this.#prevWasPrepare && !isPrepare) {
      this.#durations.push(now - this.#prevTime);
    }

    this.#prevWasPrepare = isPrepare;
    this.#prevTime = now;

    if (isPrepare) {
      return undefined;
    }

    const remaining = step.total * this.#targetCount - this.#completedSamples;
    this.#completedSamples++;

    if (this.#durations.length === 0) {
      return undefined;
    }

    const mean =
      this.#durations.reduce((sum, duration) => sum + duration, 0) / this.#durations.length;

    return mean * remaining;
  }
}

const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3600;

/**
 * Join a whole unit with an optional remainder unit — `"3m 5s"`, or just
 * `"3m"` when the remainder is zero.
 */
function formatRemainder(
  whole: number,
  wholeUnit: string,
  remainder: number,
  remainderUnit: string,
): string {
  return remainder > 0
    ? `~${String(whole)}${wholeUnit} ${String(remainder)}${remainderUnit} left`
    : `~${String(whole)}${wholeUnit} left`;
}

/**
 * Returns a `"~<value><unit> left"` string, clamped to a minimum of 1
 * second, with at most two unit tiers (e.g. `"~3m 5s left"`).
 */
export function formatEta(ms: number): string {
  const totalSeconds = Math.max(1, Math.round(ms / 1000));

  if (totalSeconds < SECONDS_PER_MINUTE) {
    return `~${String(totalSeconds)}s left`;
  }

  if (totalSeconds < SECONDS_PER_HOUR) {
    const minutes = Math.floor(totalSeconds / SECONDS_PER_MINUTE);
    const seconds = totalSeconds % SECONDS_PER_MINUTE;
    return formatRemainder(minutes, "m", seconds, "s");
  }

  const hours = Math.floor(totalSeconds / SECONDS_PER_HOUR);
  const minutes = Math.floor((totalSeconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
  return formatRemainder(hours, "h", minutes, "m");
}
