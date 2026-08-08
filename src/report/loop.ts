/**
 * How the loop states an iteration: its run header, and the verdict it closes on.
 *
 * The table between the two is the comparison report `text.ts` already renders —
 * the loop only replaces the header above it and appends the verdict below it,
 * so a reader who knows `gymrat compare` reads an iteration without relearning
 * anything.
 */

import type { Style } from "./format.js";
import { formatDelta, formatHintLabel, formatLabel } from "./format.js";
import { pairedSamples } from "./text.js";
import type { MetricComparisons } from "./types.js";

/**
 * What an iteration amounted to.
 *
 * Narrower than a metric verdict: an iteration has no `unstable` of its own,
 * because a primary figure too noisy to read is a figure that reported nothing,
 * which is what `no-signal` already says.
 */
export type LoopOutcome = "improved" | "regressed" | "no-signal";

/**
 * The one figure an iteration is read on, and how far it moved.
 *
 * A geomean carries no name: it is the aggregate over the run's gating metrics,
 * and it is direction-normalized, so a negative value improves whichever way its
 * metrics point. A named metric keeps its name, because the direction it
 * improves in is its own metadata's to say.
 */
export type LoopPrimary =
  | { readonly kind: "geomean"; readonly deltaPct: number }
  | { readonly kind: "metric"; readonly name: string; readonly deltaPct: number };

/**
 * One metric a confirmation rerun re-measured, and whether the rerun agreed.
 *
 * A regression only stands when both runs call it, so the block reports the
 * rerun's answer either way: a confirmed regression is the iteration's news, and
 * an unconfirmed one explains why a metric that read `regressed` in the table's
 * first pass now rests at no signal.
 */
export interface RerunConfirmation {
  readonly metric: string;
  readonly confirmed: boolean;
}

/** The candidate an iteration measures: the experiment, judged against the baseline. */
export const EXPERIMENT_INDEX = 0;

/** What the loop's header says it compared, the pair being fixed for every iteration. */
const COMPARED = "experiment vs baseline";

/** What an iteration that met the configured target says, and what it asks for. */
const TARGET_REACHED = "target reached — keep it";

/** The word each outcome is announced with. */
const OUTCOME_WORDS: Record<LoopOutcome, string> = {
  improved: "IMPROVED",
  regressed: "REGRESSED",
  "no-signal": "NO-SIGNAL",
};

/**
 * How each outcome's word is painted.
 *
 * Emboldened whatever it says — the word is the line's news — and colored only
 * where there is a direction to report. A no-signal iteration is neither good
 * nor bad, so it wears no color rather than a hedged one.
 */
const OUTCOME_STYLES: Record<LoopOutcome, Style> = {
  improved: ["bold", "green"],
  regressed: ["bold", "red"],
  "no-signal": ["bold"],
};

/**
 * The dimmed `·` the loop's lines separate their parts with.
 *
 * Built per call rather than held in a module constant: `styleText` decides on
 * color from the environment each time it runs, and a constant would pin
 * whatever the environment said at import time onto every later render.
 */
function separator(): string {
  return formatLabel(" · ", ["dim"]);
}

/**
 * The run header of one iteration: which iteration it is, what it compared, and
 * how many rounds stand behind the comparison.
 *
 * Passed to `renderReport` as its header override, so the table below it opens
 * on the loop's own terms rather than on `gymrat compare`'s. The adapter goes
 * unnamed: a session fixes it once, and repeating it every iteration spends a
 * header on news the reader had at `gymrat start`.
 */
export function formatLoopHeader(seq: number, samples: number): string {
  return [formatLabel(`iteration ${seq}`, ["bold"]), COMPARED, pairedSamples(samples)].join(
    separator(),
  );
}

/** What the rerun settled about one metric, painted the way the table paints that answer. */
function formatRerunLine(rerun: RerunConfirmation): string {
  const phrase = rerun.confirmed
    ? formatLabel("regression confirmed on rerun", ["red"])
    : formatLabel("regression not confirmed on rerun", ["dim"]);
  return `${rerun.metric}: ${phrase}`;
}

/**
 * The lines closing an iteration: what a confirmation rerun settled, what the
 * primary figure did, the verdict read off it, and the step that follows.
 *
 * The rerun lines open the block rather than close it because they qualify the
 * table above — a metric the table shows at rest that the first run had called a
 * regression is only readable once the rerun is named.
 *
 * A reached target is stated last, directly above the next step, because it is
 * an instruction rather than a reading: the loop only stops once the iteration
 * that reached the target is kept.
 *
 * Returned as lines rather than a block of text so the caller appends them to
 * the report it already holds as lines.
 */
export function formatVerdictBlock(
  outcome: LoopOutcome,
  primary: LoopPrimary,
  nextStep: string,
  reruns: readonly RerunConfirmation[] = [],
  targetReached = false,
): readonly string[] {
  const verdict = formatLabel(OUTCOME_WORDS[outcome], OUTCOME_STYLES[outcome]);
  return [
    ...reruns.map(formatRerunLine),
    `primary: ${formatDelta(primary.deltaPct)}${separator()}verdict: ${verdict}`,
    ...(targetReached ? [formatLabel(TARGET_REACHED, ["green"])] : []),
    `${formatHintLabel()} ${nextStep}`,
  ];
}

/** Whether any metric the run is gated on came back regressed for the experiment. */
function hasGatingRegression(metrics: MetricComparisons): boolean {
  return Object.values(metrics).some(
    (metric) =>
      metric.meta.gating && metric.candidates[EXPERIMENT_INDEX]?.verdict?.verdict === "regressed",
  );
}

/** Whether the primary figure moved the way its direction calls an improvement. */
function primaryImproved(metrics: MetricComparisons, primary: LoopPrimary): boolean {
  if (primary.kind === "geomean") {
    return primary.deltaPct < 0;
  }
  const direction = metrics[primary.name]?.meta.direction;
  if (direction === undefined) {
    return false;
  }
  return direction === "higher" ? primary.deltaPct > 0 : primary.deltaPct < 0;
}

/**
 * What an iteration amounted to, read off its metrics and its primary figure.
 *
 * A gating regression settles it whatever the primary did: the run is judged on
 * every metric it gates, so a headline that improved while a gate broke is still
 * an iteration to fix rather than one to keep.
 *
 * Everything that is neither a regression nor an improvement in the primary's
 * own direction reads `no-signal` — including a primary the run never measured,
 * which reports nothing rather than reporting zero.
 */
export function deriveOutcome(metrics: MetricComparisons, primary: LoopPrimary): LoopOutcome {
  if (hasGatingRegression(metrics)) {
    return "regressed";
  }
  return primaryImproved(metrics, primary) ? "improved" : "no-signal";
}
