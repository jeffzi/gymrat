/**
 * How the loop states an iteration — its run header and the verdict it closes on
 * — and how `gymrat status` states a whole session.
 *
 * The table between an iteration's header and its verdict is the comparison
 * report `text.ts` already renders: the loop only replaces the header above it
 * and appends the verdict below it, so a reader who knows `gymrat compare` reads
 * an iteration without relearning anything. The status lines follow the same
 * conventions — the table's glyphs, the verdict block's colors, the dimmed `·`
 * — so a session reads as the iterations it is made of.
 */

import type { ConfigStop } from "../config.js";
import { assertNever } from "../errors.js";
import { computeMedian } from "../math.js";
import type {
  BaselineRecord,
  FinalizeRecord,
  KeepRecord,
  SessionRecord,
} from "../session/records.js";
import type { DisplayClass, Style } from "./format.js";
import {
  formatDelta,
  formatHintLabel,
  formatLabel,
  formatValue,
  getGlyph,
  pluralize,
} from "./format.js";
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
 *
 * `deltaPct` is `null` where the ratio has no value — a baseline median of zero
 * leaves it nothing to normalize against. That is a figure with no direction to
 * read, not a figure that stood still, so nothing may coerce it to `0`.
 */
export type LoopPrimary =
  | { readonly kind: "geomean"; readonly deltaPct: number | null }
  | { readonly kind: "metric"; readonly name: string; readonly deltaPct: number | null };

/**
 * What a confirmation rerun had to say about one metric it was asked to
 * re-measure.
 *
 * `absent` is not a weaker `disagreed`: a rerun that never reported the metric
 * disproved nothing, so the regression the first run called still stands. Only
 * `disagreed` — the rerun measured the metric and did not call it regressed —
 * takes a regression back.
 */
type RerunAnswer = "confirmed" | "disagreed" | "absent";

/**
 * One metric a confirmation rerun was asked about, and what it answered.
 *
 * A regression only stands when both runs call it, so the block reports the
 * rerun's answer whatever it was: a confirmed regression is the iteration's
 * news, a disagreement explains why a metric that read `regressed` in the
 * table's first pass now rests at no signal, and an absent one says the rerun
 * had nothing to say about a metric still shown as regressed.
 */
export interface RerunConfirmation {
  readonly metric: string;
  readonly answer: RerunAnswer;
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
 * A primary figure's move, as a signed percentage, or nothing at all when the
 * ratio had no value.
 *
 * Blank is what the table already shows for the `NaN` the engine computes there,
 * so the two agree: a reader is shown no percentage rather than one they could
 * read a direction into. Whatever stands beside it — the glyph, the verdict —
 * still says what the iteration amounted to.
 */
function formatPrimaryDelta(deltaPct: number | null): string {
  return deltaPct === null ? "" : ` ${formatDelta(deltaPct)}`;
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

/**
 * What each answer reads as, and how it is painted.
 *
 * An absent answer wears the same yellow the table paints an unstable metric
 * with, because it is the same kind of news: a reading nobody could take, not a
 * direction to act on.
 */
const RERUN_PHRASES: Record<RerunAnswer, { readonly text: string; readonly style: Style }> = {
  confirmed: { text: "regression confirmed on rerun", style: ["red"] },
  disagreed: { text: "regression not confirmed on rerun", style: ["dim"] },
  absent: { text: "not measured on rerun", style: ["yellow"] },
};

/** What the rerun settled about one metric, painted the way the table paints that answer. */
function formatRerunLine(rerun: RerunConfirmation): string {
  const phrase = RERUN_PHRASES[rerun.answer];
  return `${rerun.metric}: ${formatLabel(phrase.text, phrase.style)}`;
}

interface VerdictBlockOptions {
  outcome: LoopOutcome;
  primary: LoopPrimary;
  nextStep: string;
  reruns?: readonly RerunConfirmation[];
  targetReached?: boolean;
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
export function formatVerdictBlock(options: VerdictBlockOptions): readonly string[] {
  const { outcome, primary, nextStep, reruns = [], targetReached = false } = options;
  const verdict = formatLabel(OUTCOME_WORDS[outcome], OUTCOME_STYLES[outcome]);
  return [
    ...reruns.map(formatRerunLine),
    // The delta renders blank when it is undefined, so it is joined rather than
    // interpolated — a blank between a space and the separator reads as a gap.
    [`primary:${formatPrimaryDelta(primary.deltaPct)}`, `verdict: ${verdict}`].join(separator()),
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

/**
 * Whether the primary figure moved the way its direction calls an improvement.
 *
 * A figure whose ratio had no value moved in no direction at all, so it improves
 * nothing — the same answer the `NaN` it stands for gives, every comparison
 * against which is false.
 */
function primaryImproved(metrics: MetricComparisons, primary: LoopPrimary): boolean {
  if (primary.deltaPct === null) {
    return false;
  }
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

/**
 * What became of a measured iteration.
 *
 * A blocked keep is its own state rather than a variant of `unsettled`: the
 * iteration is still waiting to be settled, but the log knows why the last
 * attempt did not land, and that reason is what the agent acts on.
 */
export type SettleState =
  | { readonly kind: "kept"; readonly commit?: string }
  | { readonly kind: "discarded" }
  | { readonly kind: "unsettled" }
  | { readonly kind: "keep-blocked"; readonly reason?: NonNullable<KeepRecord["reason"]> };

/** One iteration as a status line states it. */
export interface StatusIteration {
  readonly seq: number;
  /** How far the iteration's primary figure moved, or `null` where the ratio had no value. */
  readonly deltaPct: number | null;
  readonly outcome: LoopOutcome;
  readonly settle: SettleState;
}

/** What a session adds up to: how its iterations settled, and how near its stop it is. */
export interface StatusSummary {
  readonly iterationCount: number;
  readonly keepCount: number;
  readonly discardCount: number;
  /** The configured stop conditions; absent, the session runs until the agent stops it. */
  readonly stop?: ConfigStop;
  /** Whether a kept iteration reached the configured target. */
  readonly targetReached: boolean;
}

/** How many hex digits of a sha a display shows. */
export const SHORT_SHA_LENGTH = 7;

/** A session's baseline, in the `ref@sha` form (sha shortened) every loop summary states it with. */
export function formatBaselineRef(baseline: SessionRecord["baseline"]): string {
  return `${baseline.ref}@${baseline.sha.slice(0, SHORT_SHA_LENGTH)}`;
}

/**
 * The glyph each outcome wears, borrowed from the comparison table's vocabulary.
 *
 * A no-signal iteration takes the table's within-noise glyph: both say the same
 * thing — the figure moved by nothing the run can stand behind.
 */
const OUTCOME_GLYPHS: Record<LoopOutcome, DisplayClass> = {
  improved: "improved",
  regressed: "regressed",
  "no-signal": "within-noise",
};

/**
 * The session a status report opens on: what it is, what it forked from, where
 * it works, and how it measures.
 *
 * Both worktree paths are named for the reason `gymrat start` names them: the
 * agent edits in one of them and must never touch the other.
 */
export function formatStatusHeader(session: SessionRecord): readonly string[] {
  const baseline = formatBaselineRef(session.baseline);
  return [
    [
      formatLabel(`session ${session.sessionId}`, ["bold"]),
      `baseline ${baseline}`,
      `adapter ${session.config.adapter}`,
    ].join(separator()),
    `branch ${session.branch}`,
    `experiment worktree ${session.worktrees.experiment}`,
    `baseline worktree ${session.worktrees.baseline}`,
  ];
}

/**
 * How an iteration was settled, in the words `status` reports it with.
 *
 * A settling record that settled no iteration — a keep refused for want of a
 * measurement — stands on a line of its own, and this is all that line says.
 */
export function formatStatusSettle(settle: SettleState): string {
  switch (settle.kind) {
    case "kept":
      return settle.commit === undefined
        ? "kept"
        : `kept ${settle.commit.slice(0, SHORT_SHA_LENGTH)}`;
    case "discarded":
      return "discarded";
    case "unsettled":
      return "unsettled";
    case "keep-blocked":
      return settle.reason === undefined ? "keep-blocked" : `keep-blocked (${settle.reason})`;
    default:
      return assertNever(settle);
  }
}

/**
 * One iteration of the session's history: which one it was, what it did, and
 * what became of it.
 *
 * The glyph carries the outcome's own color, so a session's course is legible
 * down the left of the report before a word of it is read.
 */
export function formatStatusIteration(iteration: StatusIteration): string {
  const glyph = formatLabel(
    getGlyph(OUTCOME_GLYPHS[iteration.outcome]),
    OUTCOME_STYLES[iteration.outcome],
  );
  return [
    `iteration ${iteration.seq}`,
    `${glyph}${formatPrimaryDelta(iteration.deltaPct)}`,
    formatStatusSettle(iteration.settle),
  ].join(separator());
}

/** The median each metric measured across `samples`, in the order the rounds first named them. */
function metricMedians(samples: readonly Record<string, number>[]): [string, number][] {
  const readings = samples.flatMap((round) => Object.entries(round));
  const byMetric = Map.groupBy(readings, ([name]) => name);
  return [...byMetric].map(([name, entries]) => [
    name,
    computeMedian(entries.map(([, value]) => value)),
  ]);
}

/**
 * A recorded baseline measurement: what was measured, and the median each of
 * its metrics came to.
 *
 * The log stores every round the measurement took, so the medians are computed
 * here rather than stored — a later statistics change re-reads the same records
 * instead of invalidating them.
 */
export function formatStatusBaseline(record: BaselineRecord): string {
  return [
    `baseline ${record.label}`,
    ...metricMedians(record.samples).map(([name, median]) => `${name} ${formatValue(median)}`),
  ].join(separator());
}

/** Where the session stands against its configured stop, or nothing when none is configured. */
function formatStopState(summary: StatusSummary): string | undefined {
  const { stop } = summary;
  if (stop === undefined) {
    return undefined;
  }

  const parts = [
    ...(stop.maxIterations === undefined
      ? []
      : [`${summary.iterationCount} of ${stop.maxIterations} iterations`]),
    ...(stop.targetValue === undefined
      ? []
      : [summary.targetReached ? "target reached" : "target pending"]),
  ];
  if (parts.length === 0) {
    return undefined;
  }
  return `${formatLabel("stop:", ["dim"])} ${parts.join(separator())}`;
}

/**
 * The lines closing a status report: how the session's iterations settled, and
 * how near it is to the stop it was given.
 *
 * The stop line is left out when nothing is configured rather than reported as
 * unlimited: a loop the agent stops when it likes has no state to state.
 */
export function formatStatusFooter(summary: StatusSummary): readonly string[] {
  const totals = [
    pluralize(summary.iterationCount, "iteration"),
    `${summary.keepCount} kept`,
    `${summary.discardCount} discarded`,
  ].join(separator());
  const stop = formatStopState(summary);
  return stop === undefined ? [totals] : [totals, stop];
}

/**
 * The line a finalized session's report ends on: where its work ended up.
 *
 * It sits under the totals rather than in the header because closing the
 * session is the last thing that happened to it, and the branch and commit it
 * names are what the reader goes to next — everything above them is history.
 */
export function formatStatusFinalized(finalized: FinalizeRecord): string {
  return [
    formatLabel("finalized", ["bold"]),
    `branch ${finalized.branch}`,
    `commit ${finalized.commit.slice(0, SHORT_SHA_LENGTH)}`,
  ].join(separator());
}
