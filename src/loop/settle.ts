import type { BenchlessConfig } from "../config.js";
import { GymratError } from "../errors.js";
import { exec } from "../exec.js";
import { formatDelta } from "../report/format.js";
import type { DiscardRecord, IterationRecord, KeepRecord } from "../session/records.js";
import { appendRecord, requireOpenSession } from "../session/store.js";
import {
  advanceBaseline,
  commitWorkspace,
  isWorktreeDirty,
  revertWorkspace,
  worktreeHead,
} from "../session/workspace.js";
import { limitOutput } from "./output-limit.js";

const MS_PER_SECOND = 1000;

/** What the checks command answered, once it has run. */
interface ChecksRun {
  passed: boolean;
  /** Both streams as the agent needs to read them, each cut to the relay limit. */
  output: string;
  /** What the command wrote on stdout, however much of it {@link output} carries. */
  stdoutBytes: number;
  /** What the command wrote on stderr, however much of it {@link output} carries. */
  stderrBytes: number;
}

/** What a caller can hand a keep beyond its configuration. */
export interface KeepOptions {
  /** The commit message; absent, one is generated from the iteration being kept. */
  message?: string;
  /** Aborting it kills the in-flight checks command. Omitted, nothing can interrupt the checks. */
  signal?: AbortSignal;
}

/** One settled — or refused — keep: what was written to the log, and what to print about it. */
export interface KeepResult {
  /** The record appended to the session log, committed or blocked. */
  record: KeepRecord;
  /** The keep as the agent reads it: the commit, or the reason there is none. */
  report: string;
}

/** One reverted iteration: what was written to the log, and what to print about it. */
export interface DiscardResult {
  /** The record appended to the session log. */
  record: DiscardRecord;
  /** The discard as the agent reads it. */
  report: string;
}

/**
 * Commit the measured edit standing in the experiment worktree, if it may be kept.
 *
 * Three gates guard the commit, and each one that trips is recorded rather than
 * thrown: a blocked keep is history the agent — and `gymrat status` — can read
 * back, which a raised error would leave nowhere. The caller turns a blocked
 * record into the exit code; every other failure here is a `GymratError`.
 *
 * Holding the repository lock across the call is the caller's job: the baseline
 * worktree moves in the middle of it, and a concurrent `iterate` must not sample
 * it mid-advance.
 *
 * @throws GymratError when no session has been started, or when git refuses to
 *   commit the worktree or to advance the baseline.
 */
export async function keepSession(
  root: string,
  config: BenchlessConfig,
  options: KeepOptions = {},
): Promise<KeepResult> {
  const { session, state, jsonlPath } = requireOpenSession(root, "settling an edit");
  const configured = config.checks !== undefined;

  const iteration = state.unsettled ? state.lastIteration : undefined;
  if (iteration === undefined) {
    return blockedKeep(
      jsonlPath,
      // The refusal settles nothing, so it takes the number no iteration has used
      // yet: numbering it `lastSeq` would leave the log with two settlement
      // records against an iteration that was already kept or discarded.
      state.lastSeq + 1,
      "nothing-measured",
      { configured },
      "Keep refused: nothing has been measured since the last keep or discard.\nHint: run gymrat iterate first — an unmeasured commit is one the loop cannot account for.",
    );
  }

  if (hasStandingGatingRegression(iteration)) {
    return blockedKeep(
      jsonlPath,
      iteration.seq,
      "gating-regression",
      { configured },
      gatingRefusal(iteration),
    );
  }

  const experimentDir = session.worktrees.experiment;

  if (!isWorktreeDirty(experimentDir)) {
    // The worktree is clean — either the agent made no changes (nothing to
    // commit), or a prior keep already committed the work but failed before
    // recording it (retry). The baseline's current position distinguishes them.
    return keepCleanWorktree(
      jsonlPath,
      experimentDir,
      session.worktrees.baseline,
      state.lastKeptCommit ?? session.baseline.sha,
      iteration,
      configured,
      options,
    );
  }

  const checks = await runChecks(config, experimentDir, options.signal);
  if (checks !== undefined && !checks.passed) {
    return blockedKeep(
      jsonlPath,
      iteration.seq,
      "checks-failed",
      {
        configured: true,
        passed: false,
        stdoutBytes: checks.stdoutBytes,
        stderrBytes: checks.stderrBytes,
      },
      `Keep refused: the checks command failed.\n\n${checks.output}\nHint: fix the failures and run gymrat keep again.`,
    );
  }

  const message = options.message ?? generatedMessage(iteration);
  const commit = commitWorkspace(experimentDir, message);
  const checksField: KeepRecord["checks"] =
    checks === undefined ? { configured: false } : { configured: true, passed: true };

  return commitKeep(
    jsonlPath,
    session.worktrees.baseline,
    iteration.seq,
    commit,
    message,
    checksField,
  );
}

/**
 * Settle a keep against a worktree that has nothing left to commit: either
 * nothing was measured (agent never edited the tree) or a prior keep already
 * committed the work but failed before recording it, in which case the commit
 * already made is picked up rather than repeated.
 */
function keepCleanWorktree(
  jsonlPath: string,
  experimentDir: string,
  baselineDir: string,
  baselinePosition: string,
  iteration: IterationRecord,
  configured: boolean,
  options: KeepOptions,
): KeepResult {
  const head = worktreeHead(experimentDir);

  if (head === baselinePosition) {
    // HEAD matches the baseline: the agent measured an iteration but never
    // edited the worktree. There is nothing to commit, and running checks
    // or git-commit would waste time on a tree that has nothing to give.
    return blockedKeep(
      jsonlPath,
      iteration.seq,
      "nothing-to-commit",
      { configured },
      "Keep refused: the experiment worktree has nothing to commit.\nHint: edit the code in the experiment worktree, then run gymrat keep again.",
    );
  }

  // HEAD is ahead of the baseline: a prior call committed the work but
  // failed at advanceBaseline or appendRecord. Skip the commit and pick up
  // where the prior call left off — the checks already passed on the first
  // attempt, and the commit is already made.
  const message = options.message ?? generatedMessage(iteration);
  return commitKeep(jsonlPath, baselineDir, iteration.seq, head, message, { configured });
}

/**
 * Record a keep that committed, advance the baseline to it, and phrase the report.
 *
 * Both the fresh commit made by `keepSession` and the one recovered by
 * {@link keepCleanWorktree} on retry settle through here, so the record shape
 * and the report wording stay identical whichever path produced the commit.
 */
function commitKeep(
  jsonlPath: string,
  baselineDir: string,
  seq: number,
  commit: string,
  message: string,
  checks: KeepRecord["checks"],
): KeepResult {
  const record: KeepRecord = {
    type: "keep",
    seq,
    at: new Date().toISOString(),
    status: "committed",
    commit,
    message,
    checks,
  };
  // Move the baseline before recording the keep: a record written first would
  // settle the iteration even when git refuses the advance, leaving the loop
  // sampling a baseline the log says it has already left behind. Failing with
  // the iteration still unsettled lets the agent retry the keep.
  advanceBaseline(baselineDir, commit);
  appendRecord(jsonlPath, record);

  return {
    record,
    report: `Kept iteration ${seq} as ${commit}\n  message: ${message}\n  the baseline now measures against this commit`,
  };
}

/**
 * Throw the experiment worktree's uncommitted work away and record that it went.
 *
 * A clean worktree is discarded just as loudly as a dirty one: the record is what
 * settles the iteration, and gymrat does not guess whether an agent that changed
 * nothing meant to. What there must be is an edit to throw away — either an
 * unsettled iteration, or the one a gating regression refused to commit, which is
 * settled in the log yet still standing in the worktree. Anywhere else the discard
 * would number itself after an iteration the log already settled, and history
 * would read as two settlements of a single iteration.
 *
 * Holding the repository lock across the call is the caller's job — the revert
 * and the record it explains must not be separable by another run.
 *
 * @throws GymratError when no session has been started, when nothing has been
 *   measured since the last keep or discard, or when git refuses to revert the worktree.
 */
export function discardSession(root: string): DiscardResult {
  const { session, state, jsonlPath } = requireOpenSession(root, "settling an edit");

  if (!state.unsettled && !state.endsOnGatingBlock) {
    throw new GymratError(
      "Discard refused: nothing has been measured since the last keep or discard.",
      "Run gymrat iterate to measure an edit before settling it.",
    );
  }

  revertWorkspace(session.worktrees.experiment);

  const record: DiscardRecord = {
    type: "discard",
    // The block already settled the iteration it refused, so the discard behind it
    // takes the number no iteration has used yet — the same number a refused keep
    // takes. Reusing the iteration's own seq would make the discard the last
    // settling record to carry it, and `gymrat status` would render it in place of
    // the block instead of alongside it.
    seq: state.endsOnGatingBlock ? state.lastSeq + 1 : state.lastSeq,
    at: new Date().toISOString(),
  };
  appendRecord(jsonlPath, record);

  return {
    record,
    report: `Discarded iteration ${state.lastSeq}: the experiment worktree is back at its last commit`,
  };
}

/**
 * Whether the iteration carries a regression the loop refuses to commit over.
 *
 * Both halves are required: the outcome is what the agent was shown, and a
 * gating metric standing behind the regression is what makes it real. A noisy
 * metric earns that standing from the confirmation rerun — a regression the
 * rerun would not repeat leaves the iteration keepable, which is the whole
 * point of measuring it twice. An exact metric is deterministic, so
 * `confirmRegressions` skips it and its `confirmed` stays `false`; rerunning it
 * could only reproduce the same number, and gating on `confirmed` alone would
 * let every exact regression through.
 *
 * Silence earns the same standing as disagreement: a metric the rerun was asked
 * about and never reported back on lands in `confirm.absent`, its `confirmed`
 * still `false` because nothing re-measured it. The gate fails closed on those —
 * a rerun that cannot see the metric is not evidence the regression went away,
 * and treating no answer as a clean answer is how a regression walks into the
 * baseline.
 */
function hasStandingGatingRegression(iteration: IterationRecord): boolean {
  if (iteration.outcome !== "regressed") {
    return false;
  }
  if (unmeasuredGatingRegressions(iteration).length > 0) {
    return true;
  }
  return Object.values(iteration.metrics).some(
    (metric) => isGatingRegression(metric) && (metric.confirmed || metric.method === "exact"),
  );
}

/** The gating metrics that regressed and that the confirmation rerun never reported back on. */
function unmeasuredGatingRegressions(iteration: IterationRecord): string[] {
  const absent = new Set(iteration.confirm?.absent ?? []);
  return Object.entries(iteration.metrics)
    .filter(([name, metric]) => isGatingRegression(metric) && absent.has(name))
    .map(([name]) => name);
}

/** Whether a metric is a gating metric that the checks called a regression. */
function isGatingRegression(metric: IterationRecord["metrics"][string]): boolean {
  return metric.gating && metric.verdict === "regressed";
}

/**
 * How the refusal reads to the agent that has to act on it.
 *
 * A regression the rerun stood behind needs no explaining beyond the number the
 * iteration already reported. One the rerun never re-measured does: the agent is
 * looking at a metric its own report called regressed and unconfirmed, and
 * without the missing measurement named, the block reads as gymrat contradicting
 * itself. The extra hint points at the likeliest cause — a filter template that
 * narrows the rerun to a subset the bench does not answer with.
 */
function gatingRefusal(iteration: IterationRecord): string {
  const refusal = `Keep refused: iteration ${iteration.seq} regressed a gating metric.`;
  const settleHint = "fix the regression and run gymrat iterate again, or run gymrat discard";

  const unmeasured = unmeasuredGatingRegressions(iteration);
  if (unmeasured.length === 0) {
    return `${refusal}\nHint: ${settleHint}.`;
  }

  const named = unmeasured
    .map((name) => `  ${name}: not measured on the confirmation rerun, so the regression stands`)
    .join("\n");
  return `${refusal}\n${named}\nHint: check that the filter template (or the bench itself) reports ${unmeasured.join(", ")}, then ${settleHint}.`;
}

/**
 * Run the configured checks in the experiment worktree.
 *
 * A timeout counts as a failure with whatever the command managed to write: the
 * gate asks whether the tree is provably good, and a run that never finished has
 * not answered.
 *
 * Each stream is cut to the relay limit on its own, so a test suite that writes
 * its failures to stderr is as readable as one that writes them to stdout.
 *
 * @returns What the command answered, or `undefined` when no checks are
 *   configured — in which case the missing gate is warned about instead.
 */
async function runChecks(
  config: BenchlessConfig,
  experimentDir: string,
  signal?: AbortSignal,
): Promise<ChecksRun | undefined> {
  const command = config.checks;
  if (command === undefined) {
    process.stderr.write(
      "Warning: no checks command is configured, so gymrat keep is committing with the gate off.\n" +
        'Hint: set "checks" in gymrat.json to the command that must pass before an edit is kept.\n',
    );
    return undefined;
  }

  const result = await exec(command, {
    cwd: experimentDir,
    timeoutMs: config.timeoutSeconds * MS_PER_SECOND,
    signal,
  });
  const timedOut = "kind" in result;

  return {
    passed: !timedOut && result.exitCode === 0,
    output: [
      ...(timedOut ? [`${command} timed out after ${result.timeoutMs}ms`] : []),
      limitOutput(result.stdout),
      limitOutput(result.stderr),
    ]
      .filter((text) => text.trim() !== "")
      .join("\n"),
    // What the command wrote, not what was relayed: a figure above the relay
    // limit is how a reader of the log learns the report was cut short.
    stdoutBytes: Buffer.byteLength(result.stdout, "utf-8"),
    stderrBytes: Buffer.byteLength(result.stderr, "utf-8"),
  };
}

/** Record the refusal so the log carries it, and phrase it for the agent. */
function blockedKeep(
  jsonlPath: string,
  seq: number,
  reason: NonNullable<KeepRecord["reason"]>,
  checks: KeepRecord["checks"],
  report: string,
): KeepResult {
  const record: KeepRecord = {
    type: "keep",
    seq,
    at: new Date().toISOString(),
    status: "blocked",
    reason,
    checks,
  };
  appendRecord(jsonlPath, record);
  return { record, report };
}

/**
 * The commit message a keep writes when the agent supplied none.
 *
 * It names the iteration and the figure it was read on, so the branch's history
 * reads back as the loop that produced it rather than as a run of unlabelled
 * commits.
 *
 * A figure whose ratio had no value says so in words: the report can print a
 * blank percentage and let the glyph beside it carry the news, but a commit
 * subject has nothing beside it, and would trail off on the figure's bare name.
 */
function generatedMessage(iteration: IterationRecord): string {
  const { primary } = iteration;
  const moved = primary.deltaPct === null ? "delta undefined" : formatDelta(primary.deltaPct);
  return `iteration ${iteration.seq}: ${primary.name ?? primary.kind} ${moved}`;
}
