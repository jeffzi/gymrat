import type { BenchlessConfig } from "../config.js";
import { GymratError } from "../errors.js";
import { exec } from "../exec.js";
import { formatDelta } from "../report/format.js";
import type { DiscardRecord, IterationRecord, KeepRecord } from "../session/records.js";
import { appendRecord, endsOnGatingBlock, readRecords, requireSession } from "../session/store.js";
import { advanceBaseline, commitWorkspace, revertWorkspace } from "../session/workspace.js";

const MS_PER_SECOND = 1000;

/** What the checks command answered, once it has run. */
interface ChecksRun {
  passed: boolean;
  /** Everything the command wrote, both streams, as the agent needs to read it. */
  output: string;
}

/** What a caller can hand a keep beyond its configuration. */
export interface KeepOptions {
  /** The commit message; absent, one is generated from the iteration being kept. */
  message?: string;
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
  const { session, state, jsonlPath } = requireSession(root, "settling an edit");
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

  if (hasConfirmedGatingRegression(iteration)) {
    return blockedKeep(
      jsonlPath,
      iteration.seq,
      "gating-regression",
      { configured },
      `Keep refused: iteration ${iteration.seq} regressed a gating metric.\nHint: fix the regression and run gymrat iterate again, or run gymrat discard.`,
    );
  }

  const checks = await runChecks(config, session.worktrees.experiment);
  if (checks !== undefined && !checks.passed) {
    return blockedKeep(
      jsonlPath,
      iteration.seq,
      "checks-failed",
      { configured: true, passed: false },
      `Keep refused: the checks command failed.\n\n${checks.output}\nHint: fix the failures and run gymrat keep again.`,
    );
  }

  const message = options.message ?? generatedMessage(iteration);
  const commit = commitWorkspace(session.worktrees.experiment, message);

  const record: KeepRecord = {
    type: "keep",
    seq: iteration.seq,
    at: new Date().toISOString(),
    status: "committed",
    commit,
    message,
    checks: checks === undefined ? { configured: false } : { configured: true, passed: true },
  };
  // Move the baseline before recording the keep: a record written first would
  // settle the iteration even when git refuses the advance, leaving the loop
  // sampling a baseline the log says it has already left behind. Failing with
  // the iteration still unsettled lets the agent retry the keep.
  advanceBaseline(session.worktrees.baseline, commit);
  appendRecord(jsonlPath, record);

  return {
    record,
    report: `Kept iteration ${iteration.seq} as ${commit}\n  message: ${message}\n  the baseline now measures against this commit`,
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
  const { session, state, jsonlPath } = requireSession(root, "settling an edit");
  const afterGatingBlock = endsOnGatingBlock(readRecords(jsonlPath));

  if (!state.unsettled && !afterGatingBlock) {
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
    seq: afterGatingBlock ? state.lastSeq + 1 : state.lastSeq,
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
 */
function hasConfirmedGatingRegression(iteration: IterationRecord): boolean {
  if (iteration.outcome !== "regressed") {
    return false;
  }
  return Object.values(iteration.metrics).some(
    (metric) =>
      metric.gating &&
      metric.verdict === "regressed" &&
      (metric.confirmed || metric.method === "exact"),
  );
}

/**
 * Run the configured checks in the experiment worktree.
 *
 * A timeout counts as a failure with whatever the command managed to write: the
 * gate asks whether the tree is provably good, and a run that never finished has
 * not answered.
 *
 * @returns What the command answered, or `undefined` when no checks are
 *   configured — in which case the missing gate is warned about instead.
 */
async function runChecks(
  config: BenchlessConfig,
  experimentDir: string,
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
  });
  const timedOut = "kind" in result;

  return {
    passed: !timedOut && result.exitCode === 0,
    output: [
      ...(timedOut ? [`${command} timed out after ${result.timeoutMs}ms`] : []),
      result.stdout,
      result.stderr,
    ]
      .filter((text) => text.trim() !== "")
      .join("\n"),
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
