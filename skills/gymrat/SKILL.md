---
name: gymrat
description: >-
  Use when driving a gymrat optimization session toward a performance target.
  Not for one-shot comparisons (gymrat compare) or standalone measurements
  (gymrat measure without --record).
when_to_use: >-
  Also use when running gymrat start, gymrat iterate, gymrat keep, gymrat discard,
  gymrat finalize, or gymrat status; when a repo has a gymrat.json; when asked to
  optimize a benchmark toward a target or budget; or on errors like "has not been
  settled", "Keep refused", or "Stop condition met".
---

# Driving a gymrat optimization session

This skill covers the full session lifecycle: start, iterate, settle, finalize. Every command runs
from the repository root.

`bench` must be resolvable — from `gymrat.json` or `--bench`. `gymrat.json` is optional but is
where `checks`, `filter`, `primary`, `stop`, and `hooks` live; `adapter` and `samples` default to
`metric-lines` and `10`.

**Load the per-repo runbook before your first edit.** Read the project's CLAUDE.md for the
runbook's name and load it. If CLAUDE.md names none and no runbook exists under `.claude/skills/`,
stop and ask which metrics gate and what to optimize before editing.

## Session lifecycle

### 1. Start the session

```sh
gymrat start [ref]       # ref defaults to HEAD
```

Pins the baseline and creates two worktrees: experiment (where you edit) and baseline (read-only).

### 2. Record a baseline measurement (optional)

```sh
gymrat measure --record
```

Appends a baseline record to the session log. Do this once, before the first edit, when the
runbook asks for it.

### 3. The iteration cycle

Edit code **only in the experiment worktree** (the path `start` printed). Then:

```sh
gymrat iterate        # measure the edit, print the verdict
```

Read the verdict block. The opening lines report what a confirmation rerun already settled
(`regression confirmed on rerun`, `not confirmed`, or `not measured`). Then the verdict:

- **IMPROVED** — the primary metric moved in the right direction.
- **REGRESSED** — a gating metric regressed. For inexact metrics the rerun confirms the regression;
  exact metrics gate without a rerun (their value is deterministic). A metric absent from the rerun
  still gates — silence is not evidence the regression went away.
- **NO-SIGNAL** — the change did not move the needle.

The `Hint:` line tells you what to do next. Then settle:

```sh
gymrat keep -m "describe the optimization"   # commit the edit, advance the baseline
gymrat discard                                # revert the experiment worktree
```

`keep` refuses when nothing has been measured, when the iteration regressed a gating metric, or —
checked last — when the `checks` command fails. Every refusal exits 1 and is recorded.

**After a checks failure:** fix the issue and run `gymrat keep` again — the iteration stays
unsettled, so `iterate` is refused until the keep lands or you `discard`. After a
gating-regression refusal the iteration _is_ settled, so `iterate` or `discard` is correct.

Confirm `checks` is set in `gymrat.json` before the first `keep`; without it `keep` commits with
the gate off and only warns on stderr.

**One iteration at a time.** Each must be settled before the next `iterate`.

### 4. Read history

```sh
gymrat status
```

### 5. Close the session

```sh
gymrat finalize [-m "squash message"] [--branch <name>]
```

Collapses kept iterations into one squash commit on a new branch (default
`<session-branch>-final`). Requires every iteration settled, at least one keep, a clean experiment
worktree, and a free target branch (`--branch <name>` when `<session-branch>-final` already
exists). A finalized session refuses `iterate`, `keep`, `discard`, and `measure --record`.

## Loop discipline

1. **Never stop before a stop condition fires.** When `stop.maxIterations` or `stop.targetValue` is
   configured, keep iterating until `iterate` exits 1 naming the condition. When no `stop` is
   configured, the runbook's goal is the criterion — stop when it is met, or when repeated
   iterations produce NO-SIGNAL and the approach needs rethinking. When a configured target proves
   unreachable after sustained NO-SIGNAL results, report the shortfall and stop.

2. **Act on hook failures.** Hooks cannot fail the loop, but ignoring their output accumulates
   technical debt.

3. **Never run concurrent sessions.** Every mutating command holds a per-repository lock. A second
   gymrat process exits 2 — concurrent benchmarks perturb each other.

4. **Discard decisively.** A NO-SIGNAL or REGRESSED iteration that cannot be salvaged should be
   discarded immediately. Reworking without discarding first conflates the old and new changes.

5. **Finalize when the work is done.** The squash commit is the deliverable.

## Exit codes

| Code | Meaning                                                                                |
| ---- | -------------------------------------------------------------------------------------- |
| 0    | Success (report produced, no gate tripped)                                             |
| 1    | Gate tripped: `keep` refused, `iterate` hit a stop condition                           |
| 2    | Operational error: no session, finalized session, lock contention, bad config, timeout |

Exit 1 is information — read the output. Exit 2 is a real error — diagnose before retrying.
