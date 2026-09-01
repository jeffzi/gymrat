---
name: gymrat
description: >-
  Use when driving a gymrat optimization session toward a performance target. Not for one-shot
  comparisons (gymrat compare) or standalone measurements (gymrat measure without --record).
when_to_use: >-
  Also use when running gymrat start, gymrat iterate, gymrat keep, gymrat discard, gymrat finalize,
  gymrat status, gymrat supervise, gymrat sync, or gymrat measure --record; when a repo has a
  gymrat.toml; when asked to optimize a benchmark toward a target or budget; or on errors like "has
  not been settled", "Keep refused", or "Stop condition met".
---

# Driving a gymrat optimization session

Covers the full session lifecycle: start, iterate, settle, finalize. Every command runs from the
repository root.

`bench` must be resolvable — from `gymrat.toml` or `--bench`. `gymrat.toml` is where `checks`,
`filter`, `primary`, `runbook`, `stop`, and `hooks` live. `adapter` defaults to `metric-lines`;
`samples` defaults to `10`.

Load the per-repo runbook before your first edit. `gymrat start` prints the path; `gymrat status`
text output also shows it. The JSON status document carries counts only, not the runbook path or
per-iteration history. No runbook line → ask which metrics gate and what to optimize.

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

Appends a baseline record to the session log. Do this once, before the first edit, when the runbook
asks for it.

### 3. The iteration cycle

Edit code **only in the experiment worktree** (the path `start` printed). Then:

```sh
gymrat iterate        # measure the edit, print the verdict
```

Read the verdict block:

- **IMPROVED** — the primary metric moved in the right direction.
- **REGRESSED** — a gating metric regressed. For inexact metrics, a rerun confirms the regression;
  exact metrics gate without a rerun. A metric absent from the rerun still gates.
- **NO-SIGNAL** — the change had no measurable effect.

Then settle:

```sh
gymrat keep -m "describe the optimization"   # commit the edit, advance the baseline
gymrat discard                                # revert the experiment worktree
```

`keep` refuses when nothing has been measured, when the measured iteration left nothing to commit
(no edit was made), when a gating metric regressed, or when `checks` fails. Refusals exit 1.

After a checks failure, fix and re-run `gymrat keep`. After a gating-regression refusal, `keep`
stays blocked — run `iterate` or `discard`. A nothing-to-commit refusal settles the iteration:
edit the worktree, then run `gymrat iterate` — not `keep` again.

**One iteration at a time.** Each must be settled before the next `iterate`.

### 4. Read history

```sh
gymrat status
```

### 5. Close the session

```sh
gymrat finalize [-m "squash message"] [--branch <name>]
```

Collapses kept iterations into one squash commit on a new branch (default `<session-branch>-final`).
The squash commit is the deliverable. Requires every iteration settled, at least one keep, and a
clean experiment worktree. A finalized session refuses all mutating commands.

### Supervised mode

An alternative to the manual iteration cycle (steps 3-5): an agent drives `iterate`/`keep`/`discard`
on its own instead of you running them by hand.

```sh
gymrat supervise [prompt] --max-minutes <n> [--max-usd <n>] [--log <path>] [--model <name>]
```

Launches an agent to drive the session loop autonomously. Requires `runbook` in `gymrat.toml` and
`--max-minutes`. `--allow-dirty` permits uncommitted changes. Holds its own lock, separate from the
session lock.

## Keeping iterations cheap

Every iteration benches **both worktrees** — baseline and experiment are re-measured fresh, at
`samples` runs each, so one `iterate` costs 2 × `samples` bench invocations — and a REGRESSED
verdict on an inexact metric adds a confirmation rerun on top. Budget wall-clock accordingly.

**Default loop: probe cheap, verify full.** Explore with scoped, low-cost runs — `iterate --bench`
on the benchmarks the edit targets, `--samples 6` — and save the full bench at the full configured
sample count (default 10) for one final `iterate` when the change looks done, immediately before
`keep`. Never `keep` off a scoped or reduced-sample run. Levers, in order of leverage:

- **`iterate --samples N`** (`-s`) overrides the sample count for that run only. **Probe at 6,
  never below.** The significance test needs at least 6 differing pairs to run, and 6 is the
  smallest count that can reach significance at all — and only for a clean, consistent effect,
  which is exactly what a probe is for. Below 6 the verdict falls back to a coarse noise band:
  IMPROVED and REGRESSED still fire, but only when the delta is wider than the measured spread, so
  a NO-SIGNAL at 3 samples is nearly meaningless. Prefer the flag over editing `samples` in
  `gymrat.toml` mid-session.
- **`filter` in `gymrat.toml`** is a bench command template carrying a `{names}` placeholder. It
  scopes **confirmation reruns only**: the rerun benches just the regressed metric names substituted
  into `{names}`. Without it, the whole bench re-runs to confirm. It never narrows the first
  measurement pass.
- **`iterate --bench <cmd>`** (`-b`) swaps in a scoped bench command for one run, useful for probing
  a single benchmark. Metrics the scoped command does not emit go unmeasured — the reason a scoped
  run can never back a `keep`.
- **Batch related edits** into one iteration. A bench run per micro-edit burns the session budget on
  measurement, not optimization.

## Machine-readable output

`iterate`, `keep`, `discard`, and `status` accept `--format json`. When driving the loop
programmatically, always pass `--format json`. The JSON contract is additive-only (no renames or
removals without a breaking change); the text report may change between releases. `start` and
`finalize` are text-only (their outputs are one-shot summaries agents don't parse).

## Syncing main-tree edits

```sh
gymrat sync
```

Edits belong in the experiment worktree (`gymrat start` prints the path). When code must be changed
in the main working tree first (e.g. a dependency update), use `gymrat sync` to copy uncommitted
main-tree changes into the experiment worktree before running `iterate`. `sync` refuses when the
experiment worktree has conflicting uncommitted changes.

## Loop discipline

1. **Never stop before a stop condition fires.** When `stop.max_iterations` or `stop.target_value` is
   configured, keep iterating until `iterate` exits 1 naming the condition. Without `stop`, the
   runbook's goal is the criterion. Report and stop when a target proves unreachable after sustained
   NO-SIGNAL.

2. **When a hook fails, report the failure and pause for the user to decide.** Hooks cannot fail
   the loop, so a silently-ignored failure reaches `keep` unnoticed.

3. **Never run concurrent sessions.** Every mutating command holds a per-repository lock;
   `supervise` holds its own separate lock. A second gymrat process exits 2.

4. **Discard decisively.** A NO-SIGNAL or REGRESSED iteration that cannot be salvaged gets discarded
   immediately — reworking without discarding first conflates the changes.

## Common mistakes

- Forgetting `checks` in `gymrat.toml` — `keep` commits with the gate off and only warns.
- Treating NO-SIGNAL as success — it means the change had no measurable effect.
- Running the full bench for exploratory edits — probe scoped (`--bench`) at `--samples 6`; the
  full run comes once, right before `keep`.
- `keep` off a scoped or reduced-sample `iterate` — `keep` will not refuse it, but unmeasured
  gating metrics slip through and the kept record is underpowered. The last `iterate` before
  `keep` is full-bench at full samples.

## Exit codes

| Code    | Meaning                                                                                |
| ------- | -------------------------------------------------------------------------------------- |
| 0       | Success (report produced, no gate tripped)                                             |
| 1       | Gate tripped: `keep` refused, `iterate` hit stop condition, `supervise` cap fired      |
| 2       | Operational error: no session, finalized session, lock contention, bad config, timeout |
| 128 + N | Killed by signal N (e.g. 130 after Ctrl-C); cleanup ran before exiting                 |

Exit 1 is information: read the output. Exit 2 is a real error: diagnose before retrying. An exit
above 128 means the run was interrupted. The iteration may be unsettled; check `gymrat status`.
