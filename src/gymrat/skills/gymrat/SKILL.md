---
name: gymrat
description: >-
  Use when driving a gymrat optimization session toward a performance target. Not for one-off
  benchmarking of refs or branches with no session open.
when_to_use: >-
  Also use when running gymrat start, gymrat iterate, gymrat keep, gymrat discard, gymrat finalize,
  gymrat status, gymrat supervise, gymrat sync, gymrat measure, or gymrat compare; when a repo has
  a gymrat.toml; when asked to optimize a benchmark toward a target or budget, or to probe an edit
  before spending an iteration; or on errors like "has not been settled", "Keep refused", or "Stop
  condition met".
---

# Driving a gymrat optimization session

Covers the full session lifecycle: start, iterate, settle, finalize. Violating the letter of these
rules is violating their spirit — there are no technicalities.

Every command runs from the repository root. Never `cd` into a worktree: from inside
`.gymrat/worktrees/*` gymrat takes the worktree for the repository and every session command fails
with "No session". Reach a worktree through `git -C <path>` or a path argument instead.

`bench` must be resolvable — from `gymrat.toml` or `--bench`. `gymrat.toml` is where `checks`,
`filter`, `primary`, `runbook`, `stop`, and `hooks` live. `adapter` defaults to `metric-lines`;
`samples` defaults to `10`. Without `checks`, `keep` commits with the gate off and only warns.

Load the per-repo runbook before your first edit. `gymrat start` prints the path. No runbook line
→ ask which metrics gate and what to optimize.

## Session lifecycle

### 1. Start the session

```sh
gymrat start [ref]       # ref defaults to HEAD
```

Pins the baseline and creates two worktrees: experiment (where you edit) and baseline (read-only).

### 2. Record the baseline

```sh
gymrat measure --record .gymrat/worktrees/baseline
```

Appends a baseline record with every metric's samples to the session log. Do this once, before
the first edit. It is the reference every probe (below) reads against until the first `keep`.

Time it: that is the cost of one measurement side. A full-scope probe costs the same; one
`iterate` costs twice it, plus a confirmation rerun on REGRESSED. Before the first edit, state
that cost in one line. When a wall-clock cap was given to you — in the kickoff prompt or the
runbook — also state how many `iterate` runs fit, and never launch one the cap cannot fit: a
measurement the cap kills records nothing, so report what the probes measured instead. Never
estimate elapsed time yourself; the timed measurement is the only clock you have.

### 3. The iteration cycle

Edit code **only in the experiment worktree** (the path `start` printed). Then:

```sh
gymrat iterate        # measure the edit, print the verdict
```

Never pass `--bench` or `--samples` to `iterate`. The CLI accepts them and `keep` will not refuse
the record, but unmeasured gating metrics slip through and the kept record is underpowered.

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

Text output shows the runbook path, the baseline medians, one line per iteration with its verdict
and delta, and whether the last iteration is unsettled. It does not carry the experiment column of
a kept `iterate` report — note that when it prints. `--format json` carries counts only: no
runbook path, no per-iteration history. Run it after every settle and after any interrupted
command.

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
`--max-minutes`. `--allow-dirty` permits uncommitted changes.

**Never run `gymrat supervise` yourself.** If this text is in your system prompt, you are the
supervised agent: a nested launch spawns a second agent, a second cap, and a second bill against
the same repository, and the supervise lock does not stop it.

## Keeping iterations cheap

`iterate` is a committed step, not a look. Every `iterate` benches **both worktrees** fresh at the
configured `samples` — 2 × `samples` bench runs, plus a confirmation rerun on a REGRESSED inexact
metric — appends an iteration record, spends one `stop.max_iterations` slot, runs the hooks, and
leaves the session unsettled: the next `iterate` is refused until `keep` or `discard` settles it,
and `discard` reverts the edit.

**Default loop: probe with `measure`, verify with one `iterate`.**

- **Probe.**

  ```sh
  gymrat measure .gymrat/worktrees/experiment --bench "<scoped cmd>" --samples 6
  ```

  Benches the experiment worktree alone — `samples` runs on one side — with the bench command
  narrowed to the benchmarks the edit targets. It records nothing and touches no session state,
  so the session stays settled.

  - **Always pass `--bench`.** Without it a probe runs the full suite and costs a whole
    measurement side. When the runbook gives no scoped command, derive one from the bench
    harness's own filtering before the first probe. Read the primary metric only when the scoped
    command emits it under its full-bench name. With a `geomean` primary, a scoped geomean is a
    different number: read the individual metrics the edit targets instead.
  - **Read the median against the reference.** Before the first `keep`, that is the recorded
    baseline (`gymrat status` shows it). After a `keep`, it is the experiment column of the kept
    `iterate` report, or re-record with `gymrat measure --record .gymrat/worktrees/baseline`.
  - **A probe answers "did the number move?"**, not "is it significant?" — that is what the
    final `iterate` is for.
  - **A rejected edit** is reworked in place or reverted by hand. `discard` only settles a
    measured iteration and refuses when nothing has been measured since the last `keep` or
    `discard`. The revert below destroys the edit with no way back, so save anything worth
    keeping first (`git -C .gymrat/worktrees/experiment diff > <file>`), and run it from the
    repository root with both `-C` paths spelled out:

    ```sh
    git -C .gymrat/worktrees/experiment reset --hard && git -C .gymrat/worktrees/experiment clean -fd
    ```

- **Scoped verdict**, only when a probe is too close to call:

  ```sh
  gymrat compare .gymrat/worktrees/baseline .gymrat/worktrees/experiment \
    --bench "<scoped cmd>" --samples 6
  ```

  runs the same significance test as `iterate` with no record and no hooks, at 2 × `samples` runs
  on the scoped bench — twice a probe — so it is the exception, not the loop.
- **Verify.** One `gymrat iterate` — full bench, configured samples — when the edit looks done,
  immediately before `keep`. Never `keep` off anything else.

Levers, in order of leverage:

- **`--samples 6` on probes, never below.** The significance test needs 6 differing pairs, and 6
  is the smallest count that can reach significance at all, so below 6 a `compare` verdict falls
  back to a coarse noise band and a NO-SIGNAL at 3 samples means nothing. 6 is the knife edge: one
  tied pair drops the run to the band while a verdict still prints, so on metrics that repeat
  readings (integer counts, exact metrics) use more and read the method with `--verbose`. On
  `measure`, below 6 one slow run swings the spread past most real effects. Use the flag; never
  edit `samples` in `gymrat.toml` mid-session.
- **`filter` in `gymrat.toml`** is a bench command template carrying a `{names}` placeholder. It
  scopes **confirmation reruns only**: the rerun benches just the regressed metric names substituted
  into `{names}`. Without it, the whole bench re-runs to confirm. It never narrows the first
  measurement pass.
- **Batch related edits** into one iteration. Probe each, `iterate` once. A full measurement per
  micro-edit burns the session budget on measurement, not optimization.

## Machine-readable output

`measure`, `compare`, `iterate`, `keep`, `discard`, `status`, and `doctor` accept `--format json`.
When driving the loop programmatically, always pass `--format json`. The JSON contract is
additive-only (no renames or removals without a breaking change); the text report may change
between releases. `start`, `finalize`, and `sync` are text-only: their outputs are one-shot
summaries agents don't parse.

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

3. **Never run concurrent sessions.** Every command that runs the bench — `measure` and `compare`
   included — or mutates the session holds a per-repository lock, and a second gymrat process
   exits 2. `supervise` holds a separate lock that does not block a nested `supervise`; see
   Supervised mode.

4. **Discard decisively.** A NO-SIGNAL or REGRESSED iteration that cannot be salvaged gets discarded
   immediately — reworking without discarding first conflates the changes.

## Rationalizations

| Excuse                                                           | Reality                                                                                                                          |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| "The full suite is a safer probe"                                | It costs a whole measurement side to answer what a scoped `--bench` answers.                                                     |
| "This probe is close enough; `iterate` is faster than reprobing" | `iterate` costs 2 × a probe, spends a `max_iterations` slot, and leaves the session unsettled. Reprobe with a tighter `--bench`. |
| "`--samples 3` is enough to see the direction"                   | Below 6 the verdict is a noise band; a NO-SIGNAL at 3 means nothing.                                                             |
| "The cap is close, but `iterate` might just make it"             | A measurement the cap kills records nothing. Report what the probes measured.                                                    |
| "Two NO-SIGNALs; the target is unreachable"                      | Report only after sustained NO-SIGNAL, and only when no configured stop condition is still unfired.                              |
| "NO-SIGNAL, but the code is cleaner, so keep it"                 | NO-SIGNAL means no measurable effect. `keep` needs IMPROVED.                                                                     |
| "A quick `cd` into the worktree to look around"                  | The cwd sticks; every gymrat command after it reports "No session". Use `git -C`.                                                |
| "I'll rework the edit on top of the failed one"                  | Discard first; reworking without discarding conflates the changes.                                                               |

## Red flags — stop and re-read the rule

- About to pass `--bench` or `--samples` to `iterate`.
- About to run `measure` on the experiment worktree without `--bench`.
- About to run a second `iterate` before `keep` or `discard`.
- About to launch `gymrat supervise` while this text is in your system prompt.
- Drafting a stop report while a configured stop condition has not fired.
- Estimating elapsed time yourself instead of reading it off a timed measurement.

## Exit codes

| Code    | Meaning                                                                                |
| ------- | -------------------------------------------------------------------------------------- |
| 0       | Success (report produced, no gate tripped)                                             |
| 1       | Gate tripped: `keep` refused, `iterate` hit stop condition, `supervise` cap fired      |
| 2       | Operational error: no session, finalized session, lock contention, bad config, timeout |
| 128 + N | Killed by signal N (e.g. 130 after Ctrl-C); cleanup ran before exiting                 |

Exit 1 is information: read the output. Exit 2 is a real error: diagnose before retrying. An exit
above 128 means the run was interrupted. The iteration may be unsettled; check `gymrat status`.
