# gymrat

A CLI that measures whether a code change actually made your benchmarks faster — with statistics,
not vibes.

## What it does

Benchmark numbers jitter from run to run, so "it got 3% faster" is often noise. gymrat runs your
existing bench command against two or more revisions, pairs the samples, and judges every metric
with a statistical test before calling anything improved or regressed. It also drives a keep/discard
optimization loop — manually or under a supervised AI agent session with time and spend caps.

## Quick example

```console
gymrat compare main perf/faster-decode --bench "npm run bench"
```

gymrat checks out each revision into a temporary worktree, runs the bench command ten times per
side, and prints a verdict per metric plus a geomean summary:

```text
gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: metric-lines
metric                    │ main         │ perf/faster-decode │ vs main
──────────────────────────┼──────────────┼────────────────────┼────────────────
decode#time               │ 12.0ms ± 0%  │ 11.0ms ± 0%        │ ✓  -8.3%  ±0.6%
──────────────────────────┼──────────────┼────────────────────┼────────────────
geomean (1 stable metric) │              │                    │    -8.3%  ±0.6%

✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive

highlights
  ✓ decode#time   -8.3%
```

Add `--fail-on regressed` to make CI exit non-zero on a regression, or `--format json` for
machine-readable output.

## Installation

```console
uv tool install gymrat
```

or `pipx install gymrat`, or `pip install gymrat`. Requires Python 3.12+.

## Feeding it metrics

gymrat parses your bench command's stdout through an adapter:

- **`metric-lines`** (default): your script prints one `METRIC <name>=<value>` line per sample,
  e.g. `METRIC decode#time=12000000`. Repeated names are reduced to their median. Names ending in
  `#time` are treated as nanoseconds and names ending in `#heap` as bytes; anything else is a plain
  number where lower is better.
- **`mitata`**: parses [mitata](https://github.com/evanwashere/mitata) benchmark output directly.

Select one with `--adapter` or in the config file.

## Commands

| Command                             | What it does                                                        |
| ----------------------------------- | ------------------------------------------------------------------- |
| `gymrat init`                       | Scaffold a `gymrat.toml`, an agent skill file, and a runbook        |
| `gymrat compare <baseline> <cand>…` | Judge one or more candidates against a baseline                     |
| `gymrat measure [target]`           | Measure a single revision or directory on its own                   |
| `gymrat doctor`                     | Check the project setup and report problems                         |
| `gymrat start` … `gymrat finalize`  | The optimization loop (below)                                       |
| `gymrat stop -m "<report>"`         | Record a closing report in the session log (the session stays open) |
| `gymrat sync`                       | Copy uncommitted main-tree edits into the experiment worktree       |
| `gymrat supervise "<prompt>"`       | Run a supervised agent session with wall-clock and spend caps       |
| `gymrat export [session-log]`       | Replay a finished session's spans to an OpenTelemetry collector     |

Targets are git refs or directories, optionally labeled: `gymrat compare old=main new=perf/simd`.
Every command takes `-h` for its full options.

## Configuration

`gymrat init --bench "npm run bench"` writes a `gymrat.toml` so you stop repeating flags. Common
keys:

```toml
bench = "npm run bench" # required: the command whose stdout carries metrics
prepare = "npm ci && npm run build" # run once per revision before sampling
adapter = "metric-lines" # or "mitata"
samples = 10 # paired samples per target
timeout_seconds = 1800 # per bench invocation
primary = "geomean" # or a metric name

[metrics."decode#time"]
direction = "lower" # per-metric overrides: direction, gating, exact
```

Precedence: command-line flag > `GYMRAT_*` environment variable (`GYMRAT_BENCH`, `GYMRAT_SAMPLES`,
…) > `gymrat.toml` > built-in default.

The optional `[supervise]` table pins settings for `gymrat supervise` sessions:

```toml
[supervise]
model = "sonnet" # agent model; alias or full model ID
effort = "high" # low | medium | high | xhigh | max
```

Both keys are optional. The `--model` and `--effort` flags on `supervise` override the configured
values.

## The optimization loop

For iterating on performance work, gymrat manages a session with a pinned baseline and an
experiment worktree:

```console
gymrat start main          # pin the baseline and open the session
# ...edit code in the experiment worktree...
gymrat iterate             # measure the edit against the baseline
gymrat keep -m "vectorize decode loop"   # commit it if checks pass
gymrat discard             # ...or revert the worktree to its last commit
gymrat status              # session history so far
gymrat sync                # copy uncommitted main-tree edits into the worktree
gymrat stop -m "done"      # record a closing report in the session log
gymrat finalize            # squash kept iterations into one commit and close
```

Every command run inside a session appends a `command` record to `.gymrat/session.jsonl` with its
arguments, exit code, refusal reason, and duration; `status` takes the repository lock briefly to
record itself.

`gymrat supervise "optimize the decoder" --max-minutes 30 --max-usd 5` runs that loop under an AI
agent. The runbook scaffolded by `init` describes the goal and constraints. The session continues
across an early turn end — the supervisor replies and the run ends on `gymrat stop`, a stop
condition, a cap, or a guard. Three guards protect against runaway loops: the
follow-up ceiling (100 replies), the no-progress limit (3 consecutive turns
with no improvement), and the consecutive-discard limit (5 discards in a row).
Both the agent backend and the supervisor enforce `--max-usd`.

`supervise` opens the session and records the baseline itself before handing control to the agent;
`--baseline <ref>` pins the session to a specific ref (default HEAD; ignored when resuming an open
session). The wall-clock cap starts once the baseline is recorded, so a run may take the cap plus
the baseline's duration.

`iterate`, `keep`, `discard`, `status`, `sync`, `compare`, and `measure` print a time-left line so
the agent can plan around the wall-clock cap.

### Hooks

The `[hooks]` table in `gymrat.toml` runs shell commands around each `gymrat iterate` measurement.
No other command runs them.

```toml
[hooks]
before = "npm run build" # once per iteration, before measuring
after = "./scripts/notify.sh" # after the iteration record is written
```

Hooks run with the experiment worktree as their working directory and receive a JSON object on
stdin:

```json
{
  "stage": "before",
  "experiment_dir": "/path/to/experiment-worktree",
  "seq": 1,
  "last_iteration": null,
  "session": {
    "session_id": "20240115-093000-1a2b",
    "baseline": { "ref": "main", "sha": "4f2a1c9b8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a" },
    "branch": "perf/faster-decode",
    "iteration_count": 0
  }
}
```

A `before` hook gets the previous iteration's record in `last_iteration`, or `null` on the first
iteration. An `after` hook gets the record gymrat just appended for this `seq`.

gymrat reports a hook's non-zero exit in the iteration output; the iteration still succeeds.
gymrat kills a hook that runs longer than 30 seconds.

An after hook must not modify the experiment worktree: gymrat captures the worktree's contents when
it measures, so anything the hook changes afterwards goes unrecorded.

## Two workflows, one tool

gymrat serves two audiences with the same statistical engine:

- **One-shot comparisons** — `gymrat compare` and `gymrat measure` answer a point-in-time question
  ("is this branch faster?") and clean up after themselves. Use these in CI gates, code reviews, or
  any time you have specific revisions to judge.

- **Iterative optimization** — `gymrat start` through `gymrat finalize` manage a session with a
  pinned baseline and an experiment worktree, so you can iterate on performance work with
  keep/discard decisions backed by statistics. `gymrat supervise` automates the same loop under an
  AI agent.

The principles behind these choices, and what gymrat deliberately is not, are in
[VISION.md](https://github.com/jeffzi/gymrat/blob/main/VISION.md).

## Machine-readable output

Every comparison, measurement, and session-loop command (`iterate`, `keep`, `discard`, `stop`,
`status`) accepts `--format json` for structured output. Text output is for humans and may change
between releases. `start`, `sync`, `finalize`, `export`, and `supervise` are text-only.

Log records use snake_case keys, `at` timestamps are integer nanoseconds since the Unix epoch,
and the schema is additive-only from the first published release.

The session and supervisor log formats are documented in the
[event reference](https://github.com/jeffzi/gymrat/blob/main/docs/event-reference.md), with JSON
Schema and AsyncAPI files in
[`schemas/`](https://github.com/jeffzi/gymrat/tree/main/schemas).

## Monitoring

Install the optional OpenTelemetry extra and point it at any collector that accepts OpenTelemetry
Protocol (OTLP) over HTTP (Jaeger, Grafana Tempo, or your own):

```console
pip install gymrat[otel]
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

With both in place, gymrat exports spans as they happen — for every command run inside a session and
every supervised run. One trace covers the whole session:

- **`gymrat.session`** — the root span, carrying the session identity.
- **`gymrat.run`** — one per supervised run, a child of the session span, with cost, duration, and
  model attributes.
- **`gymrat.command.<name>`** — one per CLI command (iterate, keep, compare, …), parented under the
  run span when the command runs inside `supervise`, under the session span otherwise. The outcome
  records that the command appended to the session log appear as events on this span.

When a supervised session has Claude Code's own tracing on, each command span links to the
tool-call span through the `TRACEPARENT` header.

`gymrat export` replays a finished session's logs into the same span structure after the fact:

```console
gymrat export                          # current repo's session
gymrat export path/to/session.jsonl    # explicit log path
gymrat export --endpoint http://...    # override the collector URL
```

### Attribute reference

"`<name>` event" below means a span event named after the corresponding session-log or
supervisor-event-log record; see the [event reference](https://github.com/jeffzi/gymrat/blob/main/docs/event-reference.md)
for those record types.

| Attribute                      | Type   | Appears on                | Description                                                           |
| ------------------------------ | ------ | ------------------------- | --------------------------------------------------------------------- |
| `gymrat.session.id`            | string | session, run, command     | Unique identifier for the session.                                    |
| `gymrat.session.branch`        | string | session                   | Git branch created for the session.                                   |
| `gymrat.run.head_sha`          | string | run                       | Git HEAD commit SHA at launch time.                                   |
| `gymrat.run.max_minutes`       | float  | run                       | Wall-clock timeout in minutes.                                        |
| `gymrat.run.max_usd`           | float  | run (when set)            | Spend cap in US dollars, if configured.                               |
| `gymrat.run.effort`            | string | run (when set)            | Effort level, if specified.                                           |
| `gymrat.run.cost_usd`          | float  | run (at end)              | Cumulative cost of the run in US dollars.                             |
| `gymrat.run.ended_by`          | string | run (at end)              | Whether the run ended on its own (`session`) or was stopped by a cap. |
| `gymrat.run.end_reason`        | string | run (when set)            | For a guard, the guard reason; for a cap, the cap name.               |
| `gymrat.run.duration_ms`       | float  | run (at end)              | Wall-clock milliseconds the run took.                                 |
| `gen_ai.request.model`         | string | run (when set)            | Model requested for the run, if specified.                            |
| `gen_ai.provider.name`         | string | run                       | Model provider (currently always `anthropic`).                        |
| `gymrat.command.name`          | string | command                   | Name of the CLI command that ran.                                     |
| `gymrat.command.exit_code`     | int    | command                   | Process-style exit code: 0 success, 1 or 2 failure.                   |
| `gymrat.command.reason`        | string | command (when set)        | Why the command exited non-zero, when it did.                         |
| `gymrat.command.duration_ms`   | int    | command                   | Wall-clock milliseconds the command took.                             |
| `gymrat.command.args.<key>`    | scalar | command (per arg)         | One attribute per argument the command was invoked with.              |
| `gymrat.iteration.seq`         | int    | command, events           | Iteration sequence number the record belongs to.                      |
| `gymrat.iteration.outcome`     | string | iteration event           | Overall iteration outcome: improved, regressed, or no-signal.         |
| `gymrat.iteration.delta_pct`   | float  | iteration event           | Percentage change from baseline for the primary metric.               |
| `gymrat.turn.cost_usd`         | float  | turn_end event            | Cost of the turn in US dollars.                                       |
| `gymrat.turn.origin`           | string | turn_end event            | Whether the turn was agent-generated or injected by the supervisor.   |
| `gymrat.turn.budget_exhausted` | bool   | turn_end event            | Whether the turn exhausted the remaining budget.                      |
| `gymrat.follow_up.action`      | string | follow_up event           | Supervisor action taken after the turn.                               |
| `gymrat.follow_up.reason`      | string | follow_up event           | Reason for the action, if applicable.                                 |
| `gymrat.cap.name`              | string | cap event                 | Which supervision cap fired.                                          |
| `gymrat.keep.status`           | string | keep event                | Whether the iteration was committed or blocked.                       |
| `gymrat.keep.reason`           | string | keep event (when set)     | Why the keep was blocked, when status is blocked.                     |
| `gymrat.keep.commit`           | string | keep event (when set)     | Git commit SHA when status is committed.                              |
| `gymrat.keep.message`          | string | keep event (when set)     | Commit message when status is committed.                              |
| `gymrat.hook.stage`            | string | hook event                | Whether the hook ran before or after the iteration.                   |
| `gymrat.hook.exit_code`        | int    | hook event                | Process exit code of the hook command.                                |
| `gymrat.hook.duration_ms`      | float  | hook event                | Wall-clock milliseconds the hook command ran.                         |
| `gymrat.hook.stdout_bytes`     | int    | hook event                | Bytes the hook command wrote to stdout.                               |
| `gymrat.hook.stderr_bytes`     | int    | hook event (when set)     | Bytes the hook command wrote to stderr.                               |
| `gymrat.hook.timed_out`        | bool   | hook event                | Whether the hook command exceeded its timeout.                        |
| `gymrat.baseline.label`        | string | baseline event            | Human-readable label for the baseline measurement.                    |
| `gymrat.baseline.duration_ms`  | float  | baseline event (when set) | Wall-clock milliseconds the baseline measurement took.                |
| `gymrat.finalize.branch`       | string | finalize event            | Git branch the finalize squash landed on.                             |
| `gymrat.finalize.commit`       | string | finalize event            | Git commit SHA of the squash commit.                                  |
| `gymrat.finalize.message`      | string | finalize event            | Commit message of the squash commit.                                  |
| `gymrat.stop.message`          | string | stop event                | Reason the stop was requested.                                        |

## How verdicts work

Each candidate is sampled in strict alternation with its baseline, so machine drift hits both
sides equally. A metric's verdict comes from a sign-flip permutation test on the paired deltas;
below the sample floor a noise-band fallback applies, and changes inside the noise band are
reported as `no-signal` rather than celebrated. Erratic metrics come back `unstable` instead of
misleading you.

## License

[MIT](https://github.com/jeffzi/gymrat/blob/main/LICENSE)
