# gymrat

A CLI that tells you whether a code change made your benchmarks faster, judged by a statistical
test on paired samples.

## What it does

Benchmark numbers jitter from run to run, so "it got 3% faster" is often noise. gymrat runs your
existing bench command against two or more revisions, pairs the samples, and judges every metric
with a statistical test before calling anything improved or regressed. It also drives a keep/discard
optimization loop, either by hand or under an AI agent in a supervised run with time and spend caps.

## Contents

- [Quick example](#quick-example)
- [Choosing a workflow](#choosing-a-workflow)
- [Installation](#installation)
- [Feeding it metrics](#feeding-it-metrics)
- [Commands](#commands)
- [Configuration](#configuration)
- [The optimization loop](#the-optimization-loop)
- [Machine-readable output](#machine-readable-output)
- [Color control](#color-control)
- [Monitoring](#monitoring)
- [How verdicts work](#how-verdicts-work)
- [Contributing](#contributing)
- [License](#license)

## Quick example

```console
gymrat compare main perf/faster-decode --bench "npm run bench"
```

gymrat checks out each revision into a temporary worktree, runs the bench command ten times per
side, and prints a verdict per metric plus a geometric-mean (geomean) summary:

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

Add `--fail-on regressed` to make a continuous integration (CI) job exit non-zero on a regression,
or `--format json` for machine-readable output.

## Choosing a workflow

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

## Installation

```console
uv tool install gymrat
```

or `pipx install gymrat`, or `pip install gymrat`. Requires Python 3.12+.

## Feeding it metrics

gymrat parses your bench command's stdout through an adapter:

- **`metric-lines`** (default): your script prints one `METRIC <name>=<value>` line per sample,
  e.g. `METRIC decode#time=12000000`. gymrat reduces repeated names to their median. It treats names
  ending in `#time` as nanoseconds and names ending in `#heap` as bytes; any other name is a plain
  number where lower is better.
- **`mitata`**: parses [mitata](https://github.com/evanwashere/mitata) benchmark output directly.

Select one with `--adapter` or in the config file.

## Commands

| Command                                     | What it does                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `gymrat init`                               | Scaffold a `gymrat.toml`, an agent skill file, and a runbook               |
| `gymrat compare <baseline> <cand>…`         | Judge one or more candidates against a baseline                            |
| `gymrat measure [target]`                   | Measure a single revision or directory on its own                          |
| `gymrat doctor`                             | Check the project setup and report problems                                |
| `gymrat start [--baseline <ref>]`           | Pin the baseline and open a session                                        |
| `gymrat probe [names…]`                     | Spot-check the experiment worktree against the recorded baseline           |
| `gymrat iterate`                            | Measure the experiment worktree against the baseline                       |
| `gymrat keep -m "<message>"`                | Commit an improved iteration whose checks pass                             |
| `gymrat discard`                            | Revert the experiment worktree to its last commit                          |
| `gymrat status`                             | Print the session history                                                  |
| `gymrat sync`                               | Copy uncommitted main-tree edits into the experiment worktree              |
| `gymrat stop -m "<report>"`                 | Record a closing report in the session log (the session stays open)        |
| `gymrat finalize`                           | Squash the kept iterations into one commit and close the session           |
| `gymrat supervise [prompt] --max-minutes N` | Run the optimization loop under an AI agent with wall-clock and spend caps |
| `gymrat export [session-log]`               | Replay a session log's spans to an OpenTelemetry collector                 |

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
timeout_seconds = 1800 # per bench invocation; flag --timeout, variable GYMRAT_TIMEOUT
filter = "npm run bench -- --filter {names}" # scopes confirmation reruns and `gymrat probe`
primary = "geomean" # or a metric name
checks = "npm test" # must pass before `gymrat keep` commits an iteration

[stop]
max_iterations = 20 # the loop stops after this many iterations
# target_value = 9000000 # stop once the primary reaches this and is kept; needs a metric primary

[metrics."decode#time"]
direction = "lower" # per-metric overrides: direction, gating, exact
```

The file also accepts `runbook`, `unstable_noise_pct`, and per-kind `[kinds]` overrides. gymrat
names any unknown key or invalid value when it reads the file.

Precedence: command-line flag > `GYMRAT_*` environment variable (`GYMRAT_BENCH`, `GYMRAT_SAMPLES`,
…) > `gymrat.toml` > built-in default.

`gymrat probe` is the one exception. Its sample count comes from `--samples` or its own 6-sample
default; it ignores `GYMRAT_SAMPLES` and the configured `samples`. A probe is a spot check, so a
session tuned for 10-sample measurements should not make every spot check cost a full measurement.

The optional `[supervise]` table pins settings for `gymrat supervise` runs:

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
gymrat start --baseline main   # pin the baseline and open the session
# ...edit code in the experiment worktree...
gymrat probe decode#time   # bench the edit alone and print its delta; records no iteration
gymrat iterate             # measure the edit against the baseline
gymrat keep -m "vectorize decode loop"   # commit it if it improved and checks pass
gymrat keep --allow-unimproved           # ...or commit an unimproved iteration anyway
gymrat discard             # ...or revert the worktree to its last commit
gymrat status              # session history so far
gymrat sync                # copy uncommitted main-tree edits into the worktree
gymrat stop -m "done"      # record a closing report in the session log
gymrat finalize            # squash kept iterations into one commit and close
```

A committed `keep` advances the recorded baseline to the kept iteration's samples, so the next
iteration measures against what was just kept.

While a session is open, `start`, `iterate`, `keep`, `discard`, `status`, `stop`, `sync`,
`finalize`, `probe`, `compare`, and `measure` each append a `command` record to
`.gymrat/session.jsonl`. The record holds the command's arguments, exit code, the reason for a
non-zero exit, duration, and `origin` (`cli` or `tool`). `status` takes the repository lock briefly
to record itself.

### Supervised runs

`gymrat supervise "optimize the decoder" --max-minutes 30 --max-usd 5` runs the loop under an AI
agent. The agent is Claude Code, driven through the Claude Agent SDK that gymrat installs as a
dependency, so Claude Code must be able to authenticate: signed in, or with `ANTHROPIC_API_KEY`
set. `--max-usd` caps the spend Claude Code reports for the run. The runbook scaffolded by `init`
describes the goal and constraints.

`supervise` opens the session and records the baseline itself before handing control to the agent.
`--baseline <ref>` pins the session to a specific ref; it defaults to HEAD and is ignored when
resuming an open session. The wall-clock cap starts once the baseline is recorded, so a run may take
the cap plus the baseline's duration.

`--force` launches even when the cap cannot fit one iteration or a stop condition is already met.
`--allow-dirty` allows a launch with uncommitted changes, and `--log <path>` sets where the JSONL
event log goes.

#### Guards and caps

The run continues across an early turn end: the supervisor replies to the agent. The run ends on
`gymrat stop`, a stop condition, a failed hook, a cap, or a guard. Three guards stop a runaway loop:

- **Follow-up ceiling:** 100 replies.
- **No-progress limit:** 3 consecutive turns that append no iteration, keep, discard, or other
  outcome record to the session log.
- **Consecutive-discard limit:** 5 discards in a row.

Both the agent backend and the supervisor enforce `--max-usd`.

#### How a run ends

When the run ends, `supervise` settles the session before it prints the summary. It keeps an
improved iteration whose checks pass and discards one that did not improve. When the last gymrat
command has finished, nothing needs a person's decision, and at least one iteration was kept, the
run ends with a settled session and a squash branch.

Some iterations stay in the worktree for you: those whose checks failed, whose worktree changed
after measuring, or whose hook failed. Unmeasured edits stay too, and the summary names each one.
Pass `--no-finalize` to leave the session open.

Settling may overrun the wall-clock cap by up to two `timeout_seconds` periods: one waiting for a
running gymrat command, one for the checks. The git work of keeping, discarding, or finalizing adds
more time and has no timeout.

#### What the agent may do

The agent runs `probe` and `iterate` as Model Context Protocol (MCP) tools the supervisor hosts, so
a cap, a guard, or Ctrl-C stops a running measurement with the run. Every other gymrat command stays
a Bash command.

While a supervised run is live, gymrat refuses `iterate` and `probe` typed in a shell, so the agent
must use the tools. It refuses `init` outright.

gymrat also fences the agent's tool calls. An edit made through a file-editing tool (Edit, Write,
MultiEdit, NotebookEdit) must land in the experiment worktree or in a system temp directory outside
the repository; gymrat refuses any other path. It also refuses a gymrat command run in the
background. Each refusal names its rule. gymrat does not inspect shell commands for file writes.

A manual session has none of these refusals.

`iterate`, `keep`, `discard`, `status`, `stop`, `sync`, `compare`, `measure`, and `probe` print a
time-left line while a supervised run's wall-clock cap is active, so the agent can plan around it.

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
    "branch": "gymrat/20240115-093000-1a2b",
    "iteration_count": 0
  }
}
```

A `before` hook gets the previous iteration's record in `last_iteration`, or `null` on the first
iteration. An `after` hook gets the record gymrat just appended for this `seq`.

gymrat reports a hook's non-zero exit in the iteration output; the iteration still succeeds. Under
`supervise`, a failed hook ends the run and leaves the iteration it bracketed for a person to keep or
discard. gymrat kills a hook that runs longer than 30 seconds.

An after hook must not modify the experiment worktree: gymrat captures the worktree's contents when
it measures, so anything the hook changes afterwards goes unrecorded.

## Machine-readable output

`compare`, `measure`, `probe`, `doctor`, and the session commands (`start`, `iterate`, `keep`,
`discard`, `status`, `stop`, `sync`, `finalize`) accept `--format json` for structured output.
`init`, `export`, and `supervise` are text-only. Text output is for humans and may change between
releases.

Log records use snake_case keys, `at` timestamps are integer nanoseconds since the Unix epoch,
and the schema is additive-only from the first published release.

The session and supervisor log formats are documented in the
[event reference](https://github.com/jeffzi/gymrat/blob/main/docs/event-reference.md), with JSON
Schema and AsyncAPI files in
[`schemas/`](https://github.com/jeffzi/gymrat/tree/main/schemas).

## Color control

`--color` / `--no-color` can be placed before or after any subcommand. A subcommand flag beats the
root flag. Without either, `FORCE_COLOR` forces styling, `NO_COLOR` suppresses it, and otherwise
gymrat styles output only when the stream is a terminal (TTY).

## Monitoring

Install the optional OpenTelemetry extra with the same tool you installed gymrat with, then point it
at any collector that accepts OpenTelemetry Protocol (OTLP) over HTTP (Jaeger, Grafana Tempo, or
your own):

```console
uv tool install 'gymrat[otel]'   # or: pipx install 'gymrat[otel]' / pip install 'gymrat[otel]'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

With both in place, gymrat exports spans as they happen: for every command that appends a `command`
record and for every supervised run. One trace covers the whole session:

- **`gymrat.session`** — the root span, carrying the session identity.
- **`gymrat.run`** — one per supervised run, a child of the session span, with cost, duration, and
  model attributes.
- **`gymrat.command.<name>`** — one per CLI command (iterate, keep, compare, …), parented under the
  run span when the command runs inside `supervise`, under the session span otherwise. The outcome
  records that the command appended to the session log appear as events on this span.

When a supervised run has Claude Code's own tracing on, each command span links to the tool-call
span through the `TRACEPARENT` environment variable (`GYMRAT_TRACEPARENT` takes precedence).

`gymrat export` replays a session's logs into the same span structure after the fact:

```console
gymrat export                                               # the repo's current session log
gymrat export .gymrat/session-20240115-093000-1a2b.jsonl    # an earlier, archived session
gymrat export --endpoint http://localhost:4318              # override the collector URL
```

`gymrat start` archives the previous session's log as `.gymrat/session-<session_id>.jsonl`.

### Attribute reference

"`<name>` event" below means a span event named after the corresponding session-log or
supervisor-event-log record; see the [event reference](https://github.com/jeffzi/gymrat/blob/main/docs/event-reference.md)
for those record types.

| Attribute                      | Type   | Appears on                       | Description                                                                                                               |
| ------------------------------ | ------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `gymrat.session.id`            | string | session, run, command            | Unique identifier for the session.                                                                                        |
| `gymrat.session.branch`        | string | session                          | Git branch created for the session.                                                                                       |
| `gymrat.run.head_sha`          | string | run                              | Git HEAD commit SHA at launch time.                                                                                       |
| `gymrat.run.max_minutes`       | float  | run                              | Wall-clock timeout in minutes.                                                                                            |
| `gymrat.run.max_usd`           | float  | run (when set)                   | Spend cap in US dollars, if configured.                                                                                   |
| `gymrat.run.effort`            | string | run (when set)                   | Effort level, if specified.                                                                                               |
| `gymrat.run.cost_usd`          | float  | run (at end)                     | Cumulative cost of the run in US dollars.                                                                                 |
| `gymrat.run.ended_by`          | string | run (at end, live export only)   | How the run ended: `session`, `wall-clock`, `spend-cap`, `guard`, `stop-condition`, or `hook-failure`.                    |
| `gymrat.run.end_reason`        | string | run (when set, live export only) | For `guard`, the guard reason; for a cap, the cap name; for `stop-condition` and `hook-failure`, the condition's summary. |
| `gymrat.run.duration_ms`       | float  | run (at end, live export only)   | Wall-clock milliseconds the run took.                                                                                     |
| `gen_ai.request.model`         | string | run (when set)                   | Model requested for the run, if specified.                                                                                |
| `gen_ai.provider.name`         | string | run                              | Model provider (currently always `anthropic`).                                                                            |
| `gymrat.command.name`          | string | command                          | Name of the CLI command that ran.                                                                                         |
| `gymrat.command.exit_code`     | int    | command                          | Process-style exit code: 0 success, 1 or 2 failure.                                                                       |
| `gymrat.command.reason`        | string | command (when set)               | Why the command exited non-zero, when it did.                                                                             |
| `gymrat.command.duration_ms`   | int    | command                          | Wall-clock milliseconds the command took.                                                                                 |
| `gymrat.command.args.<key>`    | scalar | command (per arg)                | One attribute per argument the command was invoked with.                                                                  |
| `gymrat.iteration.seq`         | int    | command, events                  | Iteration sequence number the record belongs to.                                                                          |
| `gymrat.iteration.outcome`     | string | iteration event                  | Overall iteration outcome: improved, regressed, or no-signal.                                                             |
| `gymrat.iteration.delta_pct`   | float  | iteration event                  | Percentage change from baseline for the primary metric.                                                                   |
| `gymrat.turn.cost_usd`         | float  | turn_end event                   | Cost of the turn in US dollars.                                                                                           |
| `gymrat.turn.origin`           | string | turn_end event                   | Whether the turn was agent-generated or injected by the supervisor.                                                       |
| `gymrat.turn.budget_exhausted` | bool   | turn_end event                   | Whether the turn exhausted the remaining budget.                                                                          |
| `gymrat.follow_up.action`      | string | follow_up event                  | Supervisor action taken after the turn.                                                                                   |
| `gymrat.follow_up.reason`      | string | follow_up event                  | Reason for the action, if applicable.                                                                                     |
| `gymrat.cap.name`              | string | cap event                        | Which supervision cap fired.                                                                                              |
| `gymrat.keep.status`           | string | keep event                       | Whether the iteration was committed or blocked.                                                                           |
| `gymrat.keep.reason`           | string | keep event (when set)            | Why the keep was blocked, when status is blocked.                                                                         |
| `gymrat.keep.commit`           | string | keep event (when set)            | Git commit SHA when status is committed.                                                                                  |
| `gymrat.keep.message`          | string | keep event (when set)            | Commit message when status is committed.                                                                                  |
| `gymrat.hook.stage`            | string | hook event                       | Whether the hook ran before or after the iteration.                                                                       |
| `gymrat.hook.exit_code`        | int    | hook event                       | Process exit code of the hook command.                                                                                    |
| `gymrat.hook.duration_ms`      | float  | hook event                       | Wall-clock milliseconds the hook command ran.                                                                             |
| `gymrat.hook.stdout_bytes`     | int    | hook event                       | Bytes the hook command wrote to stdout.                                                                                   |
| `gymrat.hook.stderr_bytes`     | int    | hook event (when set)            | Bytes the hook command wrote to stderr.                                                                                   |
| `gymrat.hook.timed_out`        | bool   | hook event                       | Whether the hook command exceeded its timeout.                                                                            |
| `gymrat.baseline.label`        | string | baseline event                   | Human-readable label for the baseline measurement.                                                                        |
| `gymrat.baseline.duration_ms`  | float  | baseline event (when set)        | Wall-clock milliseconds the baseline measurement took.                                                                    |
| `gymrat.finalize.branch`       | string | finalize event                   | Git branch the finalize squash landed on.                                                                                 |
| `gymrat.finalize.commit`       | string | finalize event                   | Git commit SHA of the squash commit.                                                                                      |
| `gymrat.finalize.message`      | string | finalize event                   | Commit message of the squash commit.                                                                                      |
| `gymrat.stop.message`          | string | stop event                       | Reason the stop was requested.                                                                                            |

## How verdicts work

Each candidate is sampled in strict alternation with its baseline, so machine drift hits both
sides equally. A metric's verdict comes from a sign-flip permutation test on the paired deltas;
below the sample floor a noise-band fallback applies. Each metric lands in one class:

- **✓ improved** / **✗ regressed** — the metric moved in its better or worse direction.
- **≈ unstable** — the samples are too erratic to judge.
- **= identical** — every pair tied.
- **~ within noise** — the change stays inside the noise band.
- **? inconclusive** — fewer than 6 pairs, too few for a statistical verdict.

JSON output stores the last three classes as `no-signal`.

## Contributing

Install [Task](https://taskfile.dev), [uv](https://docs.astral.sh/uv/), and
[dprint](https://dprint.dev), then run `task install`. `task test` runs the test suite and
`task check` runs every lint and format hook. Report bugs on the
[issue tracker](https://github.com/jeffzi/gymrat/issues).

## License

[MIT](https://github.com/jeffzi/gymrat/blob/main/LICENSE)
