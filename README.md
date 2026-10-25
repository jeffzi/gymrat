# gymrat

A command-line interface (CLI) that tells you whether a code change made your benchmarks faster,
judged by a statistical test on paired samples.

## What it does

Benchmark numbers jitter from run to run, so "it got 3% faster" is often noise. gymrat runs your
existing bench command against two or more revisions, pairs the samples, and judges every metric
with a statistical test before calling anything improved or regressed. It also drives a keep/discard
optimization loop, either by hand or under an artificial intelligence (AI) agent in a supervised
run with wall-clock and spend caps.

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

gymrat checks out each revision into a temporary worktree, runs the bench command 12 times per
side, and prints a verdict per metric plus a geometric-mean (geomean) summary:

```text
gymrat compare · baseline main ↔ perf/faster-decode · 12 paired samples · adapter: metric-lines
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
or `--format json` for machine-readable JavaScript Object Notation (JSON) output.

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

gymrat is not on the Python Package Index (PyPI) yet, so install it from the GitHub repository:

```console
uv tool install git+https://github.com/jeffzi/gymrat
```

From the first PyPI release, `uv tool install gymrat`, `pipx install gymrat`, or
`pip install gymrat` work too. Requires Python 3.12+ and git on your `PATH`.

## Feeding it metrics

gymrat parses your bench command's stdout through an adapter:

- **`metric-lines`** (default): your script prints one `METRIC <name>=<value>` line per sample,
  e.g. `METRIC decode#time=12000000`. gymrat reduces repeated names to their median. It treats names
  ending in `#time` as nanoseconds and names ending in `#heap` as bytes; any other name is a plain
  number where lower is better.
- **`mitata`**: parses [mitata](https://github.com/evanwashere/mitata) benchmark output directly.

Select one with `--adapter` or in the config file.

gymrat reads at most 64 MiB of each bench run's stdout and fails a run that prints more, rather
than parse output it cut short. Keep logging out of the code you measure.

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
| `gymrat keep [-m "<message>"]`              | Commit an improved iteration whose checks pass                             |
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
samples = 12 # paired samples per target; see How verdicts work for valid counts
timeout_seconds = 1800 # default; per bench invocation; flag --timeout, variable GYMRAT_TIMEOUT
filter = "npm run bench -- --filter {names}" # scopes confirmation reruns and `gymrat probe`
primary = "geomean" # default; or a metric name
checks = "npm test" # must pass before `gymrat keep` commits an iteration

[stop]
max_iterations = 20 # the loop stops after this many iterations
# target_value = 9000000 # stop once the primary reaches this and is kept; needs a metric primary

[metrics."decode#time"]
direction = "lower" # per-metric overrides: direction, gating, exact, resolution
resolution = 41.67 # one measurement step, in nanoseconds (ns): a timer that ticks at 24 MHz

[kinds.time]
resolution = 41.67 # every time metric that sets no step of its own
```

The file also accepts `runbook`, `unstable_noise_pct`, and per-kind `[kinds]` overrides of
`gating` and `resolution`, and a `[hooks]` table of shell commands that run around each
`gymrat iterate` (see [Hooks](#hooks)). Kinds are `time` (`#time` metrics), `memory` (`#heap`
metrics), and `other` (every other metric). gymrat names any unknown key or invalid value when it
reads the file; it ignores a `[kinds]` entry that matches no kind.

- `gating`, per metric or per kind, decides whether a regression on that metric fails
  `--fail-on regressed` and blocks `keep`.
- `unstable_noise_pct` (default 200, minimum 0.5) is the noise-band width, in percent, above which
  a metric is reported unstable.
- `runbook` names the runbook file, relative to the directory holding `gymrat.toml`. The file must
  exist.

`resolution` is the size of one measurement step, in the metric's unit: 41.67 for a timer that
ticks at 24 MHz and is read in nanoseconds. gymrat gives `#heap` metrics a built-in step of one
byte. A change of one step never signals, because the true change behind it can be anything from
nothing to two steps. Where the bench allows it, time more iterations per reading instead of
declaring a step: the step then shrinks relative to each reading, so smaller changes can signal.
gymrat warns when a time metric's readings look quantized and no step is declared. The gap the
warning names is the largest the step can be, so declare the timer's tick rather than the gap.

Precedence: command-line flag > `GYMRAT_*` environment variable (`GYMRAT_BENCH`, `GYMRAT_SAMPLES`,
…) > `gymrat.toml` > built-in default.

`gymrat probe` is the one exception. Its sample count comes from `--samples` or its own 6-sample
default; it ignores `GYMRAT_SAMPLES` and the configured `samples`. A probe is a spot check, so a
session tuned for 12-sample measurements should not make every spot check cost a full measurement.

The optional `[supervise]` table pins settings for `gymrat supervise` runs:

```toml
[supervise]
model = "sonnet" # agent model; alias or full model ID
effort = "high" # low | medium | high | xhigh | max
```

Both keys are optional. The `--model` and `--effort` flags on `supervise` override the configured
values.

## The optimization loop

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

`finalize` points a new branch at the squash commit: `gymrat/<session_id>-final` by default, or the
name you pass with `--branch <name>`. Your current branch and the session branch stay as they are.

While a session is open, `start`, `iterate`, `keep`, `discard`, `status`, `stop`, `sync`,
`finalize`, `probe`, `compare`, `measure`, and `supervise` each append a `command` record to
`.gymrat/session.jsonl`. The record holds the command's arguments, exit code, the reason for a
non-zero exit, duration, and `origin` (`cli` or `tool`). `status` takes the repository lock briefly
to record itself.

### Supervised runs

`gymrat supervise "optimize the decoder" --max-minutes 30 --max-usd 5` runs the loop under an AI
agent. The agent is Claude Code, driven through the Claude Agent software development kit
(SDK) that gymrat installs as a dependency. Claude Code must be able to authenticate: signed in,
or with `ANTHROPIC_API_KEY` set. `--max-usd` caps the spend Claude Code reports for the run. The
runbook scaffolded by `init` describes the goal and constraints.

`supervise` opens the session and records the baseline itself before handing control to the agent.
`--baseline <ref>` pins the session to a specific ref; it defaults to HEAD and is ignored when
resuming an open session. The wall-clock cap starts once the baseline is recorded, so a run may take
the cap plus the baseline's duration.

`--force` launches even when the cap cannot fit one iteration or a stop condition is already met.
`--allow-dirty` allows a launch with uncommitted changes, and `--log <path>` sets where the
supervisor event log, in JSON Lines (JSONL) format, goes.

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
run ends with a settled session and the `-final` branch that `finalize` creates.

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
MultiEdit, NotebookEdit) must land in the experiment worktree or a system temp directory outside the
repository. gymrat refuses any other path. It also refuses a gymrat command run in the
background. Each refusal names its rule. gymrat does not inspect shell commands for file writes.

A manual session has none of these refusals.

While a supervised run's wall-clock cap is active, `compare`, `measure`, `probe`, and every session
command print a time-left line, so the agent can plan around it. The session commands are `start`,
`iterate`, `keep`, `discard`, `status`, `stop`, `sync`, and `finalize`.

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

An `after` hook must not modify the experiment worktree: gymrat captures the worktree's contents when
it measures, so anything the hook changes afterwards goes unrecorded.

## Machine-readable output

`compare`, `measure`, `probe`, `doctor`, and every session command accept `--format json` for
structured output. The session commands are `start`, `iterate`, `keep`, `discard`, `status`,
`stop`, `sync`, and `finalize`.
`init`, `export`, and `supervise` are text-only. Text output is for humans and may change between
releases. From the first release published to PyPI, the log schema and the `--format json`
documents are additive-only. Each JSON document starts with a `schema_version` field.

The [event reference](https://github.com/jeffzi/gymrat/blob/main/docs/event-reference.md) documents
the session log and supervisor event log formats, with JSON Schema and AsyncAPI files in
[`schemas/`](https://github.com/jeffzi/gymrat/tree/main/schemas).

## Color control

Place `--color` or `--no-color` before or after any subcommand. A subcommand flag beats the
root flag. Without either, `FORCE_COLOR` forces styling, `NO_COLOR` suppresses it, and otherwise
gymrat styles output only when the stream is a terminal (TTY).

## Monitoring

Install the optional OpenTelemetry extra with the same tool you installed gymrat with. Then point
it at any collector that accepts OpenTelemetry Protocol (OTLP) over Hypertext Transfer Protocol
(HTTP), such as Jaeger, Grafana Tempo, or your own:

```console
uv tool install 'gymrat[otel] @ git+https://github.com/jeffzi/gymrat'
# from the first PyPI release: uv tool install 'gymrat[otel]' (or pipx / pip)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

With both in place, gymrat exports spans as they happen: for every command that appends a `command`
record and for every supervised run. One trace covers the whole session:

- **`gymrat.session`** — the root span, carrying the session identity.
- **`gymrat.run`** — one per supervised run, a child of the session span, with cost, duration, and
  model attributes.
- **`gymrat.command.<name>`** — one per CLI command (iterate, keep, compare, …), parented under the
  run span when the supervised agent runs the command, under the session span otherwise. The
  outcome records that the command appended to the session log appear as events on this span.

The [telemetry reference](https://github.com/jeffzi/gymrat/blob/main/docs/telemetry-reference.md)
lists every span and event attribute.

A command span also links to the trace context in its environment. For a command the supervised
agent runs, the link always targets the run span, which gymrat passes as `GYMRAT_TRACEPARENT`. For
any other command, `supervise` itself included, it targets whatever the `TRACEPARENT` environment
variable names.

`gymrat export` replays a session's logs into the same span structure after the fact:

```console
gymrat export                                               # the repo's current session log
gymrat export .gymrat/session-20240115-093000-1a2b.jsonl    # an earlier, archived session
gymrat export --endpoint http://localhost:4318              # override the collector URL
```

`gymrat start` archives the previous session's log as `.gymrat/session-<session_id>.jsonl`.

## How verdicts work

Each candidate is sampled in rounds with its baseline: every round runs each side once, so machine
drift hits both sides equally. The run order changes from round to round so each target runs in
every position, and after every other target, equally often. The sample count must be a multiple of
the order's cycle: an even count for two targets, a multiple of `N` for an even number `N` of
targets, and a multiple of `2N` for an odd number. The default of 12 balances 1, 2, 3, 4, 6
and 12 targets. gymrat refuses any other count before a bench or hook runs, and names the nearest valid
counts. `gymrat doctor` lists the target counts from 2 through 8 the configured `samples` balances.

A metric's verdict comes from a sign-flip permutation test on the signed-rank sum of the paired
differences. Tied pairs carry no sign to flip. When fewer than 6 pairs differ, a noise-band
fallback applies: the percentage change between the two sides' medians must exceed the metric's
noise band.

A metric's noise band is 1.5 times the larger of the two sides' half-ranges, never less than 0.5%.
A half-range is half the span from lowest to highest sample, taken as a percentage of that side's
median. A metric measured in whole steps must also move by more than 1.25 steps: its declared
`resolution`, or one byte for a `#heap` metric that declares none. Its band therefore never sits
below 1.25 steps as a share of the baseline median, and a change within those 1.25 steps counts as
noise whatever the pairs say. Setting `exact = true` on a metric skips the statistical test and the
noise band: any difference between its medians is a change.

Each metric lands in one class:

- **✓ improved** / **✗ regressed** — the metric moved in its better or worse direction.
- **≈ unstable** — the samples are too erratic to judge.
- **= identical** — every pair tied (an `exact` metric shows within noise instead).
- **~ within noise** — gymrat could not confirm a change. The change is indistinguishable from
  noise, falls within 1.25 measurement steps, or has no finite percentage. Or it runs against the
  pairs: the medians do not move the way the pairs do.
- **? inconclusive** — fewer than 6 pairs, too few for a statistical verdict.

When a metric lands within noise, gymrat also states the smallest change the run could have
detected for it, as a percentage of the baseline median. The text table prints it as
`detects ≥X%` after the noise band; the JSON output and the session log carry it as
`detectable_pct`. A real change smaller than that figure can go unseen at this sample count.

JSON output stores identical and within-noise as `no-signal`. An inconclusive verdict keeps the
value the noise band gave it (`improved`, `regressed`, `unstable`, or `no-signal`), with fewer than
6 pairs behind it. A `no-signal` verdict whose medians differ carries a `reason`, the first of these
that applies:

- `non-finite-delta` — the change has no finite percentage.
- `resolution` — the change falls within 1.25 measurement steps.
- `noise` — the noise band rejects the change, too few pairs differ, or the permutation test is
  not significant.
- `direction-mismatch` — the pairs move the other way from the medians.
- `unconfirmed` — a confirmation rerun did not reproduce the regression.

## Contributing

Install [Task](https://taskfile.dev), [uv](https://docs.astral.sh/uv/), and
[dprint](https://dprint.dev), then run `task install`. `task test` runs the test suite and
`task check` runs every lint and format hook. Report bugs on the
[issue tracker](https://github.com/jeffzi/gymrat/issues).

## License

[MIT](https://github.com/jeffzi/gymrat/blob/main/LICENSE)
