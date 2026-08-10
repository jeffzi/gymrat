# gymrat

[![npm version](https://img.shields.io/npm/v/gymrat)](https://www.npmjs.com/package/gymrat)
[![Continuous integration build status](https://github.com/jeffzi/gymrat/actions/workflows/ci.yml/badge.svg)](https://github.com/jeffzi/gymrat/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/jeffzi/gymrat/blob/main/LICENSE)

Standalone A/B benchmark runner with paired sampling and benchstat-style reports.

`gymrat compare` runs a benchmark command against a baseline revision and one or more candidates,
cycling samples across them so every target sees the same machine noise, and prints a report that
tells you, per metric, whether each candidate is a real improvement, a real regression, or noise.
Verdicts come from a two-sided Wilcoxon signed-rank test once there are enough samples, and from a
noise band below that. `gymrat measure` runs the same sampling against a single target and reports
its figures with nothing to compare them to. No session state, no config required to start.

## Install

Requires Node ≥ 22 and `git` on your `PATH`.

```sh
npm install -g gymrat
```

Then, from inside the project you want to benchmark:

```sh
gymrat compare main my-branch --bench "npm run bench"
```

gymrat resolves git refs in the current repository, so run it from inside your project.

## Usage

```text
gymrat compare [label=]<baseline> [label=]<candidate>... [options]
```

The first positional is the baseline; every later positional is a candidate, and each one is judged
against the baseline alone — candidates are never compared with each other. Deltas are measured
against the baseline (the report's `vs <baseline>` column). Each target is either a path to an
existing directory (used in place, never removed) or a git ref that gymrat resolves with
`git rev-parse` and checks out into a temporary detached worktree. An existing directory wins over a
git ref of the same name, so prefix the ref with `refs/heads/` to disambiguate.

An optional `label=` prefix sets the display name. Without it, a git target is labelled with its
ref and a path target with the directory's base name, resolved through symlinks. Pass `label=`
when two paths share a base name. The prefix splits at the first `=`, so `label=a=b` passes a
target containing `=`; a bare `a=b` without a label cannot.

```sh
# Compare two git refs
gymrat compare main perf/faster-decode --bench "npm run bench"

# Judge several candidates against one baseline
gymrat compare main perf/simd perf/lookup-table --bench "npm run bench"

# Label the columns
gymrat compare old=main new=perf/faster-decode --bench "npm run bench"

# Build each revision before benchmarking, take 20 samples, parse mitata JSON
gymrat compare main my-branch \
  --prepare "npm ci && npm run build" \
  --adapter mitata \
  --samples 20
```

### `measure`

```text
gymrat measure [[label=]<ref|dir>] [options]
```

The target is optional and defaults to the current directory; when given, it is a git ref or
directory path resolved the same way as a `compare` target, with the same optional `label=` prefix.
`measure` accepts the shared options — `--bench`, `--prepare`, `--adapter`, `--samples`,
`--timeout`, `--config`, `--format`, and `--no-color` — but not `--verbose` or `--fail-on`: there is
no baseline for `--verbose` to name a verdict method against, and no candidate for `--fail-on` to
gate.

`-r, --record` appends the run to the repository's session log as a baseline record, so a later
`gymrat status` reads it back alongside the session's iterations. It requires a session (see
[The session loop](#the-session-loop)); the session is resolved before the first sample is taken, so
a run with nowhere to write fails before it benches rather than after.

```sh
# Measure the current directory
gymrat measure --bench "npm run bench"

# Measure a git ref, labelled
gymrat measure release=v2.0.0 --bench "npm run bench" --adapter mitata

# Record the measurement in the session log
gymrat measure --bench "npm run bench" --record
```

### The session loop

`compare` and `measure` answer one question and forget it. The session loop is the stateful path:
gymrat pins a baseline, gives you an experiment worktree to edit, and keeps a log of every
measurement and every decision so an agent (or you) can pick the work up in a fresh process.

| Command                | What it does                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------ |
| `gymrat start [ref]`   | Create the repository's session, or resume the one it has. Pins the baseline at `ref` (default `HEAD`) |
| `gymrat iterate`       | Bench the experiment worktree against the baseline worktree, record the result, print the verdict      |
| `gymrat keep [-m msg]` | Commit the measured edit once the checks pass, and advance the baseline to that commit                 |
| `gymrat discard`       | Revert the experiment worktree to its last commit                                                      |
| `gymrat status`        | Print the session's history, read back from the log                                                    |

```sh
gymrat start                     # pin the baseline at HEAD, create the worktrees
# ...edit the experiment worktree...
gymrat iterate                   # measure the edit
gymrat keep -m "cache the table" # commit it and move the baseline forward
gymrat status                    # read the whole session back
```

`start` is safe to re-run: an existing log is never appended to, and only a worktree that went
missing is put back. Each iteration must be settled — kept or discarded — before the next one
measures, so the log never holds two settlements against one iteration.

`keep` runs the `checks` command in the experiment worktree first and refuses to commit when it
fails; it also refuses when nothing has been measured since the last settle, and when the iteration
regressed a gating metric. Every refusal is recorded as a blocked keep, which `status` reads back —
gymrat never erases a decision from the log. A refused keep exits 1.

Every command that runs your commands or writes session state holds a per-repository lock for the
duration — `start`, `iterate`, `keep`, and `discard`, and `compare` and `measure` too. A second
gymrat run against the same repository exits 2 rather than benchmarking alongside the first, since
concurrent runs perturb each other's measurements. `status` only reads the log, so it takes no lock.

### Options

| Option                  | Default         | Description                                                               |
| ----------------------- | --------------- | ------------------------------------------------------------------------- |
| `--bench <cmd>`         | — (required\*)  | Bench command run in each target directory                                |
| `--prepare <script>`    | none            | Per-target setup, e.g. `"npm ci && npm run build"`                        |
| `--adapter <type>`      | `metric-lines`  | Output parser: `metric-lines` or `mitata`                                 |
| `--samples <number>`    | `10`            | Paired samples per target                                                 |
| `--timeout <number>`    | `1800`          | Timeout in seconds per `prepare` and per bench run                        |
| `--config <file>`       | `./gymrat.json` | Config file (loaded automatically when present)                           |
| `--format <value>`      | `text`          | Output format: `text` or `json`                                           |
| `--no-color`            | auto            | Print the report without ANSI styles                                      |
| `--verbose`             | off             | Name the statistical method behind each verdict in the footer             |
| `--fail-on <condition>` | none            | Exit 1 when a condition trips (repeatable; see [Exit codes](#exit-codes)) |
| `-r, --record`          | off             | `measure` only: append the run to the session log as a baseline           |

\*`--bench` is required either on the command line or in the config file.

`gymrat --version` prints the installed version; `gymrat compare --help` prints these options from
the binary.

Sampling is strictly sequential: for each of N windows gymrat runs the bench command once per
target, baseline first, then each candidate in the order given. Both `bench` and `prepare` run
through the shell with the working directory set to the target's directory.

### Exit codes

- `0`: a report was produced and no gate tripped.
- `1`: a gate tripped. Three things trip one: a `--fail-on` condition on `compare`, a keep the loop
  refused (`gymrat keep`), and a stop condition that refuses another iteration (`gymrat iterate`).
  The full report is printed before the exit, so you can inspect the results. A regressed verdict
  from `iterate` is not a gate — the verdict block says so, and the command still exits 0.
- `2`: an operational error (unresolvable target, nonzero bench/prepare exit, timeout, zero metrics
  parsed, config error, or invalid usage). gymrat surfaces the captured command output so you can
  see what went wrong.
- `129` / `130` / `143`: interrupted by `SIGHUP`, `SIGINT`, or `SIGTERM`. No report is produced.

#### `--fail-on` conditions

`--fail-on` is repeatable. Each condition is checked against every candidate independently:

- `regressed` — trips when any gating metric has a `regressed` verdict on any candidate. Unstable
  and non-gating metrics never trip this gate.
- `geomean:<pct>` — trips when any candidate's gated geomean, on any gating **kind** (the class an
  adapter assigns a metric, such as `time` or `memory`; see the [`kinds` config
  key](#configuration) for the full explanation), reaches `<pct>` percent in the
  costly direction (e.g. `geomean:2` trips
  at +2.0% or worse for lower-is-better metrics). Each kind is evaluated independently: a non-gating
  kind can never trip this gate regardless of its value. A candidate whose gating kinds all have
  zero stable metrics cannot trip this gate; gymrat warns on stderr instead.

```sh
# Block on any regression
gymrat compare main my-branch --bench "npm run bench" --fail-on regressed

# Block on regressions or geomean drift above 2%
gymrat compare main my-branch --bench "npm run bench" \
  --fail-on regressed --fail-on geomean:2
```

gymrat removes temporary worktrees created for git-ref targets on success, on error, and on
`SIGHUP`/`SIGINT`/`SIGTERM`. The report footer states how many were removed and how many were left behind,
naming each leftover directory with the reason git gave. On the signal path there is no report to
carry that footer, so nothing is printed: gymrat kills the running bench command, sweeps the
worktrees, and exits.

## Reading the report

A `compare` report opens with a per-metric table — each side's median and spread, then the delta and
its verdict — closes each section with a geomean row, and ends with a verdict tally and a highlights
block. Glyphs are direction-aware (`✓` improved, `✗` regressed, `≈` unstable, `=` identical, `~`
within noise, `?` inconclusive), so you never do better-is-higher math yourself, and the delta prints
even under `~`.

Verdicts come from a two-sided Wilcoxon signed-rank test at ≥ 6 nonzero paired differences, a
half-range noise band below that, and direct median comparison for config-flagged `exact` metrics.
Add `--verbose` to name the method behind each verdict in the footer.

The [reference](docs/reference.md#report-anatomy) has an annotated example and the full anatomy —
noise bands, spread columns, multi-kind sections, one-sided metrics — plus the exact verdict rules.

## CI integration

### GitHub Actions

Use `--fail-on` to gate CI on regressions, and `--format json` for machine-readable output:

```yaml
- name: Benchmark comparison
  run: |
    gymrat compare main ${{ github.head_ref }} \
      --bench "npm run bench" \
      --format json \
      --fail-on regressed \
      --fail-on geomean:2 \
      > bench-report.json
```

### JSON schema

`--format json` produces a stable, versioned JSON document — `schemaVersion: 2` for `compare`;
`measure` emits its own document, versioned separately (currently `schemaVersion: 1`). Both
increment on breaking changes. Field-by-field schemas are in the
[JSON output reference](docs/reference.md#json-output).

## The `metric-lines` format

The default adapter, `metric-lines`, is the universal "just printf your numbers" path. No
benchmark library required. gymrat scans bench **stdout** (never stderr) for lines matching:

```text
METRIC <name>=<value>
```

- A line ends at a line feed, a carriage return, or the two together, so a bench that redraws a
  progress line with a bare `\r` still gets the metrics printed after it read. No metric name can
  contain a line break.
- gymrat trims each line and keeps the ones starting with `METRIC` (note the mandatory space after
  the prefix). Everything after the prefix is trimmed and split at its **last** `=`, so metric names
  may themselves contain `=` (e.g. `decode/text=digits`).
- The left side becomes the metric name verbatim, whitespace included. `METRIC decode time=1.4`
  yields a metric named `decode time`. Check the report's metric column when a name looks wrong.
- A line starting with `METRIC` but missing the separating space — `METRICS foo=1`,
  `METRICbar=2`, `METRIC_foo=1` — produces a warning on gymrat's stderr, since it looks like a
  near-miss. Completely unrelated lines are silently ignored.
- The right side goes through JavaScript's `Number()`, which accepts more than decimal notation
  (`0x1f` parses as 31); non-finite results are rejected.
- An **empty right side** (`METRIC name=`, the shape an unset shell variable produces) is skipped,
  not read as `0`, so a missing measurement never enters the median as a real reading. A right side
  of only whitespace is the same case.
- Every non-matching line is **ignored**, so gymrat tolerates arbitrary surrounding output. A line
  starting with `METRIC` whose remainder has no `=`, has an empty name, or has an empty or
  non-finite value emits a warning on gymrat's stderr without failing the run.
- A name carrying **U+2028** (line separator) or **U+2029** (paragraph separator) draws that same
  warning and is skipped. Both count as line terminators to JavaScript's regular-expression engine,
  so a session record holding such a name could never be read back.
- A **repeated name within one run** produces within-run samples: gymrat takes the median of the
  occurrences as the run's value.
- A run in which **zero** metrics are found is an operational error (exit 2; see
  [Exit codes](#exit-codes)).
- Every metric defaults to **lower-is-better** and is rounded to the nearest integer, with no unit
  scaling. Emit nanoseconds or microseconds rather than fractional seconds, or every value collapses
  to `0` or `1`. Override direction per metric in the config file.

Example bench command:

```sh
#!/bin/sh
# Print one `METRIC name=value` line per metric; replace the values with your
# real measurements. Values here are nanoseconds, since gymrat rounds
# `metric-lines` values to whole numbers. gymrat takes the median across runs.
echo "METRIC decode/time=1420"
echo "METRIC encode/time=910"
```

The `mitata` adapter parses the JSON that [mitata](https://github.com/evanwashere/mitata) prints in
its JSON mode, flattening each benchmark to `<alias>/time` (from `stats.p50`) and `<alias>/heap`
(from `stats.heap.avg`, when mitata measured it). For parameterized benchmarks, `$name` placeholders
in the alias are replaced with `name=value`, so an alias of `decode/$text` becomes
`decode/text=digits/time`. `/time` metrics report under the `time` kind and `/heap` metrics under
`memory`; `metric-lines` names no kind, so all its metrics report under `other`.

## Configuration

gymrat loads `./gymrat.json` automatically when present (override with `--config <path>`). All keys
are optional. Precedence is **flags > config file > built-in defaults**. Unknown top-level keys are
an error, to catch typos.

```json
{
  "bench": "npm run bench",
  "prepare": "npm ci && npm run build",
  "adapter": "mitata",
  "samples": 10,
  "timeoutSeconds": 1800,
  "unstableNoisePct": 200,
  "metrics": {
    "decode/time": { "direction": "lower", "gating": true, "exact": false }
  },
  "kinds": {
    "memory": { "gating": false }
  },
  "checks": "npm test",
  "filter": "npm run bench -- --filter {names}",
  "primary": "decode/time",
  "stop": { "targetValue": 900, "maxIterations": 20 },
  "hooks": { "before": "./scripts/note-start.sh", "after": "./scripts/note-end.sh" }
}
```

- `bench`, `prepare`, `adapter`, `samples`, `timeoutSeconds` mirror the command-line options.
- `unstableNoisePct` (default `200`, minimum `0.5`) is the noise band width, in percent, above which
  a metric is too noisy to judge: its verdict becomes `unstable` (tallied under `≈` in the summary,
  serialized as `"unstable"` in JSON) and it drops out of the geomean. The comparison is strict — a
  metric sitting exactly on the threshold keeps its verdict. Values below `0.5` are rejected because
  the noise band is floored at 0.5%, so a lower threshold would mark every metric unstable. It has no
  flag — set it in the config file or leave the default.
- `metrics` keys are exact metric names. Per-metric config overrides the adapter's defaults:
  - `direction`: `"lower"` or `"higher"` (which way is better).
  - `gating`: whether the metric counts toward the geomean and the `--fail-on` gate. Defaults to
    `true`. A per-metric `gating` entry takes precedence over the kind-level setting.
  - `exact`: when `true`, any median difference is a signal and a single sample suffices.
- A `metrics` key that matches no metric the run produced is ignored without a warning, unlike an
  unknown top-level key. When an override seems to do nothing, check the spelling against the
  report's metric column.
- `kinds` keys are kind names reported by the adapter (`time`, `memory` for `mitata`; `other` for
  `metric-lines`). Each entry accepts:
  - `gating`: whether every metric of this kind counts toward the geomean and the `--fail-on` gate.
    Defaults to `true`. A per-metric `gating` entry in `metrics` overrides the kind-level setting.
- A `kinds` key that matches no kind the run produced is silently ignored, the same as an unmatched
  `metrics` key.
- An unknown sub-key inside a `metrics` or `kinds` entry is an error, the same as an unknown
  top-level key.

### Session loop keys

These five keys configure [the session loop](#the-session-loop) and are ignored by `compare` and
`measure`.

- `checks` is the command `gymrat keep` must see succeed before it commits, run in the experiment
  worktree. Without it, `keep` commits with the gate off and warns on stderr that it did.
- `filter` is the bench command a confirmation rerun uses to re-measure only the metrics that
  regressed. It must contain the `{names}` placeholder — gymrat substitutes the space-separated
  metric names there — and a `filter` without it is a config error. Left unset, a confirmation rerun
  re-runs the whole `bench` command instead.
- `primary` names the figure each iteration is read on. It defaults to `"geomean"`, the aggregate
  over every gating metric; set it to a metric name to read the loop on that metric alone.
- `stop` ends the loop. `maxIterations` (a positive integer) refuses another `iterate` once the log
  holds that many iterations. `targetValue` (a number) stops once the primary metric reaches it, in
  that metric's own direction, and only after the iteration that reached it is **kept** — so it
  requires `primary` to name a metric, and pairing it with the default geomean primary is a config
  error. A refused iteration exits 1 and prints what condition ended the loop.
- `hooks.before` and `hooks.after` are commands gymrat runs around each iteration, in the experiment
  worktree, with a JSON payload on stdin describing the stage, the session, and the last iteration.
  A hook cannot brick the loop: one that fails, crashes, or overruns its 30-second timeout becomes a
  log record and a labeled block in the report, never a failed iteration. Each hook's relayed stdout
  is capped at 8 KB, cut on a whole line.

## License

[MIT](LICENSE)
