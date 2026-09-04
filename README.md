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

| Command                             | What it does                                                  |
| ----------------------------------- | ------------------------------------------------------------- |
| `gymrat init`                       | Scaffold a `gymrat.toml`, an agent skill file, and a runbook  |
| `gymrat compare <baseline> <cand>…` | Judge one or more candidates against a baseline               |
| `gymrat measure [target]`           | Measure a single revision or directory on its own             |
| `gymrat doctor`                     | Check the project setup and report problems                   |
| `gymrat start` … `gymrat finalize`  | The optimization loop (below)                                 |
| `gymrat sync`                       | Copy uncommitted main-tree edits into the experiment worktree |
| `gymrat supervise "<prompt>"`       | Run a supervised agent session with wall-clock and spend caps |

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
gymrat finalize            # squash kept iterations into one commit and close
```

`gymrat supervise "optimize the decoder" --max-minutes 30 --max-usd 5` runs that loop under an AI
agent: the runbook scaffolded by `init` describes the goal and constraints, and the session ends
when the agent finishes or a cap trips. `iterate`, `keep`, `discard`, `status`, `sync`, `compare`,
and `measure` print a time-left line so the agent can plan around the wall-clock cap.

### Hooks

The `[hooks]` table in `gymrat.toml` runs shell commands around each `gymrat iterate` measurement.
No other command runs them.

```toml
[hooks]
before = "npm run build" # once per iteration, before measuring
after = "./scripts/notify.sh" # after the iteration record is written
```

Hooks run with the experiment worktree as their working directory and receive a JSON object on stdin
with keys `stage`, `experimentDir`, `seq`, `lastIteration`, and `session`. A hook that exits
non-zero is reported in the iteration output but does not fail the iteration. Hooks that exceed the
30-second timeout are killed.

An after hook must not modify the experiment worktree — the iteration record's fingerprint reflects
the worktree at measurement time, and later edits go unrecorded.

## Two workflows, one tool

gymrat serves two audiences with the same statistical engine:

- **One-shot comparisons** — `gymrat compare` and `gymrat measure` answer a point-in-time question
  ("is this branch faster?") and clean up after themselves. Use these in CI gates, code reviews, or
  any time you have specific revisions to judge.

- **Iterative optimization** — `gymrat start` through `gymrat finalize` manage a session with a
  pinned baseline and an experiment worktree, so you can iterate on performance work with
  keep/discard decisions backed by statistics. `gymrat supervise` automates the same loop under an
  AI agent.

## Machine-readable output

Every comparison, measurement, and session-loop command (`iterate`, `keep`, `discard`, `status`)
accepts `--format json` for structured output. The JSON key shapes are a stability contract:
additions only, no renames or removals without a breaking change. Text output is for humans and may
change between releases. `start`, `sync`, and `finalize` are text-only.

## How verdicts work

Each candidate is sampled in strict alternation with its baseline, so machine drift hits both
sides equally. A metric's verdict comes from a sign-flip permutation test on the paired deltas;
below the sample floor a noise-band fallback applies, and changes inside the noise band are
reported as `no-signal` rather than celebrated. Erratic metrics come back `unstable` instead of
misleading you.

## License

[MIT](LICENSE)
