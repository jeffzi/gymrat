# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iterate`, `keep`, `discard`, `measure`, `compare`, `status`, and `sync` print a time-left line
  when a supervised session has a wall-clock cap active.
- `iterate`, `keep`, `discard`, `status`, `measure`, and `compare` add a top-level `budget` object
  to their JSON output, with the cap in minutes and the whole seconds remaining.
- Iteration and baseline records now carry the measurement's elapsed duration.
- Iteration records carry a fingerprint of the experiment worktree at measurement time.

### Changed

- `supervise` refuses to launch when the wall-clock cap cannot fit one iteration, unless `--force`
  is passed.
- `iterate` refuses before any hook or bench when a live budget's remaining time is smaller than the
  estimated iteration duration.

### Fixed

- The wall-clock cap now polls real time against the deadline instead of relying on a single sleep,
  so it still fires at the intended clock time when the machine sleeps mid-run.

## [0.13.0] - 2026-09-03

### Changed

- `gymrat discard --format json` adds a `measured` field and reports `seq` as `null` for an
  unmeasured revert.

### Fixed

- `gymrat discard` now reverts unmeasured edits in the experiment worktree instead of refusing.
- Supervised sessions now end when the agent finishes — or would have stopped to ask a question —
  instead of streaming idle until the wall-clock cap fires, and the closing summary shows the
  agent's final message.
- `supervise` refuses to launch when the experiment worktree has moved past the last kept commit,
  whether through an unsettled iteration, a blocked keep, or unmeasured edits, committed or not,
  regardless of `--allow-dirty`.
- The spend cap no longer fires on a supervised session that is already ending on its own.
- A `gymrat` command blocked by the repository lock now reports the holder's process, command, and
  start time reliably instead of incomplete or stale details.

## [0.12.0] - 2026-09-02

### Added

- Windows support.
- `gymrat sync` copies uncommitted main-tree changes into the experiment worktree, refusing when
  that worktree has conflicting uncommitted changes.
- `iterate`, `keep`, `discard`, and `status` accept `--format json`, with the same stable,
  backward-compatible schema as `compare` and `measure`.
- A `--color` flag complements `--no-color`: it forces color output on, overriding `NO_COLOR` and
  non-TTY detection.

### Changed

- The distribution and import package are now both named `gymrat`, replacing `gymrat-py` /
  `gymrat_py`: install with `pip install gymrat` (or `uv tool install gymrat`) and
  `import gymrat`. The `gymrat` command name is unchanged.
- The configuration file is now `gymrat.toml`, written in TOML with snake_case keys, replacing the
  former `gymrat.json`. Keys that were camelCase are now snake_case: `timeout_seconds`,
  `unstable_noise_pct`, `stop.target_value`, and `stop.max_iterations`. `gymrat init` scaffolds the
  new file, and existing configs must be converted to TOML and re-keyed.
- `gymrat init` is now non-interactive, driven by `--bench`, `--no-runbook`, and `--no-skill`.
- `gymrat doctor` validates the bench configuration without running a benchmark.
- Significance verdicts now use an exact sign-flip permutation test instead of Wilcoxon
  signed-rank, so results near the boundary may differ from earlier releases.
- A supervised session runs fewer full measurements per loop: edits are probed with
  `gymrat measure` and `gymrat iterate` runs only before `keep`.
- Metric names now separate the kind suffix with `#` instead of `/` (`bench/time` becomes
  `bench#time`); `/` continues to separate path segments and drives report grouping.
- Verdicts with too few paired samples for the permutation test now display as inconclusive.

### Fixed

- Band verdicts now require pairs that actually differ, so a run dominated by tied samples can no
  longer produce a false signal, and a metric whose median is zero while its samples spread is
  judged `unstable` instead of measured against a percentage band.
- `discard` after a blocked keep now reports the iteration that was actually reverted.
- `finalize` refuses when the experiment worktree has moved past the last kept commit, hinting to
  keep or discard first.
- Metric names containing spaces, parentheses, or quotes now reach the bench command intact when
  substituted into a `filter` template.
- `--debug` takes effect whether written before or after the subcommand.
- A closed output pipe ends the run quietly instead of printing a bug-report footer.
- The mitata adapter no longer fails to read its report when the bench command prints extra
  output around the JSON, and warns about entries it cannot use instead of skipping them silently.
- A session log whose final line was torn by a crash mid-write is repaired on the next run, and a
  record reported as written survives a crash immediately after.
- A `gymrat.toml` that is not valid UTF-8 is reported as unreadable instead of crashing, and an
  oversized integer in a `GYMRAT_*` environment variable is reported as invalid instead of
  crashing.
- A lock file left behind by another user in a shared temporary directory now reports a clear remedy
  instead of failing with a raw permission error.
- `--no-color` no longer leaks into the environment of the benchmark commands.
- A supervised session handles oversized benchmark output and unresponsive processes at teardown
  without crashing or hanging.

### Removed

- The interactive wizard prompts in `gymrat init` and the `--adapter`, `--checks`, `--stop-target`,
  `--stop-max-iterations`, `--primary`, `--runbook PATH`, and `--yes` flags.
- `gymrat doctor --no-bench`, along with the smoke benchmark run it used to skip.

## [0.11.0] - 2026-08-25

### Added

- `gymrat supervise [prompt]` runs an agent that drives the optimization loop, bounded by a
  wall-clock cap (`--max-minutes`) and an optional spend cap (`--max-usd`).
- `gymrat init` scaffolds a project with `gymrat.json`, a runbook stub, and the skill file.
- `gymrat doctor` checks the project setup and reports grouped findings, exiting non-zero on
  failure.
- The gymrat skill file ships inside the package.

## [0.10.0] - 2026-08-25

### Added

- `gymrat start [ref]` opens or resumes an optimization session, pinning the baseline at a ref
  (default `HEAD`).
- `gymrat iterate` measures the experiment worktree against the baseline and reports the verdict.
- `gymrat keep` commits the measured edit and advances the baseline, refusing when checks fail or
  a gating regression stands.
- `gymrat discard` reverts the experiment worktree to its last commit.
- `gymrat finalize` squashes the session's kept commits into one commit on the baseline and closes
  the session.
- `gymrat status` prints the session history.
- A noisy gating regression triggers a confirmation rerun before the verdict stands.
- Lifecycle hooks: `before` and `after` commands run around each measurement.

## [0.9.0] - 2026-08-24

### Added

- A `--record` (`-r`) flag for `gymrat measure` that appends the run to the open session log as a
  labeled baseline, refusing before it benchmarks when no open session exists.

## [0.8.0] - 2026-08-24

### Added

- `gymrat compare` compares a baseline revision against one or more candidates and reports as
  text or JSON.
- `gymrat measure` measures a single revision or directory on its own.
- A `--fail-on` gate for `compare` that exits non-zero on a gating regression or a geometric-mean
  threshold.
- A repository lock prevents two gymrat runs from colliding.
- Canceling a run stops the benchmark, removes worktrees the run created, and exits with
  `128 + N`.

## [0.7.0] - 2026-08-23

### Added

- `compare` reports a metric-by-metric table with verdicts and a geometric-mean row per kind.
- `measure` reports each metric's median and spread, grouped by kind.
- Reports include a verdict summary and a highlights block calling out the metrics that moved
  most.
- `--fail-on` reports the kind and threshold when tripped, and leftover worktrees appear in a
  closing footer.
- `compare` and `measure` support `--format json` for machine-readable output.
- Reports honor `FORCE_COLOR` and `NO_COLOR`.
- Each loop iteration reports its verdict and outcome.

## [0.6.0] - 2026-08-23

### Added

- `gymrat` reads and validates a `gymrat.json` configuration file, reporting the offending key
  when a value is wrong.
- Run configuration resolves from command-line values, `GYMRAT_*` environment variables, the
  config file, and built-in defaults, in that order.
- Per-metric metadata resolves from adapter defaults and per-metric and per-kind overrides.
- Invalid configuration reports all problems at once instead of stopping at the first.

## [0.5.0] - 2026-08-23

### Added

- Per-metric verdicts judge each metric improved, regressed, or unchanged, reporting noisy metrics
  as unstable.
- Aggregates roll up verdicts per kind, per benchmark group, and over gating metrics, excluding
  metrics that cannot be judged.
- A warning appears when a metric is present in only some rounds, reporting how many were dropped.

### Fixed

- Parsing a `METRIC` line with an out-of-range radix value (e.g. a very long hex literal) no longer
  crashes; the value is skipped with a warning like any other unparseable one.

## [0.4.0] - 2026-08-23

### Added

- Benchmark sampling collects repeated measurements for one or more targets, running an optional
  setup step before the timed runs.
- Benchmark commands run under an optional timeout and stop cleanly when canceled or timed out.
- Interrupting gymrat with Ctrl-C or a signal runs pending cleanup before exiting.

## [0.3.0] - 2026-08-23

### Added

- Two built-in benchmark-output adapters — `metric-lines` and `mitata` — parse a bench script's
  output into metrics.

## [0.2.0] - 2026-08-23

### Added

- Significance testing and summary statistics for paired benchmark samples.

## [0.1.0] - 2026-08-22

### Added

- Structured error reporting for every gymrat failure: the command prints a clear message and, where
  one applies, an actionable hint for what to do next.

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/jeffzi/gymrat/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/jeffzi/gymrat/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/jeffzi/gymrat/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/jeffzi/gymrat/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/jeffzi/gymrat/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/jeffzi/gymrat/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/jeffzi/gymrat/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jeffzi/gymrat/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jeffzi/gymrat/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jeffzi/gymrat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jeffzi/gymrat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jeffzi/gymrat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffzi/gymrat/releases/tag/v0.1.0
