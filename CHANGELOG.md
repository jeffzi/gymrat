# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-09-02

### Added

- Windows support for the optimization session: `start`, `iterate`, `keep`, `discard`,
  `finalize`, and `status` now run on Windows.
- `gymrat sync` copies uncommitted main-tree changes into the experiment worktree, for edits that
  must land in the main working tree first (for example a dependency update); it refuses when the
  experiment worktree has conflicting uncommitted changes.
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
- `gymrat init` is now non-interactive, driven by `--bench`, `--no-runbook`, and `--no-skill`,
  and re-running it over an existing `gymrat.toml` restores any missing runbook or skill file
  without touching the config.
- `measure`, `compare`, and `iterate` replace the single progress line with per-target progress
  bars showing elapsed time, with plain text when the output is not a terminal.
- `gymrat supervise` replaces its one-line progress with a live dashboard: elapsed time and spend
  against the caps, the best result so far, loop tallies, and the model and tool activity
  currently running.
- `gymrat doctor` no longer executes the bench command: the bench section now validates that the
  adapter resolves, a bench command is configured, and its executable is on `PATH`, instead of
  running a smoke benchmark.
- Significance verdicts now come from an exact sign-flip permutation test (labeled `permutation`
  in reports and JSON) instead of the Wilcoxon signed-rank test, so a metric near the significance
  boundary can land on the opposite verdict from earlier releases.
- The bundled skill now has the agent probe an edit with `gymrat measure` on the experiment
  worktree, scoped to the benchmarks the edit targets, and reserve `gymrat iterate` for the one
  full measurement right before `keep`.
- Metric names now use `#` as the suffix separator instead of `/`: a name like `bench/time` is now
  `bench#time`. The `metric-lines` and `mitata` adapters follow the new grammar; a metric name
  containing more than one `#` is rejected.
- Report tables group metrics by their `/`-separated path prefix instead of by the first dot in the
  short name.
- Verdicts with too few paired samples for the permutation test now display as inconclusive.

### Fixed

- Band verdicts now require pairs that actually differ, so a run dominated by tied samples can no
  longer produce a false signal; a metric whose median is zero while its samples spread is judged
  `unstable` instead of measured against a percentage band.
- `discard` after a blocked keep now reports the iteration that was actually reverted, and
  `finalize` refuses when the experiment worktree has moved past the last kept commit, hinting to
  keep or discard first.
- Metric names containing spaces, parentheses, or quotes now reach the bench command intact when
  substituted into a `filter` template.
- `--debug` takes effect whether written before or after the subcommand, and a closed output pipe
  ends the run quietly instead of printing a bug-report footer.
- The mitata adapter no longer fails to read its report when the bench command prints extra
  output around the JSON, and warns about entries it cannot use instead of skipping them
  silently.
- A session log whose final line was torn by a crash mid-write is read and repaired on the next
  run, and a record reported as written survives a crash immediately after.
- A `gymrat.toml` that is not valid UTF-8 is reported as unreadable instead of crashing, and an
  oversized integer in a `GYMRAT_*` environment variable is reported as invalid instead of
  crashing.
- A lock file left behind by another user in a shared temporary directory now reports a clear remedy
  instead of failing with a raw permission error.
- `--no-color` no longer leaks into the environment of the benchmark commands.
- A supervised agent session whose benchmark process emits an oversized, unterminated output line
  now ends with a clear error instead of crashing.
- Tearing down a supervised agent session no longer hangs indefinitely when the supervised process
  ignores the stop signal; teardown now waits briefly, then abandons the wait and warns.

### Removed

- The interactive wizard prompts in `gymrat init` and the `--adapter`, `--checks`, `--stop-target`,
  `--stop-max-iterations`, `--primary`, `--runbook PATH`, and `--yes` flags.
- `gymrat doctor --no-bench`, along with the smoke benchmark run it used to skip.

## [0.11.0] - 2026-08-25

### Added

- `gymrat supervise [prompt]` runs an agent that drives the optimization loop on its own, bounded by
  a wall-clock cap (`--max-minutes`) and an optional spend cap (`--max-usd`).
- A live progress line for supervised sessions showing the elapsed time and spend against the caps,
  loop progress (iterations, keeps, discards, and the last verdict), and the current tool activity,
  with a plain-text fallback when the output is not a terminal.
- `gymrat init` scaffolds a project: an interactive wizard (or `--yes` for non-interactive use)
  settles the configuration, then writes `gymrat.json`, a runbook stub, and the gymrat skill file.
- `gymrat doctor` checks the project setup — environment, configuration, workflow, and an optional
  bench smoke run — as grouped text or `--format json`, exiting non-zero when any check fails.
- The gymrat skill file now ships inside the package, so a supervised session and `gymrat init` both
  use it without a separate install step.

## [0.10.0] - 2026-08-25

### Added

- `gymrat start [ref]` opens or resumes a per-repository optimization session, pinning the baseline
  at a ref (default `HEAD`) and checking out an experiment and a baseline worktree.
- `gymrat iterate` measures the experiment worktree against the baseline and reports the verdict,
  honoring a configurable stop condition: a maximum iteration count or a target value for a named
  metric.
- `gymrat keep` commits the measured edit once the configured checks command passes, advancing the
  baseline onto it, and refuses a standing gating regression or failing checks.
- `gymrat discard` reverts the experiment worktree to its last commit, with a confirmation prompt
  on a terminal (skip it with `-f`/`--force`).
- `gymrat finalize` collapses the session's kept commits into a single squash commit on the pinned
  baseline and closes the session.
- `gymrat status` prints the whole session history, rebuilt from its log alone.
- A confirmation rerun that re-measures a noisy gating regression once and lets it stand only when
  the rerun agrees; a deterministic (exact) metric is judged on the first run alone.
- Lifecycle hooks: `before` and `after` commands run around each measurement, receive a JSON
  description of the loop on their standard input, have their output relayed under a size cap, and
  are isolated so a failing hook is reported without failing the run.

## [0.9.0] - 2026-08-24

### Added

- A `--record` (`-r`) flag for `gymrat measure` that appends the run to the open session log as a
  labeled baseline, refusing before it benchmarks when no open session exists.

## [0.8.0] - 2026-08-24

### Added

- The `gymrat compare` command: compare a baseline revision against one or more candidates — each
  an existing directory or a git ref checked out into a throwaway worktree — and report as text or
  JSON.
- The `gymrat measure` command: measure a single revision or directory on its own, defaulting to the
  current directory, with nothing to compare against.
- A repeatable `--fail-on` gate for `compare` that exits non-zero when a gating metric regresses
  (`--fail-on regressed`) or a gated geometric mean crosses a threshold (`--fail-on geomean:<pct>`).
- A live progress line reporting each prepare and sample step with an estimated time remaining,
  adapting to the terminal width and color settings and clearing itself before the report is
  printed.
- A repository lock so two gymrat runs over the same repository cannot collide; a run started
  outside a git repository proceeds without a lock.
- Interruption handling: canceling a run stops the benchmark, removes any worktrees the run created,
  and exits with the conventional signal code.

## [0.7.0] - 2026-08-23

### Added

- Comparison report tables: a metric-by-metric table pitting a baseline against one or more
  candidates, showing each side's value with its spread, the verdict with its noise band, and a
  geometric-mean row per metric kind.
- Measurement report table: a single-target table listing each metric's median and spread,
  sectioned by kind the same way, for a run with nothing to compare against.
- Verdict summary and highlights: a one-line tally of how many metrics improved, regressed, stayed
  within noise, or could not be judged, followed by a highlights block calling out the metrics that
  moved most — with a note that unstable metrics will not settle with more samples.
- Gate-trip and cleanup reporting: a crossed `--fail-on` geomean threshold names the kind, its gated
  geometric mean, and the threshold, and leftover worktrees and prune failures are reported in a
  closing footer.
- Machine-readable JSON output: comparison and measurement runs render as a stable, versioned JSON
  document with per-metric metadata, per-candidate verdicts and aggregates, and worktree-cleanup
  state — parseable by other tools and unaffected by display options.
- Color control: report output is styled when writing to a terminal and left plain for a pipe or
  file, forced on or off explicitly, with `FORCE_COLOR` and `NO_COLOR` honored.
- Loop iteration reporting: each optimization-loop iteration reports a header, a verdict block
  naming its primary figure, and an improved, regressed, or no-signal outcome.

## [0.6.0] - 2026-08-23

### Added

- Configuration file loading: read and validate a `gymrat.json` file — the
  bench, prepare, and adapter commands, sample count, timeout, noise threshold, per-metric and
  per-kind overrides, and loop settings. An unknown key, a wrong type, an empty or whitespace-only
  command, an out-of-range number, or a malformed section each fails with a message that names the
  offending key and what was expected.
- Settled run configuration: combine command-line values, `GYMRAT_*`
  environment variables, the config file, and built-in defaults in that order of precedence. The
  config file is located via `--config`, then `GYMRAT_CONFIG`, then `gymrat.json` at the repository
  root (or the working directory outside a repository); cross-field rules are checked and a runbook
  path is resolved to an absolute path.
- Per-metric metadata resolution: merge adapter-derived defaults with
  per-metric and per-kind overrides to settle each metric's direction, gating, exact-comparison
  flag, kind, unit, and display name.
- Configuration inspection: settle the same configuration while
  collecting every problem at once instead of stopping at the first.

## [0.5.0] - 2026-08-23

### Added

- Per-metric verdicts for a benchmark comparison: pair two runs by round and
  judge each metric improved, regressed, or no clear change in its own direction. A metric too noisy
  to measure against is reported unstable rather than given a misleading verdict.
- Headline and per-section aggregates: combine the per-metric verdicts into
  one summary, rolled up per kind, per benchmark group, and over only the gating metrics, the ones
  that decide the overall pass or fail. A metric that cannot be judged (no paired samples, an
  unstable verdict, or an undefined ratio) is named in an exclusion list instead of silently
  skewing the summary.
- A warning when a metric present in only some rounds thins the paired sample: the comparison still
  runs on the rounds where the metric appears, and reports how many were dropped.

### Fixed

- Parsing a `METRIC` line with an out-of-range radix value (e.g. a very long hex literal) no longer
  crashes; the value is skipped with a warning like any other unparseable one.

## [0.4.0] - 2026-08-23

### Added

- Benchmark sampling across targets: collect repeated
  measurements for one or more targets — a git ref or a directory — running an optional setup step
  once per target before the timed runs. A command that fails or exceeds its time limit stops the
  run with a message naming which target, command, and phase failed and showing its output.
- Time limits and cancellation for benchmark commands: each command runs under
  an optional timeout and can be canceled; either one stops the command and the process group it
  started, so a helper the shell left running is cleaned up too. Captured output is kept up to
  64 MiB per stream; longer output is truncated and flagged with its full byte count.
- Clean shutdown on interruption: interrupting gymrat with Ctrl-C, or a
  `SIGTERM`/`SIGHUP`, runs pending cleanup before exiting instead of leaving work half-torn-down.

## [0.3.0] - 2026-08-23

### Added

- Benchmark-output adapters: two built-in parsers, selected by name, that
  turn a bench script's stdout into a metric map. `metric-lines` reads `METRIC name=value` lines;
  `mitata` reads the JSON `mitata --json` writes, expanding parameterized benchmark aliases into one
  metric per argument combination. Both derive a metric's unit, kind, and display name from its name
  suffix.

## [0.2.0] - 2026-08-23

### Added

- Significance testing and summary statistics for paired benchmark samples:
  median, half-range, ratio normalization, and geometric-mean combination, plus Wilcoxon
  signed-rank and exact sign-flip permutation tests.

## [0.1.0] - 2026-08-22

### Added

- Structured error reporting for every gymrat failure: the command prints a clear message and, where
  one applies, an actionable hint for what to do next.

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.12.0...HEAD
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
