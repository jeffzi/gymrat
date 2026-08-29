# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Windows support for the optimization session: `start`, `iterate`, `keep`, `discard`,
  `finalize`, and `status` now run on Windows.

### Changed

- The distribution and import package are now both named `gymrat`, replacing `gymrat-py` /
  `gymrat_py`: install with `pip install gymrat` (or `uv tool install gymrat`) and
  `import gymrat`. The `gymrat` command name is unchanged.
- The configuration file is now `gymrat.toml`, written in TOML with snake_case keys, replacing the
  former `gymrat.json`. Keys that were camelCase are now snake_case: `timeout_seconds`,
  `unstable_noise_pct`, `stop.target_value`, and `stop.max_iterations`. `gymrat init` scaffolds the
  new file, and existing configs must be converted to TOML and re-keyed.
- `gymrat init` is now non-interactive: it takes `--bench` (required), `--no-runbook`, and
  `--no-skill` flags and writes only the `bench` and `runbook` config keys. It is also
  re-runnable: an existing `gymrat.toml` is left untouched and any missing runbook or skill file
  is restored, so `--bench` is only required when no config exists yet.
- `measure` and `compare` replace the single progress line with per-target progress bars carrying
  running clock timers and a metric count; `iterate` replaces its line with a step checklist
  showing live elapsed time and per-pass detail. When the output is not a terminal, plain text is
  printed instead.
- `gymrat supervise` replaces its one-line progress with a live dashboard: elapsed time and
  spend against the caps, the best result so far with wall-clock timestamps, loop tallies, and
  live progress of the measurement currently running.
- `gymrat doctor` no longer executes the bench command: the bench section now validates that the
  adapter resolves, a bench command is configured, and its executable is on `PATH`, instead of
  running a smoke benchmark. Its summary counts render in their status colors and zero counts
  are omitted.
- Report hints render dim without the former `Hint:` label.
- Significance verdicts now come from an exact sign-flip permutation test instead of the Wilcoxon
  signed-rank test. The permutation test's statistic is the delta each verdict already reports — the
  percent change in the median — so a significant result can no longer point in a different direction
  than the number shown beside it. It also computes exact results for small sample sizes instead of
  relying on approximations that can distort the outcome. Verdicts are labeled `permutation` in the
  text reports and in the JSON output. Near the significance boundary a metric can now land on the
  opposite verdict from earlier releases; in particular, a run whose samples barely move is less
  likely to be judged significant than before.

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
- `gymrat status` renders metric names and paths containing square brackets literally instead of
  interpreting them as styling.
- The mitata adapter no longer fails to read its report when the bench command prints extra
  output around the JSON, and warns about entries it cannot use instead of skipping them
  silently.
- A session log whose final line was torn by a crash mid-write is read and repaired on the next
  run, and a record reported as written survives a crash immediately after.
- A `gymrat.toml` that is not valid UTF-8 is reported as unreadable instead of crashing, and an
  oversized integer in a `GYMRAT_*` environment variable is reported as invalid instead of
  crashing.
- Interrupting a run on a terminal now clears the progress line before exiting, so no stray status
  text is left on the current line.
- A lock file left behind by another user in a shared temporary directory now reports a clear remedy
  instead of failing with a raw permission error.
- `compare` and `measure` reports now render in color when written to a terminal and drop color when
  redirected to a file or pipe, instead of always rendering plain.
- Color detection now follows one rule across reports, the progress line, and error messages, so
  `FORCE_COLOR` and `NO_COLOR` take effect consistently; a `FORCE_COLOR` of `0`, `false`, or empty no
  longer forces color on.
- `--no-color` no longer sets `NO_COLOR` in the environment of the benchmark commands it runs.
- The progress line no longer spills a garbled row on a terminal that reports zero or unknown width
  (for example `COLUMNS=0`); such a width now collapses the line instead of being treated as 80
  columns.
- A supervised agent session whose benchmark process emits an oversized, unterminated output line
  now ends with a clear error instead of crashing.
- Tearing down a supervised agent session no longer hangs indefinitely when the supervised process
  ignores the stop signal; teardown now waits briefly, then abandons the wait and warns.

### Removed

- The interactive wizard prompts in `gymrat init` and the `--adapter`, `--checks`, `--stop-target`,
  `--stop-max-iterations`, `--primary`, `--runbook PATH`, `--skill`/`--no-skill`, and
  `--yes` flags.
- `gymrat doctor --no-bench`, along with the smoke benchmark run it used to skip.

## [0.11.0] - 2026-08-25

### Added

- `gymrat supervise [prompt]` runs an agent that drives the optimization loop on its own, bounded by
  a required wall-clock cap (`--max-minutes`) and an optional spend cap (`--max-usd`). When a cap is
  reached the agent is interrupted and given a grace period to stop; the run then reports how it
  ended, its duration, and its cost. Around that core:
  - It requires a `runbook` in `gymrat.json` and refuses to start on a dirty tree unless
    `--allow-dirty` is passed.
  - It holds a lock separate from the session lock.
  - It records every step of the session to a JSONL event log (`--log`, defaulting under `.gymrat/`).
- A live progress line for supervised sessions showing the elapsed time and spend against the caps,
  loop progress (iterations, keeps, discards, and the last verdict), and the current tool activity,
  with a plain-text fallback when the output is not a terminal.
- `gymrat init` scaffolds a project: an interactive wizard (or flag-driven, with `--yes` for
  non-interactive use) settles the bench command and optional adapter, checks, stop condition,
  primary metric, runbook, and skill choices, then writes `gymrat.json`, a runbook stub, and the
  gymrat skill file — validating the configuration before writing so a bad setup leaves nothing
  behind.
- `gymrat doctor` checks the project setup and reports problems across the environment (git and
  repository), configuration, workflow (skill file, checks, stop condition, runbook), and an
  optional bench smoke run that parses the bench output and cross-checks it against the configured
  primary, metrics, and kinds. It renders as grouped text or `--format json` and exits non-zero when
  any check fails; `--no-bench` skips the smoke run.
- The gymrat skill file now ships inside the package, so a supervised session and `gymrat init` both
  use it without a separate install step.

## [0.10.0] - 2026-08-25

### Added

- An optimization-loop workflow driven by six new commands over a per-repository session that pins a
  baseline and measures an experiment worktree against it, one edit at a time:
  - `gymrat start [ref]` opens or resumes the session, pinning the baseline at a ref (default `HEAD`)
    and checking out an experiment and a baseline worktree; a finalized session's log is archived and
    a fresh session takes its place.
  - `gymrat iterate` measures the experiment worktree against the baseline and reports the verdict,
    honoring configurable stop conditions — a maximum iteration count, or a target value for a named
    metric.
  - `gymrat keep` commits the measured edit once a configured checks command passes, advancing the
    baseline onto the kept commit; it refuses a standing gating regression or failing checks and
    records the block.
  - `gymrat discard` reverts the experiment worktree to its last commit, with a confirmation prompt
    on a terminal (skip it with `-f`/`--force`).
  - `gymrat finalize` collapses the session's kept commits into a single squash commit on the pinned
    baseline and closes the session, leaving the per-iteration history in place.
  - `gymrat status` prints the whole session history, rebuilt from its log alone.
- A confirmation rerun that re-measures a noisy gating regression once and lets it stand only when
  the rerun agrees; a deterministic (exact) metric is judged on the first run alone.
- Lifecycle hooks: `before` and `after` commands run around each measurement, receive a JSON
  description of the loop on their standard input, have their output relayed under a size cap, and
  are isolated so a failing hook is reported without failing the run.

## [0.9.0] - 2026-08-24

### Added

- A `--record` (`-r`) flag for `gymrat measure` that appends the run to a per-repository session log
  as a labeled baseline, capturing each round's raw samples. Recording requires an open session; a
  missing or already-finalized session, or a run outside a git repository, stops the command before
  it benchmarks so a long run is never discarded with nowhere to record it.

## [0.8.0] - 2026-08-24

### Added

- The `gymrat compare` command: compare a baseline revision against one or more candidates and report
  each candidate against the shared baseline, as text or JSON. A target is an existing directory
  benched where it sits or a git ref — branch, tag, or commit — checked out into a throwaway
  worktree pinned to its commit; an existing directory wins over an equally named ref.
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

- Comparison report tables: a metric-by-metric table pitting a baseline against one
  candidate — or a column per candidate against a shared baseline — showing each side's value with
  its spread, the verdict with its noise band, and a geometric-mean row summarizing the run. A run
  spanning several metric kinds splits into a titled section per kind, each closed by its own
  geometric mean, and a kind that only informs (never gates) is labeled as such with the config line
  that
  made it so.
- Measurement report table: a single-target table listing each metric's median and spread,
  sectioned by kind the same way, for a run with nothing to compare against.
- Verdict summary and highlights: a one-line tally of how many metrics improved, regressed, stayed
  within noise, or could not be judged, followed by a highlights block calling out the metrics that
  moved most — with a note that unstable metrics will not settle with more samples.
- Gate-trip and cleanup reporting: when a `--fail-on` geomean threshold is crossed the report names
  the kind, its gated geometric mean, and the threshold it exceeded; leftover worktrees and prune
  failures
  are reported in a closing footer. A verbose run also spells out the statistical method behind the
  verdicts and hints at when more samples would help.
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

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.11.0...HEAD
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
