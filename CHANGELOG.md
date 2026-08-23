# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-metric verdicts for a benchmark comparison (`gymrat_py.verdict`): pair two runs by round and
  judge each metric improved, regressed, or no clear change in its own direction. A metric too noisy
  to measure against is reported unstable rather than given a misleading verdict.
- Headline and per-section aggregates (`gymrat_py.verdict`): combine the per-metric verdicts into
  one summary, rolled up per kind, per benchmark group, and over only the gating metrics — the ones
  that decide the overall pass or fail. A metric that cannot be judged — no paired samples, an
  unstable verdict, or an undefined ratio — is named in an exclusion list instead of silently
  skewing the summary.
- A warning when a metric present in only some rounds thins the paired sample: the comparison still
  runs on the rounds where the metric appears, and reports how many were dropped.

## [0.4.0] - 2026-08-23

### Added

- Benchmark sampling across targets (`gymrat_py.sampling`, `gymrat_py.targets`): collect repeated
  measurements for one or more targets — a git ref or a directory — running an optional setup step
  once per target before the timed runs. A command that fails or exceeds its time limit stops the
  run with a message naming which target, command, and phase failed and showing its output.
- Time limits and cancellation for benchmark commands (`gymrat_py.exec`): each command runs under
  an optional timeout and can be cancelled; either one stops the command and the process group it
  started, so a helper the shell left running is cleaned up too. Captured output is kept up to
  64 MiB per stream; longer output is truncated and flagged with its full byte count.
- Clean shutdown on interruption (`gymrat_py.signals`): interrupting gymrat with Ctrl-C, or a
  `SIGTERM`/`SIGHUP`, runs pending cleanup before exiting instead of leaving work half-torn-down.

## [0.3.0] - 2026-08-23

### Added

- Benchmark-output adapters (`gymrat_py.adapters`): two built-in parsers, selected by name, that
  turn a bench script's stdout into a metric map. `metric-lines` reads `METRIC name=value` lines;
  `mitata` reads the JSON `mitata --json` writes, expanding parameterized benchmark aliases into one
  metric per argument combination. Both derive a metric's unit, kind, and display name from its name
  suffix.

## [0.2.0] - 2026-08-23

### Added

- Significance testing and summary statistics for paired benchmark samples (`gymrat_py.stats`):
  median, half-range, ratio normalization, and geometric-mean combination, plus Wilcoxon
  signed-rank and exact sign-flip permutation tests.
- Value types for benchmark-comparison results (`gymrat_py.model`): effect sizes, metric metadata,
  verdict records, and aggregates for paired samples.

## [0.1.0] - 2026-08-22

### Added

- Structured errors for every gymrat failure: each derives from `GymratError` and can carry a
  `hint`; failed subprocesses raise `CommandError`.

[Unreleased]: https://github.com/jeffzi/gymrat-py/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/jeffzi/gymrat-py/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jeffzi/gymrat-py/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jeffzi/gymrat-py/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffzi/gymrat-py/releases/tag/v0.1.0
