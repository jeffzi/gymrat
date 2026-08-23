# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-23

### Added

- Benchmark-output adapters (`gymrat_py.adapters`): two built-in parsers, selected by name, that
  turn a bench script's stdout into a metric map. `metric-lines` reads `METRIC name=value` lines;
  `mitata` reads the JSON `mitata --json` writes, expanding parameterized benchmark aliases into one
  metric per argument combination. Both derive a metric's unit, kind, and display name from its name
  suffix.

## [0.2.0] - 2026-08-23

### Added

- Statistics module (`gymrat_py.stats`): descriptive helpers for median, half-range, ratio
  normalization, and geometric-mean combination, plus Wilcoxon signed-rank and exact sign-flip
  permutation significance tests.
- Core model value types (`gymrat_py.model`): effect sizes, metric metadata, verdict records, and
  aggregate results for comparing paired benchmark samples.

## [0.1.0] - 2026-08-22

### Added

- `GymratError` exception hierarchy with an optional `hint` field, plus `CommandError` for failed subprocesses.

[Unreleased]: https://github.com/jeffzi/gymrat-py/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jeffzi/gymrat-py/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jeffzi/gymrat-py/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffzi/gymrat-py/releases/tag/v0.1.0
