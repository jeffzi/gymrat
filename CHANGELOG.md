# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-31

### Added

- `compare` command: run a bench command against a baseline and one or more candidates — git refs
  checked out into temporary worktrees, or existing directories — cycling samples across targets so
  every target sees the same machine noise.
- Format adapters: `metric-lines` (scans stdout for `METRIC name=value` lines) and `mitata`
  (parses mitata's JSON output).
- Per-metric tri-state verdicts (improved, regressed, no signal) from a two-sided Wilcoxon
  signed-rank test once there are enough samples, with a noise-band fallback below that; `exact`
  metrics treat any variation as a signal from a single sample.
- Benchstat-style comparison report — median ± spread per target, deltas with verdict markers, and
  a geomean summary row — in `text`, `markdown`, and `json` formats, with automatic color detection
  (`--no-color`, `NO_COLOR`).
- `gymrat.json` config file for metric metadata (direction, gating, exact), merged with CLI flags.
- `--prepare` per-target setup, `--samples`, `--timeout`, and repeatable `--fail-on` conditions for
  failing CI on regressions.

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jeffzi/gymrat/releases/tag/v0.1.0
