# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ETA countdown displayed between samples during comparison runs, with interpolation between updates.
- `identical` verdict for metrics where tied samples starve the statistical test of usable pairs.
- Per-cause hint lines when a metric falls back to the noise-band method, printed regardless of
  `--verbose`.

### Changed

- Text report redesign: sub-field column alignment, label truncation with emphasized variant names,
  reshaped headers, and a dimmed echo row under the geomean.
- Spread display capped at 100%; unstable metrics show a futility note instead.
- Legend removed from the default report; method lines now require `--verbose`.
- Report color scoped via a `ReportOptions.color` option instead of environment-variable mutation.

### Fixed

- Band method returned a false signal from a single paired sample.
- ETA remaining-sample count off by one.
- `styleWithin` applied styling at the wrong position when the pattern occurred multiple times.
- Markdown renderer ignored the color option under `FORCE_COLOR`.
- Hint underline extended past the hint word into the trailing colon.
- Variant names rendered with spurious quotes in plain-text output.
- Error hints checked stdout instead of stderr for color capability.

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
