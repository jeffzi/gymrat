# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-08

### Added

- `measure` command for single-target benchmarking: `gymrat measure [[label=]<ref|dir>]` samples
  one target — a git ref (benched in a throwaway worktree), an existing directory in place, or,
  with the argument omitted, the current tree — and reports per-metric median ± spread grouped by
  kind, with no delta, verdict, or geomean output. It shares `compare`'s configuration surface
  (`--bench`, `--prepare`, `--adapter`, `--samples`, `--timeout`, `--config`, `--no-color`,
  `gymrat.json`), and runs are ephemeral — nothing is recorded.
- `measure --format json` emits a machine-readable measurement report with its own
  `schemaVersion: 1`, independent of `compare`'s comparison schema.

## [0.2.0] - 2026-08-07

### Added

- ETA countdown displayed between samples during comparison runs, with interpolation between updates.
- `identical` display class (`=`) for metrics whose samples all tied, split out from `within noise`.
  Presentation only — JSON, `--fail-on`, and geomean gating still see `no signal`.
- A `Hint:` line, printed regardless of `--verbose`, when a metric fell back to the noise-band
  method for want of samples and more samples would buy a statistical verdict.
- `inconclusive` display class (`?`) for metrics with too few paired samples for the noise band to
  carry signal. Presentation only — JSON, `--fail-on`, and geomean gating still see `no signal`.
- Geomean and sub-geomean rows print their propagated noise band (`±N.N%`) after the delta.
- Header reads `1 paired sample` (singular) for single-sample runs.
- Byte-metric noise floor: the reported noise percentage never falls below one byte relative to each
  median, preventing a one-byte quantization flip from producing a spurious verdict.
- Per-kind metric sections: when a run produces metrics of more than one kind (e.g. `time` and
  `memory`), reports render a section per kind with its own geomean and group sub-geomeans.
- `kinds` config key for per-kind gating: `"kinds": { "memory": { "gating": false } }` switches an
  entire kind to informational — its metrics stay in the report but leave the geomean and the
  `--fail-on` gate. Per-metric `gating` overrides still win over the kind-level setting.

### Changed

- Text report redesign: column alignment, label truncation with emphasized target names, spread
  capped at 100%, and compact geomean provenance.
- Legend removed from the default report; the new `--verbose` flag names the statistical method
  behind each verdict in the report footer.
- **Breaking:** JSON schema bumped to `schemaVersion: 2`. `perCandidate[].geomean` is replaced by
  `perCandidate[].kinds[]`, an array of per-kind aggregates carrying section geomean, groups, gated
  geomean, band, and exclusions. Per-metric entries gain `kind` and `group` fields.
- **Breaking:** `--fail-on geomean:<pct>` now evaluates per gating kind rather than on a blended
  cross-kind geomean. A non-gating kind can never trip the gate regardless of its value.
- Group sub-headers in the text report render blue; geomean labels render bold.
- Multi-kind reports prefix highlights with their kind and echo tripped `--fail-on` gates per kind.

### Fixed

- Bench output was lost or garbled: output from processes the bench command spawned never reached
  the adapter, and multi-byte characters split across stream chunks decoded as replacement
  characters. A command that failed to spawn is now a run failure rather than an unhandled error.
- `METRIC name=` with an empty or whitespace-only value (the shape an unset shell variable
  produces) was read as `0` and entered the median as a real measurement; it is now skipped with a
  warning.
- Delta and noise percentages were computed incorrectly for metrics whose median was negative or
  zero, and a zero delta was not always classified as no-signal, producing wrong verdicts.
- Band method returned a false signal from a single paired sample, and a single observation reported
  `± 0%` spread rather than no spread at all.
- Metric names taken from bench output are no longer looked up against `Object.prototype`, so a
  metric named `constructor`, `__proto__`, or `toString` no longer corrupts the report or the config
  merge.
- Mitata alias substitution interpreted replacement metacharacters (`$&`, ``$` ``, `$'`) inside
  argument values; two runs collapsing onto one metric name now warn on stderr instead of silently
  keeping the last.
- `--samples` and `--timeout` silently accepted trailing garbage (`10abc` became `10`), and a
  `--timeout` above roughly 24.8 days overflowed the timer into no timeout at all; both are now
  usage errors.
- A `geomean:<pct>` gate with no stable gating metrics passed silently; the gate now warns on stderr
  that it was never evaluated.
- Reports larger than the pipe buffer (64 KiB on most systems) were truncated when redirected to a
  file or a pipe.
- A `gymrat.json` saved with a UTF-8 byte-order mark failed to parse.
- A `SIGHUP` (closed terminal or dropped SSH session) left temporary worktrees behind; they are
  now swept, and the run exits `129` to match the `SIGINT` and `SIGTERM` handling.
- JSON report nulled a candidate's measured values when the metric had no verdict.
- Report rendering: unit-threshold rounding, special characters in metric names breaking table
  layout, highlighting on wrong occurrence, and spurious quotes in target names.
- Near-miss `METRIC` prefix typos (e.g. `METRICS`, `Metric`) and lines missing the required space
  after the prefix were silently ignored; they now produce a warning.
- Annotated git tags failed to resolve because gymrat did not peel tag objects to the underlying
  commit.
- Progress lines wider than the terminal wrapped and garbled the display; they are now truncated to
  fit.
- Displayed medians could disagree with paired-sample deltas when unpaired observations existed.
- Spread was displayed for zero-median metrics where the percentage is meaningless.
- Error hints took their color from stdout although hints print to stderr.

### Removed

- **Breaking:** `--format markdown` is no longer accepted. Use `--format text` (the default) for
  human-readable output or `--format json` for machine-readable output.

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

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jeffzi/gymrat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jeffzi/gymrat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffzi/gymrat/releases/tag/v0.1.0
