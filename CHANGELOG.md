# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ETA countdown displayed between samples during comparison runs, with interpolation between updates.
- `identical` display class (`=`) in the text report for metrics whose samples all tied, splitting
  them out from `within noise`. Presentation only — the JSON report, `--fail-on`, and geomean gating
  still see these as `no signal`.
- A `Hint:` line, printed regardless of `--verbose`, when a metric fell back to the noise-band
  method for want of samples and more samples would buy a statistical verdict.
- `inconclusive` display class (`?`) for metrics resting on too few paired samples for the noise
  band to carry signal. These metrics print no `±` band, fold with the quiet rows, and tally in
  their own summary bucket instead of inflating `within noise`. Presentation only — the JSON report,
  `--fail-on`, and geomean gating still see `no signal`.
- Geomean and sub-geomean rows in the single-candidate text table print their propagated noise band
  (`±N.N%`) after the delta, so the reader sees the threshold the figure was judged against.
- Header reads `1 paired sample` (singular) for single-sample runs.
- Byte-metric noise floor: for metrics measured in whole bytes, the reported noise percentage never
  falls below one byte relative to each median, preventing a one-byte quantization flip from
  producing a double-digit verdict.
- Per-kind metric sections: when a run produces metrics of more than one kind (e.g. `time` and
  `memory` from `mitata`), reports render a section per kind with its own geomean and group
  sub-geomeans for dotted metric names. Single-kind runs keep the flat layout unchanged.
- `kinds` config key for per-kind gating: `"kinds": { "memory": { "gating": false } }` switches an
  entire kind to informational — its metrics stay in the report but leave the geomean and the
  `--fail-on` gate. Per-metric `gating` overrides still win over the kind-level setting.

### Changed

- Text report redesign: sub-field column alignment, label truncation with emphasized target names,
  reshaped headers, a dimmed echo row under the geomean, spread capped at 100% with a futility note
  for unstable metrics, compact `(N/M)` geomean provenance, and bold-only geomean rows when every
  constituent metric is within noise.
- Legend removed from the default report; the new `--verbose` flag names the statistical method
  behind each verdict in the report footer.
- **Breaking:** JSON schema bumped to `schemaVersion: 2`. `perCandidate[].geomean` is replaced by
  `perCandidate[].kinds[]`, an array of per-kind aggregates carrying section geomean, groups, gated
  geomean, band, and exclusions. Per-metric entries gain `kind` and `group` fields.
- **Breaking:** `--fail-on geomean:<pct>` now evaluates per gating kind rather than on a blended
  cross-kind geomean. A non-gating kind can never trip the gate regardless of its value.
- Group sub-headers in the text report render blue; geomean labels render bold.
- Multi-kind highlights prefix each metric with its kind (`✗ time · encode  +2.2%`); single-kind
  runs are unchanged.
- When `--fail-on geomean:<pct>` would trip, highlights echo the tripped gate per kind
  (`⚑ time geomean +3.1% exceeded --fail-on geomean:2`), so the reader sees why the run exits 1.

### Fixed

- Bench output was lost or garbled: output from processes the bench command spawned never reached
  the adapter, and multi-byte characters split across stream chunks decoded as replacement
  characters. A command that failed to spawn is now a run failure rather than an unhandled error.
- `METRIC name=` with an empty or whitespace-only value (the shape an unset shell variable
  produces) was read as `0` and entered the median as a real measurement; it is now skipped with a
  warning.
- Delta and noise percentages were computed incorrectly for metrics whose median was negative or
  zero, producing wrong verdicts for those metrics.
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
- Report rendering: values just below a unit threshold printed in the wrong unit (999.5 bytes as
  `1000B`), pipes and backticks in metric names and labels broke markdown tables and code spans,
  highlighting landed on the wrong occurrence of a repeated name, and target names carried spurious
  quotes in plain text.
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

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jeffzi/gymrat/releases/tag/v0.1.0
