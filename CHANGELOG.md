# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-08-10

### Added

- Session loop: five commands that turn a one-shot comparison into an optimization session.
  `gymrat start [ref]` pins a baseline and creates a pair of persistent worktrees — one baseline,
  one experiment — to work in; `gymrat iterate` benches the experiment against the baseline and
  reports a verdict; `gymrat keep` commits the measured edit and advances the baseline to it;
  `gymrat discard` reverts the experiment worktree; `gymrat status` reads the whole session back.
  Every measurement and decision is appended to a per-repository JSONL log, so a session survives
  across processes and an agent can resume one it did not start.
- `gymrat keep` refuses to commit in three cases, each recorded in the log rather than thrown away:
  nothing measured since the last `keep` or `discard`, a gating metric that regressed, and a failing
  `checks` command. A refused keep exits 1.
- Confirmation rerun: an iteration that regresses a gating metric is re-measured once, and a
  regression the rerun will not reproduce is demoted to no signal. Exact metrics never take part.
- Loop configuration keys in `gymrat.json`: `checks` (the command `keep` must see pass), `filter`
  (a bench command template whose `{names}` placeholder narrows a confirmation rerun to the
  regressed metrics), `primary` (the figure each iteration is read on — the geomean by default, or
  a named metric), `stop` (`maxIterations` and `targetValue`, which end the loop), and `hooks`
  (`before`/`after` commands run around each iteration with a JSON payload on stdin).
- `gymrat measure --record` appends the run to the session log as a baseline, so `gymrat status`
  reads it back alongside the session's iterations. This supersedes 0.3.0's note that `measure`
  runs are ephemeral.

### Changed

- The minimum supported Node version is now 22.12, the floor gymrat's dependencies already required.
  Installing on 22.0 through 22.11 reports an `EBADENGINE` warning instead of failing later at
  runtime.
- Every command that runs consumer commands or writes session state now holds a per-repository
  lock, `compare` and `measure` included. A second gymrat run against the same repository exits 2
  with `Lock held by PID …` rather than benchmarking alongside the first — concurrent runs perturb
  each other's measurements. `gymrat status` only reads the log, so it takes no lock.

### Fixed

- A stale lockfile left by a crashed run could be taken over by two processes at once: both read the
  same dead holder, and the second rename displaced the first's live lockfile. The takeover now
  claims exclusive rights via a hard link before displacing anything, so only one racer wins. A run
  killed mid-takeover leaves the lock permanently blocked; the error names the lockfile and the
  blocking claim link to delete.
- Lock error hints no longer advise waiting for a process that may already be dead, and no longer
  suggest deleting a lockfile whose holder was probed alive.
- A bench command could outlive the run that started it. When one of its output streams failed,
  gymrat settled the call without killing the process group, leaving the benchmark running against
  a worktree the run had already finished with.
- `bench` and `prepare` accepted an empty string, from the config file or from `--bench ""` /
  `--prepare ""`, and an empty `--bench` was reported as a missing one. Both are now rejected where
  they are given, as `checks` and the hook commands already were.
- A `gymrat.json` whose top-level key was the empty string reported `Unknown config key:` with
  nothing after the colon.
- A target that existed but could not be examined — a symlink loop, an unreadable parent directory
  — surfaced a raw `ELOOP: too many symbolic links` instead of the usual
  `Cannot resolve target '<name>'` with its hint.
- The `metric-lines` adapter only ended a line on a newline, so a progress-style bench that wrote
  `50%\rMETRIC name=value` lost the metric, and a metric name could carry a bare carriage return
  into the session log. Lines now end on a carriage return too, and a name containing U+2028 or
  U+2029 is skipped with the usual parse warning rather than producing a record gymrat cannot read
  back.
- The `mitata` adapter printed its duplicate-metric warning itself instead of routing it like every
  other warning, which spliced it into the progress line; and a benchmark reporting a non-finite
  `p50` or heap average entered the medians and the geomean. Such runs are now skipped, as other
  unusable runs already were.
- Two comparison runs in one process left the first run's signal handling in charge, so a Ctrl-C
  during the second run exited without sweeping its worktrees — and each run added another handler,
  eventually tripping Node's max-listeners warning. Only library consumers could reach this; the
  CLI runs one comparison per process.
- A negative value with a unit rendered in the smallest tier (`-1500000B` rather than `-1.5MB`).
- A metric whose name contains the word "unstable" stole the color meant for the verdict beside it.
- Truncating a long branch label could split a multi-byte character, leaving half of it in the
  report.
- A benchmark reporting a few hundred thousand samples crashed with a `RangeError` while computing
  its spread; and a non-finite sample was silently dropped from that computation instead of making
  the spread undefined.
- `gymrat --help` printed unframed, while every subcommand's help was boxed.
- The warning that a `--fail-on geomean:<pct>` gate had no stable metrics to measure could be lost
  when the run exited.

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

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/jeffzi/gymrat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jeffzi/gymrat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jeffzi/gymrat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffzi/gymrat/releases/tag/v0.1.0
