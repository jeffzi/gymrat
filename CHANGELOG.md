# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.18.0] - 2026-09-12

### Added

- Add `gymrat probe` to spot-check the experiment worktree against the recorded baseline without
  touching session state.

### Changed

- Append a baseline record from the kept samples when `keep` commits, so the recorded reference
  advances with the session.
- Refuse `keep` on an iteration that did not improve unless `--allow-unimproved` is passed.

### Fixed

- Find the session when a command runs from inside a session worktree.
- Keep the supervise dashboard's session result visible after a tool finishes.
- Report the supervised run's span duration from the run result instead of re-measuring
  wall-clock time.

## [0.17.0] - 2026-09-11

### Added

- Append a `command` record to the session log for every command run inside a session.
- Add a `compaction` event to the supervisor event log marking when the agent's context is compacted.
- Add the `otel` optional extra (`pip install gymrat[otel]`) to export OpenTelemetry spans for every
  command and supervised run when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- Add `gymrat export` to replay a finished session's logs into the same span structure for post-hoc
  analysis.
- Add `session_id` to the supervisor launch event.
- Publish the session and supervisor log formats as JSON Schema, AsyncAPI 3.0, and a Markdown event
  reference under `schemas/` and `docs/`.

### Changed

- **Breaking:** Reshape session log records, supervisor event lines, `--format json` documents, and
  hook payloads: snake_case keys, integer-nanosecond timestamps under `at`, and the header version
  field renamed to `schema`; delete existing logs and event files.
- Make `status` contend for the repository lock, so it waits behind another running `gymrat` command.

### Fixed

- Keep the supervise dashboard's timers running for the whole run.
- Report the action the supervisor took on a cap — in the dashboard, plain output, and supervisor
  event log — instead of always saying "interrupting".
- Stop sending a follow-up to a session a cap is ending.
- Reject a second span processor once tracing is configured.

## [0.16.0] - 2026-09-07

### Added

- Add `follow_up` and `turn_end` lines to the supervisor event log.
- Add a `guard` end reason to the closing summary (exit 1) when the follow-up ceiling, no-progress,
  or consecutive-discard guard trips.

### Changed

- Continue supervised sessions across early turn ends instead of ending the moment the agent ends a
  turn.

### Fixed

- Fire the spend cap on live cost updates instead of waiting for a settled total.
- Interrupt the session instead of ending it outright when a wall-clock or spend cap fires while a
  reply was just sent, giving the agent its grace period.
- Stop resetting the discard streak on a keep that was not committed, so the consecutive-discard
  guard still trips as expected.

## [0.15.0] - 2026-09-06

### Added

- Run a doctor pre-flight before `supervise` launches: any failed check renders the report to stderr
  and exits before any lock is taken.
- Add `supervise --baseline <ref>` to pin a freshly opened session to the given git ref (defaults to
  HEAD; ignored when resuming an existing session).
- Add `gymrat stop -m "<report>"` to record a closing report in the session log without closing the
  session; `stop` also accepts `--format json`.
- Report whether the session is stopped in `gymrat status`, in both the text report and a new
  `stopped` key in JSON output.

### Changed

- Refuse to launch `supervise` when a stop condition is already met (exit 2); `--force` downgrades
  launch refusals to a warning.
- Open the session and record the baseline in `supervise` before the agent starts; the wall-clock
  cap starts only once the baseline is recorded.

### Fixed

- Stop another `gymrat` command from interleaving with `supervise` startup.

## [0.14.0] - 2026-09-05

### Added

- Add the `[supervise]` table in `gymrat.toml` to pin a model and effort level for supervised
  sessions.
- Add `supervise --effort <level>` to set the agent's reasoning effort, overriding the configured
  level.
- Print a time-left line from `iterate`, `keep`, `discard`, `measure`, `compare`, `status`, and
  `sync` when a supervised session has a wall-clock cap active.
- Add a top-level `budget` object to the JSON output of `iterate`, `keep`, `discard`, `measure`,
  `compare`, and `status`.
- Add elapsed duration to baseline records (measurement duration) and iteration records (whole
  iteration duration).
- Add a fingerprint of the experiment worktree at measurement time to iteration records.

### Changed

- Raise the agent's shell-command ceiling to the run's wall-clock cap, so a long measurement runs to
  completion.
- Refuse to launch `supervise` when the wall-clock cap cannot fit one iteration, unless `--force` is
  passed.
- Refuse `iterate` before any hook or bench when the remaining time is smaller than the estimated
  iteration duration.

### Fixed

- Fire the wall-clock cap at the intended clock time even when the machine sleeps mid-run.

## [0.13.0] - 2026-09-03

### Added

- Add a `measured` field to `gymrat discard --format json`.

### Changed

- Report `seq` as `null` in `gymrat discard --format json` for an unmeasured revert.
- Refuse to launch `supervise` when the experiment worktree has moved past the last kept commit.

### Fixed

- Revert unmeasured edits in `gymrat discard` instead of refusing.
- End supervised sessions when the agent finishes — or would have stopped to ask a question —
  instead of streaming idle until the wall-clock cap fires, and show the agent's final message in
  the closing summary.
- Stop firing the spend cap on a supervised session that is already ending on its own.
- Report the lock holder's process, command, and start time reliably when a `gymrat` command is
  blocked by the repository lock.

## [0.12.0] - 2026-09-02

### Added

- Add Windows support.
- Add `gymrat sync` to copy uncommitted main-tree changes into the experiment worktree, refusing
  when that worktree has conflicting uncommitted changes.
- Add `--format json` to `iterate`, `keep`, `discard`, and `status`, with the same stable,
  backward-compatible schema as `compare` and `measure`.
- Add a `--color` flag that forces color output on, overriding `NO_COLOR` and non-TTY detection.

### Changed

- **Breaking:** Rename the distribution and import package to `gymrat`, replacing `gymrat-py` /
  `gymrat_py`; install with `pip install gymrat` and `import gymrat`.
- **Breaking:** Replace `gymrat.json` with `gymrat.toml` using snake_case keys; convert existing
  configs to TOML and re-key them.
- Make `gymrat init` non-interactive, driven by `--bench`, `--no-runbook`, and `--no-skill`.
- Validate the bench configuration in `gymrat doctor` without running a benchmark.
- Switch significance verdicts to an exact sign-flip permutation test instead of Wilcoxon
  signed-rank, so results near the boundary may differ from earlier releases.
- Reduce full measurements per supervised loop: probe edits with `gymrat measure` and run
  `gymrat iterate` only before `keep`.
- **Breaking:** Separate metric kind suffixes with `#` instead of `/`.
- Display verdicts with too few paired samples for the permutation test as inconclusive.
- Refuse `finalize` when the experiment worktree has moved past the last kept commit, hinting to
  keep or discard first.

### Fixed

- Judge metrics dominated by tied samples or with a zero median as `unstable` instead of producing
  a false band verdict.
- Report the actually reverted iteration when `discard` follows a blocked keep.
- Pass metric names containing spaces, parentheses, or quotes intact to the bench command when
  substituted into a `filter` template.
- Accept `--debug` whether written before or after the subcommand.
- End the run quietly on a closed output pipe instead of printing a bug-report footer.
- Read the mitata adapter's report correctly when the bench command prints extra output around the
  JSON, and warn about unusable entries instead of skipping them silently.
- Keep the session log readable and complete after a crash mid-write.
- Report a `gymrat.toml` that is not valid UTF-8 as unreadable and an oversized integer in a
  `GYMRAT_*` environment variable as invalid, instead of crashing.
- Report a clear remedy for a lock file left behind by another user in a shared temporary directory
  instead of failing with a raw permission error.
- Stop leaking `--no-color` into the environment of benchmark commands.
- Handle oversized benchmark output and unresponsive processes at teardown without crashing or
  hanging.

### Removed

- Remove the interactive wizard prompts in `gymrat init` and the `--adapter`, `--checks`,
  `--stop-target`, `--stop-max-iterations`, `--primary`, `--runbook PATH`, and `--yes` flags.
- Remove `gymrat doctor --no-bench` and the smoke benchmark run it used to skip.

## [0.11.0] - 2026-08-25

### Added

- Add `gymrat supervise [prompt]` to run an agent that drives the optimization loop, bounded by a
  wall-clock cap (`--max-minutes`) and an optional spend cap (`--max-usd`).
- Add `gymrat init` to scaffold a project with `gymrat.json`, a runbook stub, and the skill file.
- Add `gymrat doctor` to check the project setup and report grouped findings, exiting non-zero on
  failure.
- Ship the gymrat skill file inside the package.

## [0.10.0] - 2026-08-25

### Added

- Add `gymrat start [ref]` to open or resume an optimization session, pinning the baseline at a ref
  (default `HEAD`).
- Add `gymrat iterate` to measure the experiment worktree against the baseline and report the
  verdict.
- Add `gymrat keep` to commit the measured edit and advance the baseline, refusing when checks fail
  or a gating regression stands.
- Add `gymrat discard` to revert the experiment worktree to its last commit.
- Add `gymrat finalize` to squash the session's kept commits into one commit on the baseline and
  close the session.
- Add `gymrat status` to print the session history.
- Trigger a confirmation rerun for a noisy gating regression before the verdict stands.
- Add lifecycle hooks: `before` and `after` commands run around each measurement.

## [0.9.0] - 2026-08-24

### Added

- Add a `--record` (`-r`) flag to `gymrat measure` to append the run to the open session log as a
  labeled baseline, refusing when no open session exists.

## [0.8.0] - 2026-08-24

### Added

- Add `gymrat compare` to compare a baseline revision against one or more candidates and report as
  text or JSON.
- Add `gymrat measure` to measure a single revision or directory on its own.
- Add a `--fail-on` gate to `compare` that exits non-zero on a gating regression or a
  geometric-mean threshold.
- Add a repository lock to prevent two gymrat runs from colliding.
- Stop the benchmark, remove worktrees, and exit with `128 + N` on cancellation.

## [0.7.0] - 2026-08-23

### Added

- Report a metric-by-metric table with verdicts and a geometric-mean row per kind in `compare`.
- Report each metric's median and spread, grouped by kind, in `measure`.
- Include a verdict summary and a highlights block calling out the metrics that moved most.
- Report the kind and threshold from `--fail-on` when tripped, and show leftover worktrees in a
  closing footer.
- Add `--format json` to `compare` and `measure` for machine-readable output.
- Honor `FORCE_COLOR` and `NO_COLOR` in reports.
- Report each loop iteration's verdict and outcome.

## [0.6.0] - 2026-08-23

### Added

- Read and validate a `gymrat.json` configuration file, reporting the offending key when a value is
  wrong.
- Resolve run configuration from command-line values, `GYMRAT_*` environment variables, the config
  file, and built-in defaults, in that order.
- Resolve per-metric metadata from adapter defaults and per-metric and per-kind overrides.
- Report all configuration problems at once instead of stopping at the first.

## [0.5.0] - 2026-08-23

### Added

- Judge each metric improved, regressed, or unchanged, reporting noisy metrics as unstable.
- Roll up verdicts per kind, per benchmark group, and over gating metrics, excluding metrics that
  cannot be judged.
- Warn when a metric is present in only some rounds, reporting how many were dropped.

### Fixed

- Skip a `METRIC` line with an out-of-range radix value with a warning instead of crashing.

## [0.4.0] - 2026-08-23

### Added

- Collect repeated benchmark measurements for one or more targets, running an optional setup step
  before the timed runs.
- Run benchmark commands under an optional timeout and stop cleanly on cancellation or timeout.
- Run pending cleanup before exiting on Ctrl-C or a signal.

## [0.3.0] - 2026-08-23

### Added

- Add two built-in benchmark-output adapters — `metric-lines` and `mitata` — to parse a bench
  script's output into metrics.

## [0.2.0] - 2026-08-23

### Added

- Add significance testing and summary statistics for paired benchmark samples.

## [0.1.0] - 2026-08-22

### Added

- Add structured error reporting for every gymrat failure: a clear message and, where applicable, an
  actionable hint.

[Unreleased]: https://github.com/jeffzi/gymrat/compare/v0.18.0...HEAD
[0.18.0]: https://github.com/jeffzi/gymrat/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/jeffzi/gymrat/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/jeffzi/gymrat/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/jeffzi/gymrat/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/jeffzi/gymrat/compare/v0.13.0...v0.14.0
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
