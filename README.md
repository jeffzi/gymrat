# gymrat

Standalone A/B benchmark runner with paired sampling and benchstat-style reports.

`gymrat compare` runs a benchmark command against a baseline revision and one or more candidates,
cycling samples across them so every target sees the same machine noise, and prints a report that
tells you, per metric, whether each candidate is a real improvement, a real regression, or noise.
Verdicts come from a two-sided Wilcoxon signed-rank test once there are enough samples, and from a
noise band below that. No session state, no config required to start.

## Install

Requires Node ≥ 22 and `git` on your `PATH`.

```sh
npm install -g gymrat
```

Then, from inside the project you want to benchmark:

```sh
gymrat compare main my-branch --bench "npm run bench"
```

gymrat resolves git refs in the current repository, so run it from inside your project.

## Usage

```text
gymrat compare [label=]<baseline> [label=]<candidate>... [options]
```

The first positional is the baseline; every later positional is a candidate, and each one is judged
against the baseline alone — candidates are never compared with each other. Deltas are measured
against the baseline (the report's `vs <baseline>` column). Each target is either a path to an
existing directory (used in place, never removed) or a git ref that gymrat resolves with
`git rev-parse` and checks out into a temporary detached worktree. An existing directory wins over a
git ref of the same name, so prefix the ref with `refs/heads/` to disambiguate.

An optional `label=` prefix sets the display name. Without it, a git target is labelled with its
ref and a path target with the directory's base name, resolved through symlinks. Pass `label=`
when two paths share a base name. The prefix splits at the first `=`, so `label=a=b` passes a
target containing `=`; a bare `a=b` without a label cannot.

```sh
# Compare two git refs
gymrat compare main perf/faster-decode --bench "npm run bench"

# Judge several candidates against one baseline
gymrat compare main perf/simd perf/lookup-table --bench "npm run bench"

# Label the columns
gymrat compare old=main new=perf/faster-decode --bench "npm run bench"

# Build each revision before benchmarking, take 20 samples, parse mitata JSON
gymrat compare main my-branch \
  --prepare "npm ci && npm run build" \
  --adapter mitata \
  --samples 20
```

### Options

| Option                  | Default         | Description                                                               |
| ----------------------- | --------------- | ------------------------------------------------------------------------- |
| `--bench <cmd>`         | — (required\*)  | Bench command run in each target directory                                |
| `--prepare <script>`    | none            | Per-target setup, e.g. `"npm ci && npm run build"`                        |
| `--adapter <type>`      | `metric-lines`  | Output parser: `metric-lines` or `mitata`                                 |
| `--samples <number>`    | `10`            | Paired samples per target                                                 |
| `--timeout <number>`    | `1800`          | Timeout in seconds per `prepare` and per bench run                        |
| `--config <file>`       | `./gymrat.json` | Config file (loaded automatically when present)                           |
| `--format <value>`      | `text`          | Output format: `text` or `json`                                           |
| `--no-color`            | auto            | Print the report without ANSI styles                                      |
| `--verbose`             | off             | Name the statistical method behind each verdict in the footer             |
| `--fail-on <condition>` | none            | Exit 1 when a condition trips (repeatable; see [Exit codes](#exit-codes)) |

\*`--bench` is required either on the command line or in the config file.

`gymrat --version` prints the installed version; `gymrat compare --help` prints these options from
the binary.

Sampling is strictly sequential: for each of N windows gymrat runs the bench command once per
target, baseline first, then each candidate in the order given. Both `bench` and `prepare` run
through the shell with the working directory set to the target's directory.

### Exit codes

- `0`: a report was produced and no `--fail-on` gate tripped.
- `1`: a `--fail-on` gate tripped. The full report is printed before the exit, so you can inspect
  the results.
- `2`: an operational error (unresolvable target, nonzero bench/prepare exit, timeout, zero metrics
  parsed, config error, or invalid usage). gymrat surfaces the captured command output so you can
  see what went wrong.
- `129` / `130` / `143`: interrupted by `SIGHUP`, `SIGINT`, or `SIGTERM`. No report is produced.

#### `--fail-on` conditions

`--fail-on` is repeatable. Each condition is checked against every candidate independently:

- `regressed` — trips when any gating metric has a `regressed` verdict on any candidate. Unstable
  and non-gating metrics never trip this gate.
- `geomean:<pct>` — trips when any candidate's gated geomean, on any gating **kind** (the class an
  adapter assigns a metric, such as `time` or `memory`; see [Reading the report](#reading-the-report)
  for the full explanation), reaches `<pct>` percent in the costly direction (e.g. `geomean:2` trips
  at +2.0% or worse for lower-is-better metrics). Each kind is evaluated independently: a non-gating
  kind can never trip this gate regardless of its value. A candidate whose gating kinds all have
  zero stable metrics cannot trip this gate; gymrat warns on stderr instead.

```sh
# Block on any regression
gymrat compare main my-branch --bench "npm run bench" --fail-on regressed

# Block on regressions or geomean drift above 2%
gymrat compare main my-branch --bench "npm run bench" \
  --fail-on regressed --fail-on geomean:2
```

gymrat removes temporary worktrees created for git-ref targets on success, on error, and on
`SIGINT`/`SIGTERM`. The report footer states how many were removed and how many were left behind,
naming each leftover directory with the reason git gave. On the signal path there is no report to
carry that footer, so nothing is printed: gymrat kills the running bench command, sweeps the
worktrees, and exits.

## Reading the report

```sh
gymrat compare main perf/faster-decode --bench "node bench.js" --adapter metric-lines --samples 10
```

With a `gymrat.json` marking `encode/heap` as an exact metric, that prints:

```text
gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: metric-lines
metric                      │        main │ perf/faster-decode │ vs main
────────────────────────────┼─────────────┼────────────────────┼──────────────────
decode/text=digits/time     │   1700 ± 1% │          1400 ± 1% │ ✓  -17.9%  ±2.5%
decode/text=words/time      │   3100 ± 1% │          3100 ± 3% │ ~   +0.9%  ±2.5%
encode/time                 │    914 ± 1% │           934 ± 1% │ ✗   +2.2%  ±2.5%
encode/heap                 │  49200 ± 0% │         45300 ± 0% │ ✓   -7.9%
────────────────────────────┼─────────────┼────────────────────┼──────────────────
geomean (4 stable metrics)  │             │                    │     -6.0%
                            │        main │ perf/faster-decode │ vs main

✓ 2 improved   ✗ 1 regressed   ≈ 0 unstable   = 0 identical   ~ 1 within noise   ? 0 inconclusive

highlights
  ✗ encode/time               +2.2%
  ✓ decode/text=digits/time  -17.9%
  ✓ encode/heap               -7.9%  (exact)
```

Add `--verbose` to name the method behind each verdict in the footer:

```text
verdicts: Wilcoxon signed-rank on pairs (n=10 ≥ 6) · ~ = no signal at α=0.05
```

**Anatomy:**

- The **summary line** (`✓ 2 improved  ✗ 1 regressed ...`) tallies every verdict class at a glance,
  and doubles as the legend for the glyphs used throughout the report.
- The **highlights** block lists regressions first, then improvements, with the delta and method
  evidence. Exact metrics show `(exact)`.
- Metrics marked `≈ unstable` (noise band wider than `unstableNoisePct`) show the noise in the
  metric's own units (`±<noise> noise on a <median> median`); the `noise ±N%` form appears only
  when the relative spread stays below 100%. Unstable metrics are too jittery to judge and are
  excluded from the geomean. A candidate with an unstable metric closes the block with a note that
  unstable metrics won't stabilize with more samples.
- In a multi-kind run, each highlight is prefixed with its kind (`✗ time · encode  +2.2%`);
  single-kind runs omit the prefix.
- When `--fail-on geomean:<pct>` would trip, the highlights close with a **gate-trip echo** per
  tripping kind (`⚑ time geomean +3.1% exceeded --fail-on geomean:2`), so the reader sees why the
  run will exit 1 without cross-referencing the gate conditions.
- The **`±` noise band** in the verdict column is the half-range-derived spread the signed-rank and
  noise-band methods both compute. The band decides the verdict only on the band path; signed-rank
  decides on `p`. It appears only for non-exact metrics in the single-candidate table —
  multi-candidate tables drop the band from cells to save width. Unstable metrics omit the band
  (the word `unstable` replaces it).
- The **delta is always shown**, even under `~`, so "-0.9% but no signal" is visible rather than
  hidden.
- The **glyph is direction-aware**: `✓` improved, `✗` regressed, `≈` unstable, `=` identical, `~`
  within noise. You never do better-is-higher math yourself.
- The **± spread** in the value columns is the cross-run half-range of the per-run values as a
  percentage of the median, the same dispersion the noise band uses. Past 100%, the spread is
  restated in the metric's own units (e.g. `5B ± 381B` instead of `5B ± 7620%`).
- **Value columns vs. delta/verdict:** the value columns show each side's median and spread over
  the windows that reported the metric; the delta and verdict come from paired windows only
  (windows where both sides reported the metric). When a metric is missing from some windows, the
  two sets can differ.
- Values **scale to units** only when the adapter supplies one (`mitata` emits `ns`/`bytes`);
  `metric-lines` values carry no unit and are rounded to the nearest integer.
- When a run produces metrics of more than one **kind** (e.g. `time` and `memory` from `mitata`),
  the report renders a section per kind: a kind heading, group sub-headers for dotted metric names,
  a per-group geomean, and a per-kind geomean at the bottom of each section. A kind with no gating
  metric carries an `informational — gating off` tag on its heading; when a `kinds` entry in the
  config file switched the kind off wholesale, the tag also names the key, e.g.
  `informational — gating off (config: kinds.memory.gating = false)`. Single-kind runs keep the flat
  layout unchanged.
- The **geomean** row in each kind section aggregates all of that kind's metrics. Unstable metrics
  are excluded automatically. All metrics are gating by default; disable per metric or per kind in
  the config file. The `--fail-on geomean:<pct>` gate evaluates a separate gated geomean per kind
  that covers only gating metrics — there is no cross-kind blended geomean. When every constituent
  metric is within noise, the geomean renders bold-only (no green/red coloring) regardless of its
  propagated value.
- A single-kind run closes with one `geomean (4 stable metrics)` row naming how many stable metrics
  remain. A multi-kind run closes each section with a `geomean · <kind> (n)` row instead, where `n`
  is the count of stable metrics that stood behind the figure, or `geomean · <kind> (n/m)` when
  exclusions thinned `m` metrics down to `n`. The geomean cell shows `±<band>%` — the propagated
  noise — when the aggregate band is nonzero. A band of zero (all-exact metrics) omits it.

A `mitata` run reporting both `time` and `memory` metrics, with `kinds.memory.gating` set to
`false`, sections the report like this:

```text
gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata

──────────────────────┬────────────┬────────────────────┬──────────────────
time                  │       main │ perf/faster-decode │ vs main
──────────────────────┼────────────┼────────────────────┼──────────────────
entity                │            │                    │
  alive_check         │ 100ns ± 1% │          90ns ± 1% │ ✓  -10.0%  ±2.5%
  spawn               │ 100ns ± 1% │         104ns ± 1% │ ✗   +4.0%  ±2.5%
geomean · entity (2)  │            │                    │     -3.1%

warmup                │ 100ns ± 1% │         100ns ± 1% │ ~   +0.3%  ±2.5%
──────────────────────┼────────────┼────────────────────┼──────────────────
geomean · time (3)    │            │                    │     -3.2%
                      │       main │ perf/faster-decode │ vs main

informational — gating off (config: kinds.memory.gating = false)
──────────────────────┬────────────┬────────────────────┬──────────────────
memory                │       main │ perf/faster-decode │ vs main
──────────────────────┼────────────┼────────────────────┼──────────────────
encode                │  100B ± 1% │           93B ± 1% │ ✓   -7.0%  ±2.5%
──────────────────────┼────────────┼────────────────────┼──────────────────
geomean · memory (1)  │            │                    │     -7.0%
                      │       main │ perf/faster-decode │ vs main

✓ 2 improved   ✗ 1 regressed   ≈ 0 unstable   = 0 identical   ~ 1 within noise   ? 0 inconclusive

highlights
  ✗ time · entity.spawn/time         +4.0%
  ✓ time · entity.alive_check/time  -10.0%
  ✓ memory · encode/heap             -7.0%
```

- A metric present on only one side renders one-sided: its value in the present column, a blank cell
  on the other, and no verdict.
- The **`Hint:` line** prints regardless of `--verbose`, and only when a metric fell back to the
  noise band for want of samples.
- **Display-width limitation:** column alignment assumes one character equals one display column.
  Chinese, Japanese, Korean (CJK) or other wide characters in metric names or labels may
  misalign columns; label truncation can split a multi-byte character. A display-width
  dependency is not planned.

### How verdicts are decided

Per metric, sample window _i_ pairs target-A run _i_ with target-B run _i_. `delta%` is computed
from the per-side medians.

- **Signed-rank** (≥ 6 nonzero differences): a two-sided Wilcoxon signed-rank test. Signal when
  `p < 0.05`.
- **Noise band** (fewer than 6 nonzero differences): the band is
  `max(150% × max(halfRange/median over both sides), 0.5%, byteFloorPct)`, and `|delta%|` must
  exceed it to count as signal. `byteFloorPct` applies only to byte-valued metrics (`bytes` unit):
  it is the percentage one byte represents against each side's median, ensuring a one-step
  quantization move is never called a signal. With fewer than 2 pairs, the band is meaningless and
  the metric reads _inconclusive_ regardless of delta — the band has no observable spread to measure
  against. Rendered as e.g. `~  -1.9%  ±3.0%  n=4` (glyph, delta, band, and pair count when it
  differs from `--samples`). Runs of 6 or more samples land here too when ties leave fewer than 6
  nonzero differences.
- **Exact metrics** (config-flagged, e.g. binary size): any difference between medians is a signal;
  a single sample suffices.

## CI integration

### GitHub Actions

Use `--fail-on` to gate CI on regressions, and `--format json` for machine-readable output:

```yaml
- name: Benchmark comparison
  run: |
    gymrat compare main ${{ github.head_ref }} \
      --bench "npm run bench" \
      --format json \
      --fail-on regressed \
      --fail-on geomean:2 \
      > bench-report.json
```

### JSON schema

`--format json` produces a stable JSON structure (currently `schemaVersion: 2`). Top-level fields:

| Field           | Description                                                                              |
| --------------- | ---------------------------------------------------------------------------------------- |
| `schemaVersion` | Currently `2`; increments on breaking changes.                                           |
| `baseline`      | Baseline label.                                                                          |
| `candidates`    | Ordered array of candidate labels.                                                       |
| `samples`       | Number of paired samples.                                                                |
| `adapter`       | Adapter used (`metric-lines` or `mitata`).                                               |
| `metrics`       | Per-metric object: baseline medians, per-candidate verdicts and deltas.                  |
| `perCandidate`  | Per-candidate `kinds` array (section geomean, groups, gated geomean) and verdict counts. |
| `worktrees`     | Cleanup state: removed count, left-behind paths, prune errors.                           |

Each metric entry carries `kind` (adapter-supplied, e.g. `"time"`, `"memory"`, or `"other"`) and
`group` (the dotted-name prefix, or `null` for ungrouped metrics).

Each candidate's `kinds` array contains one aggregate object per kind, with fields:

- `kind` — the kind name.
- `hasGating` — whether at least one metric of the kind counts toward `--fail-on geomean:<pct>`.
- `geomean` — over all of the kind's metrics: `value`, `n` (metrics included), `band` (propagated
  noise), and `excluded` (excluded metrics with their reasons).
- `groups` — one `{ group, geomean }` entry per dotted-name prefix, `geomean` shaped as above.
- `gatedGeomean` — the same shape as `geomean` but over gating metrics only; `null` when `hasGating`
  is `false`.

Each candidate's verdict includes `method` (`signed-rank`, `band`, or `exact`), `delta`, `p` (for
signed-rank), `band` (for band), and `noisePct`. Fields that don't apply to a method are `null`.
A `NaN` delta (zero baseline median, non-zero candidate) serializes as `null`; it is distinguished
from a missing verdict by the non-null `verdict` field.

## The `metric-lines` format

The default adapter, `metric-lines`, is the universal "just printf your numbers" path. No
benchmark library required. gymrat scans bench **stdout** (never stderr) for lines matching:

```text
METRIC <name>=<value>
```

- gymrat trims each line and keeps the ones starting with `METRIC` (note the mandatory space after
  the prefix). Everything after the prefix is trimmed and split at its **last** `=`, so metric names
  may themselves contain `=` (e.g. `decode/text=digits`).
- The left side becomes the metric name verbatim, whitespace included. `METRIC decode time=1.4`
  yields a metric named `decode time`. Check the report's metric column when a name looks wrong.
- A line starting with `METRIC` but missing the separating space — `METRICS foo=1`,
  `METRICbar=2`, `METRIC_foo=1` — produces a warning on gymrat's stderr, since it looks like a
  near-miss. Completely unrelated lines are silently ignored.
- The right side goes through JavaScript's `Number()`, which accepts more than decimal notation
  (`0x1f` parses as 31); non-finite results are rejected.
- An **empty right side** (`METRIC name=`, the shape an unset shell variable produces) is skipped,
  not read as `0`, so a missing measurement never enters the median as a real reading. A right side
  of only whitespace is the same case.
- Every non-matching line is **ignored**, so gymrat tolerates arbitrary surrounding output. A line
  starting with `METRIC` whose remainder has no `=`, has an empty name, or has an empty or
  non-finite value emits a warning on gymrat's stderr without failing the run.
- A **repeated name within one run** produces within-run samples: gymrat takes the median of the
  occurrences as the run's value.
- A run in which **zero** metrics are found is an operational error (exit 2; see
  [Exit codes](#exit-codes)).
- Every metric defaults to **lower-is-better** and is rounded to the nearest integer, with no unit
  scaling. Emit nanoseconds or microseconds rather than fractional seconds, or every value collapses
  to `0` or `1`. Override direction per metric in the config file.

Example bench command:

```sh
#!/bin/sh
# Print one `METRIC name=value` line per metric; replace the values with your
# real measurements. Values here are nanoseconds, since gymrat rounds
# `metric-lines` values to whole numbers. gymrat takes the median across runs.
echo "METRIC decode/time=1420"
echo "METRIC encode/time=910"
```

The `mitata` adapter parses the JSON that [mitata](https://github.com/evanwashere/mitata) prints in
its JSON mode, flattening each benchmark to `<alias>/time` (from `stats.p50`) and `<alias>/heap`
(from `stats.heap.avg`, when mitata measured it). For parameterized benchmarks, `$name` placeholders
in the alias are replaced with `name=value`, so an alias of `decode/$text` becomes
`decode/text=digits/time`. `/time` metrics report under the `time` kind and `/heap` metrics under
`memory`; `metric-lines` names no kind, so all its metrics report under `other`.

## Configuration

gymrat loads `./gymrat.json` automatically when present (override with `--config <path>`). All keys
are optional. Precedence is **flags > config file > built-in defaults**. Unknown top-level keys are
an error, to catch typos.

```json
{
  "bench": "npm run bench",
  "prepare": "npm ci && npm run build",
  "adapter": "mitata",
  "samples": 10,
  "timeoutSeconds": 1800,
  "unstableNoisePct": 200,
  "metrics": {
    "decode/time": { "direction": "lower", "gating": true, "exact": false }
  },
  "kinds": {
    "memory": { "gating": false }
  }
}
```

- `bench`, `prepare`, `adapter`, `samples`, `timeoutSeconds` mirror the command-line options.
- `unstableNoisePct` (default `200`, minimum `0.5`) is the noise band width, in percent, above which
  a metric is too noisy to judge: its verdict becomes `unstable` (tallied under `≈` in the summary,
  serialized as `"unstable"` in JSON) and it drops out of the geomean. The comparison is strict — a
  metric sitting exactly on the threshold keeps its verdict. Values below `0.5` are rejected because
  the noise band is floored at 0.5%, so a lower threshold would mark every metric unstable. It has no
  flag — set it in the config file or leave the default.
- `metrics` keys are exact metric names. Per-metric config overrides the adapter's defaults:
  - `direction`: `"lower"` or `"higher"` (which way is better).
  - `gating`: whether the metric counts toward the geomean and the `--fail-on` gate. Defaults to
    `true`. A per-metric `gating` entry takes precedence over the kind-level setting.
  - `exact`: when `true`, any median difference is a signal and a single sample suffices.
- A `metrics` key that matches no metric the run produced is ignored without a warning, unlike an
  unknown top-level key. When an override seems to do nothing, check the spelling against the
  report's metric column.
- `kinds` keys are kind names reported by the adapter (`time`, `memory` for `mitata`; `other` for
  `metric-lines`). Each entry accepts:
  - `gating`: whether every metric of this kind counts toward the geomean and the `--fail-on` gate.
    Defaults to `true`. A per-metric `gating` entry in `metrics` overrides the kind-level setting.
- A `kinds` key that matches no kind the run produced is silently ignored, the same as an unmatched
  `metrics` key.
- An unknown sub-key inside a `metrics` or `kinds` entry is an error, the same as an unknown
  top-level key.

## License

[MIT](LICENSE)
