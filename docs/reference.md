# gymrat reference

Lookup documentation for gymrat's report anatomy, verdict methods, and JSON output. For
installation, usage, and configuration, see the [README](../README.md).

## Report anatomy

The sample below uses `gymrat compare` with a `gymrat.json` marking `encode/heap` as an exact
metric:

```text
gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: metric-lines
metric                      │       main │ perf/faster-decode │ vs main
────────────────────────────┼────────────┼────────────────────┼──────────────────
decode/text=digits/time     │  1700 ± 1% │          1400 ± 1% │ ✓  -17.9%  ±2.5%
decode/text=words/time      │  3100 ± 1% │          3100 ± 3% │ ~   +0.9%  ±2.5%
encode/time                 │   914 ± 1% │           934 ± 1% │ ✗   +2.2%  ±2.5%
encode/heap                 │ 49200 ± 0% │         45300 ± 0% │ ✓   -7.9%
────────────────────────────┼────────────┼────────────────────┼──────────────────
geomean (4 stable metrics)  │            │                    │     -6.0%

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

### Table columns

- The **`±` noise band** in the verdict column is the half-range-derived spread the signed-rank and
  noise-band methods both compute. The band decides the verdict only on the band path; signed-rank
  decides on `p`. It appears only for non-exact metrics in the single-candidate table —
  multi-candidate tables drop the band from cells to save width. Unstable metrics omit the band
  (the word `unstable` replaces it).
- The **delta is always shown**, even under `~`, so "-0.9% but no signal" is visible rather than
  hidden.
- The **glyph is direction-aware**: `✓` improved, `✗` regressed, `≈` unstable, `=` identical, `~`
  within noise, `?` inconclusive. You never do better-is-higher math yourself.
- The **± spread** in the value columns is the cross-run half-range of the per-run values as a
  percentage of the median, the same dispersion the noise band uses. Past 100%, the spread is
  restated in the metric's own units (e.g. `5B ± 381B` instead of `5B ± 7620%`).
- **Value columns vs. delta/verdict:** the value columns show each side's median and spread over
  the windows that reported the metric; the delta and verdict come from paired windows only
  (windows where both sides reported the metric). When a metric is missing from some windows, the
  two sets can differ.
- Values **scale to units** only when the adapter supplies one (`mitata` emits `ns`/`bytes`);
  `metric-lines` values carry no unit and are rounded to the nearest integer.

### Summary and highlights

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

### Geomean rows

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

### Multi-kind example

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
geomean · entity (2)  │            │                    │     -3.1%  ±1.5%

warmup                │ 100ns ± 1% │         100ns ± 1% │ ~   +0.3%  ±2.5%
──────────────────────┼────────────┼────────────────────┼──────────────────
geomean · time (3)    │            │                    │     -3.2%  ±2.0%

informational — gating off (config: kinds.memory.gating = false)
──────────────────────┬────────────┬────────────────────┬──────────────────
memory                │       main │ perf/faster-decode │ vs main
──────────────────────┼────────────┼────────────────────┼──────────────────
encode                │  100B ± 1% │           93B ± 1% │ ✓   -7.0%  ±2.5%
──────────────────────┼────────────┼────────────────────┼──────────────────
geomean · memory (1)  │            │                    │     -7.0%

✓ 2 improved   ✗ 1 regressed   ≈ 0 unstable   = 0 identical   ~ 1 within noise   ? 0 inconclusive

highlights
  ✗ time · entity.spawn         +4.0%
  ✓ time · entity.alive_check  -10.0%
  ✓ memory · encode             -7.0%
```

### Rendering notes

- A metric present on only one side renders one-sided: its value in the present column, a blank cell
  on the other, and no verdict.
- The **`Hint:` line** prints regardless of `--verbose`, and only when a metric fell back to the
  noise band for want of samples. The text reads
  `Hint: re-run with --samples 6 or more for statistical verdicts`.
- **Display-width limitation:** column alignment assumes one character equals one display column.
  Chinese, Japanese, Korean (CJK) or other wide characters in metric names or labels may
  misalign columns; label truncation can split a multi-byte character. A display-width
  dependency is not planned.

## How verdicts are decided

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

## JSON output

### `compare` schema

`gymrat compare --format json` produces a stable JavaScript Object Notation (JSON) structure
(currently `schemaVersion: 2`). Top-level fields:

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

Example, trimmed to one metric and one candidate:

```json
{
  "schemaVersion": 2,
  "baseline": "main",
  "candidates": ["perf/faster-decode"],
  "samples": 10,
  "adapter": "metric-lines",
  "metrics": {
    "decode/text=digits/time": {
      "unit": null,
      "direction": "lower",
      "gating": true,
      "kind": "other",
      "group": null,
      "baseline": { "median": 1700, "spreadPct": 1 },
      "candidates": [
        {
          "label": "perf/faster-decode",
          "median": 1400,
          "spreadPct": 1,
          "verdict": "improved",
          "method": "signed-rank",
          "delta": -17.9,
          "noisePct": 2.5,
          "p": 0.002,
          "band": null
        }
      ]
    }
  },
  "perCandidate": [
    {
      "label": "perf/faster-decode",
      "kinds": [
        {
          "kind": "other",
          "hasGating": true,
          "geomean": { "value": -17.9, "n": 1, "excluded": [], "band": 2.5 },
          "groups": [],
          "gatedGeomean": { "value": -17.9, "n": 1, "excluded": [], "band": 2.5 }
        }
      ],
      "verdictCounts": { "improved": 1, "regressed": 0, "unstable": 0, "noSignal": 0 }
    }
  ],
  "worktrees": { "removed": 0, "leftBehind": [], "pruneError": null }
}
```

### `measure` schema

`gymrat measure --format json` produces its own document, versioned separately (currently
`schemaVersion: 1`), since a single-target measurement has no baseline to pair against and no
candidates to judge:

| Field           | Description                                                                      |
| --------------- | -------------------------------------------------------------------------------- |
| `schemaVersion` | Currently `1`; increments on breaking changes, independently of `compare`'s.     |
| `label`         | The target's label.                                                              |
| `samples`       | Number of samples.                                                               |
| `adapter`       | Adapter used (`metric-lines` or `mitata`).                                       |
| `metrics`       | Per-metric object: `median`, `spreadPct`, `exact` — flat, no `baseline` nesting. |
| `worktrees`     | Cleanup state: removed count, left-behind paths, prune errors.                   |

Example, trimmed to one metric:

```json
{
  "schemaVersion": 1,
  "label": "main",
  "samples": 10,
  "adapter": "metric-lines",
  "metrics": {
    "decode/text=digits/time": {
      "median": 1700,
      "spreadPct": 1,
      "unit": null,
      "direction": "lower",
      "gating": true,
      "kind": "other",
      "group": null,
      "exact": false
    }
  },
  "worktrees": { "removed": 0, "leftBehind": [], "pruneError": null }
}
```

## Session log records

The session log (`.gymrat/session.jsonl`) is a newline-delimited JSON file. Each line is a record
with a `type` field discriminating the record kind. The `finalize` record closes a session:

### `finalize` record

| Field     | Type   | Description                                                            |
| --------- | ------ | ---------------------------------------------------------------------- |
| `type`    | string | Always `"finalize"`.                                                   |
| `at`      | string | ISO 8601 timestamp of the finalize.                                    |
| `branch`  | string | The branch pointing at the squash commit.                              |
| `commit`  | string | The full SHA of the squash commit.                                     |
| `message` | string | The squash commit's message (user-supplied or generated from history). |

```json
{
  "type": "finalize",
  "at": "2026-08-11T14:30:00.000Z",
  "branch": "gymrat/20260811-143000-a1b2-final",
  "commit": "abc123def456789...",
  "message": "gymrat: squash 3 kept iterations\n\ncache the regex\nflatten the lookup\ninline the hot path"
}
```
