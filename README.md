# gymrat

Standalone A/B benchmark runner with paired sampling and benchstat-style reports.

`gymrat compare` runs a benchmark command against two revisions, alternates samples between them so
both see the same machine noise, and prints a report that tells you — per metric — whether the
change is a real improvement, a real regression, or noise. Verdicts come from a two-sided Wilcoxon
signed-rank test once there are enough samples, and from a noise band below that. No session state,
no config required to start.

## Install

gymrat is not yet published to npm. For now, install from source:

```sh
git clone https://github.com/jeffzi/gymrat.git
cd gymrat
npm install
npm run build
npm link
```

`npm link` puts `gymrat` on your `PATH`. Then, from inside the project you want to benchmark:

```sh
gymrat compare main my-branch --bench "npm run bench"
```

Requires Node ≥ 22. gymrat resolves git refs in the current repository, so run it from inside your
project.

## Usage

```text
gymrat compare [label=]<target> [label=]<target> [options]
```

The first positional is the baseline; deltas are measured against it (the report's `vs old`
column). Each target is either a git ref (resolved with `git rev-parse` and checked out into a
temporary detached worktree) or a path to an existing directory (used in place, never removed). An
optional `label=` prefix sets the display name; the default label is the target string itself.

```sh
# Compare two git refs
gymrat compare main perf/faster-decode --bench "npm run bench"

# Label the columns
gymrat compare old=main new=perf/faster-decode --bench "npm run bench"

# Build each revision before benchmarking, take 20 samples, parse mitata JSON
gymrat compare main my-branch \
  --prepare "npm ci && npm run build" \
  --adapter mitata \
  --samples 20
```

### Options

| Option               | Default         | Description                                        |
| -------------------- | --------------- | -------------------------------------------------- |
| `--bench <cmd>`      | — (required\*)  | Bench command run in each target directory         |
| `--prepare <script>` | none            | Per-target setup, e.g. `"npm ci && npm run build"` |
| `--adapter <type>`   | `metric-lines`  | Output parser: `metric-lines` or `mitata`          |
| `--samples <number>` | `10`            | Paired samples per target                          |
| `--timeout <number>` | `1800`          | Timeout in seconds per bench invocation            |
| `--config <file>`    | `./gymrat.json` | Config file (loaded automatically when present)    |

\*`--bench` is required either on the command line or in the config file.

Sampling is strictly sequential: for each of N windows, gymrat runs the bench command in target 1,
then in target 2. Both `bench` and `prepare` run through the shell with the working directory set to
the target's directory.

### Exit codes

- `0` — a report was produced. Verdicts never affect the exit code.
- `1` — an operational error: an unresolvable target, a nonzero bench/prepare exit, a timeout, zero
  metrics parsed, or a config error. The captured command output is surfaced so you can see what
  went wrong.

Temporary worktrees created for git-ref targets are removed on success, on error, and on
`SIGINT`/`SIGTERM`; the report footer confirms they were removed and reports how many (if any) were
left behind.

## Reading the report

```text
$ gymrat compare old=main new=perf/faster-decode --samples 10

gymrat compare · main ↔ perf/faster-decode · 10 paired samples · adapter: mitata

metric                        │ old (main)   │ new (faster-decode) │ vs old
──────────────────────────────┼──────────────┼─────────────────────┼─────────────────────────
decode/text=digits time       │ 1.726µ ± 1%  │ 1.423µ ± 2%         │ ✓ -17.5%  (p=0.002 n=10)
decode/text=words  time       │ 3.070µ ± 2%  │ 3.081µ ± 3%         │ ~         (p=0.62  n=10)
encode             time       │  912n  ± 1%  │  934n  ± 1%         │ ✗  +2.4%  (p=0.014 n=10)
decode             heap_bytes │ 48.0k        │ 44.2k               │ ✓  -7.9%  (exact)
──────────────────────────────┼──────────────┼─────────────────────┼─────────────────────────
geomean (gating metrics)      │              │                     │   -5.8%

verdicts: Wilcoxon signed-rank on pairs (n=10 ≥ 6) · ~ = no signal at α=0.05
worktrees removed · 0 left behind
```

- The **delta is always shown**, even under `~`, so "-1.9% but no signal" is visible rather than
  hidden.
- The **glyph is direction-aware**: `✓` improved, `✗` regressed, `~` no signal. You never do
  better-is-higher math yourself.
- The **± spread** cell is the cross-run half-range of the per-run values as a percentage of the
  median — the same dispersion the noise band uses.
- Values **scale to units** only when the adapter supplies one (`mitata` emits `ns`/`bytes`);
  `metric-lines` values render raw.
- The **geomean** row aggregates gating metrics only.
- A metric present on only one side renders one-sided — its value in the present column, a blank
  cell on the other, and no verdict.

### How verdicts are decided

Per metric, sample window _i_ pairs target-A run _i_ with target-B run _i_. `delta%` is computed
from the per-side medians.

- **Signed-rank** (≥ 6 nonzero pairs): a two-sided Wilcoxon signed-rank test. Signal when
  `p < 0.05`. Below the 6-pair floor gymrat falls back to the band method.
- **Noise band** (< 6 pairs): `band% = max(1.5 × 100 × max(halfRange/median over both sides),
0.5%)`. Signal when `|delta%|` exceeds the band. Rendered as e.g. `~ -1.9% (band ±3.0%, n=4)`.
- **Exact metrics** (config-flagged, e.g. binary size): any difference between medians is a signal;
  a single sample suffices.

## The `metric-lines` format

The default adapter, `metric-lines`, is the universal "just printf your numbers" path — no
benchmark library required. gymrat scans bench **stdout** (never stderr) for lines matching:

```text
METRIC <name>=<value>
```

- Optional leading/trailing whitespace; the literal `METRIC`, then whitespace, then `name=value`.
- The **last** `=` splits name from value, so metric names may themselves contain `=` (e.g.
  `decode/text=digits`). Names are non-empty and contain no whitespace; any other character is
  allowed. Values are finite decimal numbers — optional sign, fraction, and exponent (`-12`,
  `3.14`, `1e-9`).
- Every non-matching line is **ignored**, so gymrat tolerates arbitrary surrounding output. A line
  that starts with `METRIC` but fails to parse emits a warning on gymrat's stderr (to catch typos)
  without failing the run.
- A **repeated name within one run** is treated as within-run samples: the run's value is the median
  of the occurrences.
- A run in which **zero** metrics are found is an operational error — gymrat aborts and surfaces the
  captured bench output.
- Every metric defaults to **lower-is-better** and renders raw (no units). Override direction per
  metric in the config file.

Example bench script:

```sh
#!/bin/sh
# Print one `METRIC name=value` line per metric; replace the values with your
# real measurements. gymrat takes the median across runs.
echo "METRIC decode/time=1.42"
echo "METRIC encode/time=0.91"
```

The `mitata` adapter parses the JSON that [mitata](https://github.com/evanwashere/mitata) prints in
its JSON mode, flattening each benchmark to `alias[/arg=value]*/measure` names with `time` (p50) and
`heap_bytes` metrics.

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
  "metrics": {
    "decode/time": { "direction": "lower", "gating": true, "exact": false }
  }
}
```

- `bench`, `prepare`, `adapter`, `samples`, `timeoutSeconds` mirror the command-line options.
- `metrics` keys are exact metric names. Per-metric config overrides the adapter's defaults:
  - `direction` — `"lower"` or `"higher"` (which way is better).
  - `gating` — whether the metric counts toward the geomean. Defaults to `true`.
  - `exact` — when `true`, any median difference is a signal and a single sample suffices.

## License

[MIT](LICENSE)
