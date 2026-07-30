# gymrat

Standalone A/B benchmark runner with paired sampling and benchstat-style reports.

`gymrat compare` runs a benchmark command against two revisions, alternates samples between them so
both see the same machine noise, and prints a report that tells you, per metric, whether the change
is a real improvement, a real regression, or noise. Verdicts come from a two-sided Wilcoxon
signed-rank test once there are enough samples, and from a noise band below that. No session state,
no config required to start.

## Install

Requires Node ≥ 22 and `git` on your `PATH`.

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

gymrat resolves git refs in the current repository, so run it from inside your project.

## Usage

```text
gymrat compare [label=]<target> [label=]<target> [options]
```

The first positional is the baseline; deltas are measured against it (the report's `vs old`
column). Each target is either a path to an existing directory (used in place, never removed) or a
git ref that gymrat resolves with `git rev-parse` and checks out into a temporary detached worktree.
An existing directory wins over a git ref of the same name, so prefix the ref with `refs/heads/` to
disambiguate.

An optional `label=` prefix sets the display name. Without it, a git target is labelled with its
ref and a path target with the directory's base name, resolved through symlinks. Pass `label=`
when two paths share a base name. The prefix splits at the first `=`, so a target whose own name
contains `=` cannot be passed.

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
| `--timeout <number>` | `1800`          | Timeout in seconds per `prepare` and per bench run |
| `--config <file>`    | `./gymrat.json` | Config file (loaded automatically when present)    |

\*`--bench` is required either on the command line or in the config file.

`gymrat --version` prints the installed version; `gymrat compare --help` prints these options from
the binary.

Sampling is strictly sequential: for each of N windows gymrat runs the bench command in target 1,
then in target 2. Both `bench` and `prepare` run through the shell with the working directory set to
the target's directory.

### Exit codes

- `0`: a report was produced. Verdicts never affect the exit code.
- `1`: an operational error (unresolvable target, nonzero bench/prepare exit, timeout, zero metrics
  parsed, or config error). gymrat surfaces the captured command output so you can see what went
  wrong.
- `130` / `143`: interrupted by `SIGINT` or `SIGTERM`. No report is produced.

gymrat removes temporary worktrees created for git-ref targets on success, on error, and on
`SIGINT`/`SIGTERM`. The report footer states how many were removed and how many were left behind,
naming each leftover directory with the reason git gave. On the signal path there is no report to
carry that footer, so nothing is printed: gymrat kills the running bench command, sweeps the
worktrees, and exits.

## Reading the report

```sh
gymrat compare main perf/faster-decode --bench "node bench.js" --adapter mitata --samples 10
```

With a `gymrat.json` marking `encode/heap` as an exact metric, that prints:

```text
gymrat compare · main ↔ perf/faster-decode · 10 paired samples · adapter: mitata
metric                   │old (main)    │new (perf/faster-decode)  │vs old
─────────────────────────┼──────────────┼──────────────────────────┼──────────────────────────
decode/text=digits/time  │1.735µ ± 1%   │1.425µ ± 1%               │✓ -17.9%  (p=0.002 n=10)
decode/text=words/time   │3.065µ ± 1%   │3.093µ ± 3%               │~ +0.9%  (p=0.49 n=10)
encode/time              │914n ± 1%     │934n ± 1%                 │✗ +2.2%  (p=0.002 n=10)
encode/heap              │48.0k ± 0%    │44.2k ± 0%                │✓ -7.9%  (exact)
─────────────────────────┼──────────────┼──────────────────────────┼──────────────────────────
geomean (gating metrics) │              │                          │-6.0%
verdicts: Wilcoxon signed-rank on pairs (n=10 ≥ 6) · ~ = no signal at α=0.05
2 worktrees removed · 0 left behind
```

- The **delta is always shown**, even under `~`, so "-1.9% but no signal" is visible rather than
  hidden.
- The **glyph is direction-aware**: `✓` improved, `✗` regressed, `~` no signal. You never do
  better-is-higher math yourself.
- The **± spread** cell is the cross-run half-range of the per-run values as a percentage of the
  median, the same dispersion the noise band uses.
- Values **scale to units** only when the adapter supplies one (`mitata` emits `ns`/`bytes`);
  `metric-lines` values carry no unit and are rounded to the nearest integer.
- The **geomean** (geometric mean) row aggregates gating metrics only. All metrics are gating by
  default; disable per metric in the config file.
- A metric present on only one side renders one-sided: its value in the present column, a blank cell
  on the other, and no verdict.

### How verdicts are decided

Per metric, sample window _i_ pairs target-A run _i_ with target-B run _i_. `delta%` is computed
from the per-side medians.

- **Signed-rank** (≥ 6 nonzero differences): a two-sided Wilcoxon signed-rank test. Signal when
  `p < 0.05`.
- **Noise band** (fewer than 6 nonzero differences): the band is
  `max(150 × max(halfRange/median over both sides), 0.5%)`, and `|delta%|` must exceed it to
  count as signal. Rendered as e.g. `~ -1.9% (band ±3.0%, n=4)`. Runs of 6 or more samples land here
  too when ties leave fewer than 6 nonzero differences.
- **Exact metrics** (config-flagged, e.g. binary size): any difference between medians is a signal;
  a single sample suffices.

## The `metric-lines` format

The default adapter, `metric-lines`, is the universal "just printf your numbers" path. No
benchmark library required. gymrat scans bench **stdout** (never stderr) for lines matching:

```text
METRIC <name>=<value>
```

- gymrat trims each line and keeps the ones starting with `METRIC`. Everything after that prefix is
  trimmed and split at its **last** `=`, so metric names may themselves contain `=` (e.g.
  `decode/text=digits`).
- The left side becomes the metric name verbatim, whitespace included. Nothing separates the prefix
  from the name, so a typo like `METRICS foo=1` silently parses as a metric named `S foo`.
  Similarly, `METRIC decode time=1.4` yields a metric named `decode time`. Check the report's metric
  column when a name looks wrong.
- The right side goes through JavaScript's `Number()`, which accepts more than decimal notation
  (`0x1f` parses as 31); only non-finite results are rejected.
- Every non-matching line is **ignored**, so gymrat tolerates arbitrary surrounding output. A line
  starting with `METRIC` whose remainder has no `=`, has an empty name, or has a non-finite value
  emits a warning on gymrat's stderr without failing the run.
- A **repeated name within one run** produces within-run samples: gymrat takes the median of the
  occurrences as the run's value.
- A run in which **zero** metrics are found is an operational error. gymrat aborts and surfaces the
  captured bench output.
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
`decode/text=digits/time`.

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
  }
}
```

- `bench`, `prepare`, `adapter`, `samples`, `timeoutSeconds` mirror the command-line options.
- `unstableNoisePct` (default `200`, any positive number) is the noise band width, in percent, above
  which a metric is too noisy to judge: its verdict becomes no-signal and it drops out of the
  geomean. It has no flag — set it in the config file or leave the default.
- `metrics` keys are exact metric names. Per-metric config overrides the adapter's defaults:
  - `direction`: `"lower"` or `"higher"` (which way is better).
  - `gating`: whether the metric counts toward the geomean. Defaults to `true`.
  - `exact`: when `true`, any median difference is a signal and a single sample suffices.
- A `metrics` key that matches no metric the run produced is ignored without a warning, unlike an
  unknown top-level key. When an override seems to do nothing, check the spelling against the
  report's metric column.

## License

[MIT](LICENSE)
