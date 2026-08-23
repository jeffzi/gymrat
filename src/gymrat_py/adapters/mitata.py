"""The ``mitata`` adapter: reads the JSON that ``mitata --json`` writes.

Mitata prints a ``benchmarks`` array whose entries carry an ``alias`` and a list
of ``runs``. Each run becomes ``<alias>/time`` from ``stats.p50`` and, when mitata
measured it, ``<alias>/heap`` from ``stats.heap.avg``. Parameterized benchmarks
carry ``$name`` placeholders in the alias that are substituted with ``name=value``
so one benchmark becomes one metric per argument combination.

The JSON is located by scanning stdout for balanced top-level ``{...}`` slices, so
banner text mitata prints around it — including bare braces like ``cpu: {model}``
— does not have to be stripped by the caller.
"""

import json
import math
import re

from gymrat_py.adapters.defaults import defaults_from_suffixes
from gymrat_py.adapters.types import AdapterError, MetricDefaults, WarnSink, warn_to_stderr

_FORBIDDEN_NAME_CHARS = re.compile("[\\n\\r\\u2028\\u2029]")
"""Characters a metric name may not carry.

All four are line terminators to a JavaScript regular-expression engine, so an
anchored name check on the session record could never match a name holding one —
gymrat must never write a record it cannot read back. Unlike ``metric-lines``,
whose input is split on ``\\n`` and ``\\r`` before any name is read, mitata's JSON
can carry every one of them inside an alias or an argument value. Written as
escapes so the source stays plain ASCII.
"""


def _skip_quoted_string(text: str, pos: int) -> int:
    """Advance past a double-quoted string, starting one position after the opening quote.

    Returns the index of the closing ``"``, or the end of ``text`` when the string
    is unterminated.
    """
    i = pos
    length = len(text)
    while i < length and text[i] != '"':
        if text[i] == "\\":
            i += 1
        i += 1
    return i


def find_json_candidates(text: str) -> list[str]:
    """Scan ``text`` for balanced top-level ``{...}`` slices.

    Quote skipping only applies once a candidate is open (brace depth > 0). A
    stray unpaired ``"`` in banner text before any ``{`` — a plain apostrophe-style
    quote in prose, say — is otherwise indistinguishable from the start of a
    string and would swallow everything up to the next ``"`` it finds, frequently a
    quote inside the real JSON payload that follows. Outside a candidate there is
    no string to skip: banner text is not JSON, so a bare quote there is just a
    character like any other.

    Every balanced slice is returned as-is — no JSON validation is performed.
    """
    candidates: list[str] = []
    depth = 0
    start = -1
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == '"':
            if depth > 0:
                i = _skip_quoted_string(text, i + 1)
            i += 1
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                candidates.append(text[start : i + 1])
                start = -1
        i += 1
    return candidates


def _extract_json(stdout: str) -> dict[str, object]:
    """Find mitata's JSON object by parsing each balanced ``{...}`` candidate.

    A candidate carrying a ``benchmarks`` list wins over any earlier record that
    parses but does not — a decoy object printed before mitata's own output must
    not shadow the real payload. When no candidate has a ``benchmarks`` list, the
    first record that parsed is returned anyway, so the caller's "missing
    benchmarks array" error still names the right cause instead of a generic parse
    failure.

    When every candidate fails to parse, the diagnostic names the failure of the
    longest candidate rather than whichever was tried last — the longest balanced
    slice is the one most likely to be the actual payload rather than banner
    residue.
    """
    first_record: dict[str, object] | None = None
    longest_failure: tuple[str, str] | None = None

    for candidate in find_json_candidates(stdout):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if longest_failure is None or len(candidate) > len(longest_failure[0]):
                longest_failure = (candidate, str(exc))
            continue
        if not isinstance(parsed, dict):
            continue
        if isinstance(parsed.get("benchmarks"), list):
            return parsed
        if first_record is None:
            first_record = parsed

    if first_record is not None:
        return first_record
    if longest_failure is not None:
        msg = f"Failed to parse JSON: {longest_failure[1]}"
        raise AdapterError(msg)
    msg = "No JSON object found in stdout"
    raise AdapterError(msg)


def _parse_benchmarks(json_obj: dict[str, object]) -> list[object]:
    benchmarks = json_obj.get("benchmarks")
    if not isinstance(benchmarks, list):
        msg = "JSON missing benchmarks array"
        raise AdapterError(msg)
    if not benchmarks:
        msg = "benchmarks array is empty"
        raise AdapterError(msg)
    return benchmarks


def _record_metric(metrics: dict[str, float], name: str, value: float, warn: WarnSink) -> None:
    """Store ``value`` under ``name``, warning when it displaces an earlier reading.

    A collision means two runs resolved to one metric name — an alias missing the
    ``$placeholder`` for the argument that varies, or two benchmarks sharing an
    alias — so the report would otherwise silently show only the last run's value.
    """
    if name in metrics:
        warn(
            f"Duplicate metric name: {name} (keeping the last value; "
            "give the benchmark aliases distinct $placeholders to separate the runs)"
        )
    metrics[name] = value


def _describe_run_error(error: object) -> str:
    """Render mitata's ``run.error`` for a warning message.

    ``error`` is read from parsed JSON, so it can be any JSON value, not just a
    string — ``str()`` on a plain dict would print an unhelpful Python repr.
    ``json.dumps`` renders that case usefully instead, with a ``str()`` fallback
    for values it cannot serialize.
    """
    if isinstance(error, str):
        return error
    try:
        return json.dumps(error)
    except (TypeError, ValueError):
        return str(error)


def _serialize_arg_value(value: object) -> str:
    """Serialize a run-argument value for inclusion in a metric name.

    Primitives keep a JavaScript ``String()`` form so booleans read ``true``/
    ``false`` and ``None`` reads ``null``; objects and arrays serialize via JSON
    with recursively sorted keys so two structurally equal objects always produce
    the same metric name.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        # JS String(5.0) is "5", not "5.0"; an int falls through to str() below.
        return str(int(value))
    return str(value)


def _build_metric_name_prefix(alias: str, args: dict[str, object]) -> str:
    """Substitute each ``$key`` in ``alias`` with ``key=value``, scanning the alias once.

    Both the single scan and the literal splice are deliberate. A regex-style
    replace would read ``$&``, ``$```, ``$'`` and ``$<n>`` in an argument value —
    which is user data — as patterns. Repeated passes would be just as wrong the
    other way: an earlier value containing ``$b`` would be eaten by the pass for
    key ``b``. Substituted text is written straight to the output, never scanned again.

    Keys are matched longest-first, so an alias of ``$ab`` picks the argument
    ``ab`` over the argument ``a``.
    """
    keys = sorted(args, key=len, reverse=True)

    result: list[str] = []
    cursor = 0
    dollar = alias.find("$")
    while dollar != -1:
        key = next((k for k in keys if alias.startswith(k, dollar + 1)), None)
        if key is None:
            result.append(alias[cursor : dollar + 1])
            cursor = dollar + 1
        else:
            result.append(alias[cursor:dollar] + f"{key}={_serialize_arg_value(args[key])}")
            cursor = dollar + 1 + len(key)
        dollar = alias.find("$", cursor)
    result.append(alias[cursor:])
    return "".join(result)


def _resolve_p50(stats: dict[str, object], alias: str, warn: WarnSink) -> float | None:
    """Read ``stats.p50``, warning and returning ``None`` when it is unusable.

    ``bool`` is excluded from the numeric check: mitata never emits one, and
    Python's ``bool`` is an ``int`` subclass, so treating it as non-numeric mirrors
    JavaScript's ``typeof p50 !== "number"``.
    """
    p50 = stats.get("p50")
    if not isinstance(p50, (int, float)) or isinstance(p50, bool):
        warn(f"Skipping run with malformed stats shape: {alias} (stats.p50 is not a number)")
        return None
    if not math.isfinite(p50):
        warn(f"Skipping run with non-finite p50: {alias} ({p50})")
        return None
    return p50


def _resolve_metric_prefix(alias: str, args: dict[str, object], warn: WarnSink) -> str | None:
    prefix = _build_metric_name_prefix(alias, args)
    if _FORBIDDEN_NAME_CHARS.search(prefix):
        warn(
            f"Skipping run with a line terminator in its metric name: {alias} "
            "(the alias or one of its argument values carries one)"
        )
        return None
    return prefix


def _record_heap_metric(
    stats: dict[str, object], prefix: str, metrics: dict[str, float], warn: WarnSink
) -> None:
    """Record ``<prefix>/heap`` from ``stats.heap.avg`` when mitata measured it."""
    heap = stats.get("heap")
    if not isinstance(heap, dict):
        return
    avg = heap.get("avg")
    if isinstance(avg, (int, float)) and not isinstance(avg, bool) and math.isfinite(avg):
        _record_metric(metrics, f"{prefix}/heap", avg, warn)


def _extract_run_metrics(
    run: object, alias: str, metrics: dict[str, float], warn: WarnSink
) -> None:
    if not isinstance(run, dict):
        return

    if "error" in run and run["error"] is not None:
        warn(f"Skipping run with an error: {alias} ({_describe_run_error(run['error'])})")
        return

    # A run that never varied its arguments can omit ``args`` entirely, so absence
    # is tolerated as an empty record rather than a malformed shape.
    args: object = run.get("args", {})
    if not isinstance(args, dict):
        warn(f"Skipping run with malformed args shape: {alias} (args is not a record)")
        return

    stats = run.get("stats")
    if not isinstance(stats, dict):
        warn(f"Skipping run with malformed stats shape: {alias} (stats is not a record)")
        return

    p50 = _resolve_p50(stats, alias, warn)
    if p50 is None:
        return

    prefix = _resolve_metric_prefix(alias, args, warn)
    if prefix is None:
        return

    _record_metric(metrics, f"{prefix}/time", p50, warn)
    _record_heap_metric(stats, prefix, metrics, warn)


def _extract_benchmark_metrics(
    benchmark: object, metrics: dict[str, float], warn: WarnSink
) -> None:
    if not isinstance(benchmark, dict):
        return

    alias = benchmark.get("alias")
    if not isinstance(alias, str):
        warn(
            f"Skipping benchmark with malformed alias: {json.dumps(alias)} (alias is not a string)"
        )
        return

    runs = benchmark.get("runs")
    if not isinstance(runs, list):
        warn(f"Skipping benchmark with malformed runs shape: {alias} (runs is not an array)")
        return

    for run in runs:
        _extract_run_metrics(run, alias, metrics, warn)


class _MitataAdapter:
    """Adapter for bench scripts that print the JSON ``mitata --json`` writes."""

    name = "mitata"

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse mitata's JSON output into a metric map.

        Each run yields ``<alias>/time`` from ``stats.p50`` and, when mitata
        measured it, ``<alias>/heap`` from ``stats.heap.avg``. Runs that errored,
        reported a non-finite ``p50``, resolved to a metric name carrying a line
        terminator, or carried a malformed ``args``/``stats`` shape are skipped
        rather than failing the parse — a single bad argument combination should
        not discard the rest of the run — and likewise for a benchmark whose
        ``alias``/``runs`` shape is malformed. Every skip warns through ``warn``
        rather than vanishing silently, as does a collision between two runs
        landing on one metric name; on a collision the last run still wins.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send a complaint about a skipped run or benchmark;
                defaults to stderr.

        Returns:
            One value per metric name.

        Raises:
            AdapterError: When no JSON object is found, the JSON is malformed, the
                ``benchmarks`` array is missing or empty, or no run yields a usable
                metric.
        """
        json_obj = _extract_json(stdout)
        benchmarks = _parse_benchmarks(json_obj)
        metrics: dict[str, float] = {}
        for benchmark in benchmarks:
            _extract_benchmark_metrics(benchmark, metrics, warn)

        if not metrics:
            msg = "No valid benchmark runs found"
            raise AdapterError(msg)

        return metrics

    def defaults(self, metric_name: str) -> MetricDefaults:
        """Return name-derived defaults for ``metric_name`` via suffix matching."""
        return defaults_from_suffixes(metric_name)


mitata_adapter = _MitataAdapter()
"""The singleton ``mitata`` adapter instance callers register and invoke."""
