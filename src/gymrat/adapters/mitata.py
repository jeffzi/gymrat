"""The ``mitata`` adapter: reads the JSON that ``mitata --json`` writes.

Mitata prints a ``benchmarks`` array whose entries carry an ``alias`` and a list
of ``runs``. Each run becomes ``<alias>#time`` from ``stats.p50`` and, when mitata
measured it, ``<alias>#heap`` from ``stats.heap.avg``. Parameterized benchmarks
carry ``$name`` placeholders in the alias that are substituted with ``name=value``
so one benchmark becomes one metric per argument combination.

The JSON is located by attempting ``json.JSONDecoder().raw_decode`` at each ``{``
position in stdout, so banner text mitata prints around it — including unbalanced
braces like ``cpu: {model`` — does not prevent the real payload from being found.
"""

import json
import math
import re
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    Strict,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

from gymrat.adapters.defaults import defaults_from_suffixes
from gymrat.adapters.types import AdapterError, MetricDefaults, WarnSink, warn_to_stderr
from gymrat.metric_name import LINE_TERMINATORS
from gymrat.pydantic_errors import (
    UNKNOWN_SHAPE_PHRASE,
    describe_key,
    drop_prefix_errors,
    phrase_for_error,
)

_JSON_DECODER = json.JSONDecoder()


def _scan_json_objects(text: str) -> tuple[list[str], str | None]:
    """Scan ``text`` for JSON objects in a single pass.

    Each ``{`` in ``text`` is tried via ``raw_decode``.  Successes become
    candidates; the failure spanning the most remaining text is captured so
    ``_extract_json`` can surface an actionable diagnostic without a second scan.

    Args:
        text: The bench output to scan.

    Returns:
        A ``(candidates, longest_failure)`` pair: valid JSON text slices, and the
        error message from the longest failed attempt (or ``None``).
    """
    candidates: list[str] = []
    longest: tuple[int, str] | None = None
    length = len(text)
    pos = text.find("{")
    while pos != -1 and pos < length:
        try:
            _, end = _JSON_DECODER.raw_decode(text, pos)
        except (json.JSONDecodeError, RecursionError) as exc:
            remaining = length - pos
            reason = (
                str(exc)
                if isinstance(exc, json.JSONDecodeError)
                else f"Exceeded maximum recursion depth while parsing at position {pos}"
            )
            if longest is None or remaining > longest[0]:
                longest = (remaining, reason)
            pos = text.find("{", pos + 1)
            continue
        candidates.append(text[pos:end])
        pos = text.find("{", end)
    return candidates, longest[1] if longest is not None else None


def find_json_candidates(text: str) -> list[str]:
    """Scan ``text`` for valid JSON objects using :meth:`json.JSONDecoder.raw_decode`.

    For each ``{`` in ``text``, attempts a full JSON parse starting at that
    position. Unbalanced braces and banner text like ``cpu: {model}`` are
    rejected by the JSON decoder itself, so no hand-rolled brace-balancing
    scanner is needed.

    Args:
        text: The raw text to scan for JSON objects.

    Returns:
        Original text slices of the objects that parsed; positions that fail
        are skipped.
    """
    candidates, _ = _scan_json_objects(text)
    return candidates


def _extract_json(stdout: str) -> dict[str, object]:
    """Find mitata's JSON object using :func:`_scan_json_objects`.

    Candidates from :func:`_scan_json_objects` are already valid JSON (parsed
    via ``raw_decode``). A candidate carrying a ``benchmarks`` list wins over any
    earlier record that does not — a decoy object printed before mitata's own
    output must not shadow the real payload.

    When no candidate has a ``benchmarks`` list but a decode failure exists
    alongside a non-benchmarks record, the failure diagnostic takes priority
    over returning the record — the real payload was likely truncated or
    malformed, and the parse error is more actionable than a generic "missing
    benchmarks array" from the caller.

    When :func:`_scan_json_objects` returns no candidates, every ``{`` in
    ``stdout`` failed to start a valid JSON object. The diagnostic names the
    failure of the longest attempt — the ``{`` spanning the most remaining text
    is most likely to be the real payload.

    Args:
        stdout: The bench command's captured stdout.

    Returns:
        The parsed JSON object carrying a ``benchmarks`` list, or, when no
        decode failed, the first dict-shaped record as a fallback.

    Raises:
        AdapterError: When no usable JSON object is found, or the most
            promising candidate failed to parse.
    """
    candidates, longest_failure = _scan_json_objects(stdout)

    first_record: dict[str, object] | None = None
    for candidate in candidates:
        # Every candidate starts at a ``{``, so a successful raw_decode can only
        # have produced a JSON object — no non-dict shape check is needed.
        parsed: dict[str, object] = json.loads(candidate)
        if isinstance(parsed.get("benchmarks"), list):
            return parsed
        if first_record is None:
            first_record = parsed

    if first_record is not None:
        if longest_failure is not None:
            msg = f"Failed to parse JSON: {longest_failure}"
            raise AdapterError(msg)
        return first_record
    if longest_failure is not None:
        msg = f"Failed to parse JSON: {longest_failure}"
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

    Args:
        metrics: The readings collected so far, updated in place.
        name: The resolved metric name.
        value: The reading to store.
        warn: The sink that receives the collision warning.
    """
    if name in metrics:
        warn(
            f"Duplicate metric name: {name} (keeping the last value; "
            "give the benchmark aliases distinct $placeholders to separate the runs)"
        )
    metrics[name] = value


def _escape_line_terminator(match: re.Match[str]) -> str:
    return json.dumps(match.group())[1:-1]


def _describe_run_error(error: object) -> str:
    r"""Render mitata's ``run.error`` for a warning message.

    ``error`` is read from parsed JSON, so it can be any JSON value, not just a
    string — ``str()`` on a plain dict would print an unhelpful Python repr.
    ``json.dumps`` renders that case usefully instead, and cannot fail since the
    value round-trips from ``json.loads``.

    A string is written as-is so a plain message reads naturally, except that
    each character of ``LINE_TERMINATORS`` becomes its JSON escape (``\n``,
    ``\u000b``, ``\f``, ``\r``, ``\u001c`` to ``\u001e``, ``\u0085``,
    ``\u2028``, ``\u2029``): the warning must stay on one line.
    ``json.dumps`` keeps ``ensure_ascii`` on, so a non-string value is already
    escaped the same way.

    Args:
        error: The ``run.error`` value as parsed from JSON.

    Returns:
        A single-line, human-readable rendering of the error value.
    """
    if isinstance(error, str):
        return LINE_TERMINATORS.sub(_escape_line_terminator, error)
    return json.dumps(error)


def _serialize_arg_value(value: object) -> str:
    """Serialize a run-argument value for inclusion in a metric name.

    Primitives keep a JavaScript ``String()`` form so booleans read ``true``/
    ``false`` and ``None`` reads ``null``; objects and arrays serialize via JSON
    with recursively sorted keys so two structurally equal objects always produce
    the same metric name.

    Args:
        value: The argument value as parsed from JSON.

    Returns:
        The serialized string representation suitable for metric names.
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
    r"""Substitute each ``$key`` in ``alias`` with ``key=value`` via a single regex pass.

    Keys are matched longest-first to prevent a shorter key from consuming a
    prefix of a longer one (``$ab`` matches key ``ab`` before ``a``).  The
    callable replacement returns a literal string, so ``re.sub`` never interprets
    backslash escapes (``\1``, ``\g<0>``, ``\n``) in argument values.
    Unmatched ``$`` tokens stay as-is because the alternation only covers keys
    present in ``args``.

    Args:
        alias: The benchmark alias, possibly carrying ``$key`` placeholders.
        args: The run's arguments, keyed by name.

    Returns:
        The alias with ``$key`` placeholders replaced by ``key=value`` pairs.
    """
    if not args:
        return alias
    pattern = re.compile(
        r"\$(" + "|".join(re.escape(k) for k in sorted(args, key=len, reverse=True)) + ")"
    )

    def _repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}={_serialize_arg_value(args[m.group(1)])}"

    return pattern.sub(_repl, alias)


def _keep_integer(value: object, handler: ValidatorFunctionWrapHandler) -> float:
    number: float = handler(value)
    # Strict float validation widens an int to float; the int is kept so a
    # whole-number reading is written back as ``42``, not ``42.0``.
    return value if isinstance(value, int) else number


_Number = Annotated[float, Strict(), WrapValidator(_keep_integer)]
"""A JSON number; strict, so ``bool`` is rejected as JavaScript's ``typeof`` would."""

_FINITE_NUMBER_PHRASE = "a finite number"
_HEAP_LOC = ("stats", "heap")


class _Heap(BaseModel):
    """The ``stats.heap`` object mitata writes when it measured allocations."""

    avg: _Number | None = None


class _Stats(BaseModel):
    """A run's ``stats`` object.

    ``heap`` is left loose here and validated against :class:`_Heap` on its own,
    so a malformed heap costs only the ``#heap`` metric, never the ``#time`` one.
    """

    p50: _Number
    heap: object = None


class _Run(BaseModel):
    """One entry of a benchmark's ``runs`` array."""

    # A run that never varied its arguments can omit ``args`` entirely.
    args: dict[str, object] = Field(default_factory=dict)
    stats: _Stats


class _Benchmark(BaseModel):
    """One entry of the top-level ``benchmarks`` array."""

    alias: Annotated[str, Strict()]
    runs: list[object]


def _invalid_value_tail(key: str, phrase: str, value: object) -> str:
    """Phrase a rejected value as the tail of a skip warning.

    Args:
        key: The dotted path of the rejected value; empty for the entry itself.
        phrase: What was expected, such as ``"a number"``.
        value: The rejected value.

    Returns:
        The text that follows the skipped entry's name in the warning.
    """
    detail = f"expected {phrase}, got {json.dumps(value)}"
    return f" with invalid {key}: {detail}" if key else f": {detail}"


def _first_problem(exc: ValidationError, prefix: tuple[str, ...] = ()) -> str:
    """Phrase the first problem pydantic found as the tail of a skip warning.

    Args:
        exc: The validation failure.
        prefix: The path of the validated value inside the run, prepended to
            each error location.

    Returns:
        The text that follows the skipped entry's name in the warning.
    """
    error = drop_prefix_errors(exc.errors())[0]
    key = describe_key((*prefix, *(str(part) for part in error["loc"])))
    if error["type"] == "missing":
        return f" with missing {key}"
    return _invalid_value_tail(key, phrase_for_error(error) or UNKNOWN_SHAPE_PHRASE, error["input"])


def _warn_skip(warn: WarnSink, subject: str, tail: str) -> None:
    """Send a "Skipping <subject><tail>" warning.

    Args:
        warn: The sink that receives the warning.
        subject: What is being skipped, e.g. ``'run of "my-bench"'``. Any alias it
            names is JSON-escaped, so a line terminator in the alias cannot split the
            warning.
        tail: The reason, from :func:`_invalid_value_tail` or :func:`_first_problem`.
    """
    warn(f"Skipping {subject}{tail}")


def _resolve_metric_prefix(alias: str, args: dict[str, object], warn: WarnSink) -> str | None:
    prefix = _build_metric_name_prefix(alias, args)
    if "#" in prefix:
        msg = (
            f"Metric prefix {json.dumps(prefix)} contains '#', which is reserved as the "
            f"metric-type separator (alias: {json.dumps(alias)})"
        )
        raise AdapterError(msg)
    if LINE_TERMINATORS.search(prefix):
        # Check every entry of LINE_TERMINATORS, not just ``\n`` and ``\r``: mitata's JSON
        # can carry any of them inside an alias or an argument value.
        # json.dumps keeps ensure_ascii on so U+0085, U+2028 and U+2029 are escaped
        # like the ASCII terminators, keeping the warning on one line.
        warn(
            f"Skipping run with a line terminator in its metric name: {json.dumps(alias)} "
            "(the alias or one of its argument values carries one)"
        )
        return None
    return prefix


def _record_heap_metric(
    heap: object,
    prefix: str,
    alias: str,
    metrics: dict[str, float],
    warn: WarnSink,
) -> None:
    """Record ``<prefix>#heap`` from ``stats.heap.avg`` when mitata measured it."""
    if heap is None:
        return
    subject = f"heap metric of {json.dumps(alias)}"
    try:
        avg = _Heap.model_validate(heap).avg
    except ValidationError as exc:
        _warn_skip(warn, subject, _first_problem(exc, _HEAP_LOC))
        return
    if avg is None:
        return
    if not math.isfinite(avg):
        _warn_skip(warn, subject, _invalid_value_tail("stats.heap.avg", _FINITE_NUMBER_PHRASE, avg))
        return
    _record_metric(metrics, f"{prefix}#heap", avg, warn)


def _extract_run_metrics(
    run: object, alias: str, metrics: dict[str, float], warn: WarnSink
) -> None:
    # An errored run carries no usable stats, so the error is the one problem
    # worth reporting — checked before the shape, which it would otherwise fail.
    if isinstance(run, dict) and run.get("error") is not None:
        error = _describe_run_error(run["error"])
        warn(f"Skipping run with an error: {json.dumps(alias)} ({error})")
        return

    subject = f"run of {json.dumps(alias)}"
    try:
        parsed = _Run.model_validate(run)
    except ValidationError as exc:
        _warn_skip(warn, subject, _first_problem(exc))
        return

    p50 = parsed.stats.p50
    if not math.isfinite(p50):
        _warn_skip(warn, subject, _invalid_value_tail("stats.p50", _FINITE_NUMBER_PHRASE, p50))
        return

    prefix = _resolve_metric_prefix(alias, parsed.args, warn)
    if prefix is None:
        return

    _record_metric(metrics, f"{prefix}#time", p50, warn)
    _record_heap_metric(parsed.stats.heap, prefix, alias, metrics, warn)


def _extract_benchmark_metrics(
    benchmark: object, metrics: dict[str, float], warn: WarnSink
) -> None:
    try:
        parsed = _Benchmark.model_validate(benchmark)
    except ValidationError as exc:
        alias = benchmark.get("alias") if isinstance(benchmark, dict) else None
        label = f" {json.dumps(alias)}" if isinstance(alias, str) else ""
        _warn_skip(warn, f"benchmark{label}", _first_problem(exc))
        return

    for run in parsed.runs:
        _extract_run_metrics(run, parsed.alias, metrics, warn)


class _MitataAdapter:
    """Adapter for bench scripts that print the JSON ``mitata --json`` writes."""

    name = "mitata"

    def defaults(self, metric_name: str) -> MetricDefaults:
        """Return name-derived defaults for ``metric_name`` via suffix matching."""
        return defaults_from_suffixes(metric_name)

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse mitata's JSON output into a metric map.

        Each run yields ``<alias>#time`` from ``stats.p50`` and, when mitata
        measured it, ``<alias>#heap`` from ``stats.heap.avg``. Runs that errored,
        reported a non-finite ``p50``, resolved to a metric name carrying a line
        terminator, or carried a malformed ``args``/``stats`` shape are skipped
        rather than failing the parse — a single bad argument combination should
        not discard the rest of the run — and likewise for a benchmark whose
        ``alias``/``runs`` shape is malformed. Every skip warns through ``warn``
        rather than vanishing silently, as does a collision between two runs
        landing on one metric name; on a collision the last run still wins. A
        metric prefix carrying ``#`` is the exception: ``#`` is reserved as the
        metric-type separator, so it aborts the parse instead.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send a complaint about a skipped run or benchmark;
                defaults to stderr.

        Returns:
            One value per metric name.

        Raises:
            AdapterError: When no JSON object is found, the JSON is malformed, the
                ``benchmarks`` array is missing or empty, a substituted metric
                prefix contains ``#``, or no run yields a usable metric.
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


mitata_adapter = _MitataAdapter()
"""The singleton ``mitata`` adapter instance callers register and invoke."""
