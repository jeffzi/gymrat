"""The built-in bench-harness adapters and their closed registry.

An adapter turns a bench harness's stdout into gymrat's metric map. Two ship:

- ``metric-lines`` reads ``METRIC name=value`` lines. A bench script prints one
  such line per sample; the adapter warns about every line it cannot read and
  reduces repeated names to their median, so a run with several samples yields
  one value per name.
- ``mitata`` reads the JSON that ``mitata --json`` writes: a ``benchmarks`` array
  whose entries carry an ``alias`` and a list of ``runs``. Each run becomes
  ``<alias>#time`` from ``stats.p50`` and, when mitata measured it,
  ``<alias>#heap`` from ``stats.heap.avg``. Parameterized benchmarks carry
  ``$name`` placeholders in the alias that are substituted with ``name=value``, so
  one benchmark becomes one metric per argument combination. The JSON is located
  by attempting ``json.JSONDecoder().raw_decode`` at each ``{`` position in
  stdout, so banner text mitata prints around it — including unbalanced braces
  like ``cpu: {model`` — does not prevent the real payload from being found.

Every adapter satisfies the :class:`Adapter` contract. :class:`MetricDefaults` is
what an adapter knows about a metric from its name alone. An adapter sends a
complaint about each part of the output it cannot read cleanly to a
:data:`~gymrat.utils.WarnSink`; most such parts (a line, a run, a benchmark, a
duplicate name, a metric name with an empty path segment or an empty kind) are
skipped, so every name an adapter emits follows the grammar of
:mod:`gymrat.metric_name`. It raises :class:`AdapterError`, aborting the whole
parse, when the output yields no usable metric or a metric name with more than one
``#``.

Both derive name-based metric defaults from one suffix table.

The set of adapters is fixed at import time: nothing registers an adapter at
runtime, so :func:`get_adapter` is a lookup in a closed mapping rather than a
plugin dispatch. A name that misses the mapping is a user typo, not a missing
plugin, so the raised :class:`~gymrat.errors.GymratError` carries a ``hint``
listing every valid name — the same inventory the CLI shows when it reports what
adapters are available.
"""

import json
import math
import re
import statistics
from dataclasses import dataclass
from typing import Annotated, Final, Protocol

from pydantic import (
    BaseModel,
    Field,
    Strict,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

from gymrat.errors import GymratError
from gymrat.metric_name import MetricNameError, MultipleHashesError
from gymrat.metric_name import parse as parse_metric_name
from gymrat.model import Direction, MetricUnit
from gymrat.pydantic_errors import describe_key, drop_prefix_errors, phrase_for_error
from gymrat.utils import LINE_TERMINATORS, WarnSink, expected_got, finite_or_none, warn_to_stderr

# ---------------------------------------------------------------------------
# adapter contract
# ---------------------------------------------------------------------------


class AdapterError(GymratError):
    """A bench script produced output the adapter could not read.

    The class itself is the signal: the CLI error formatter matches on it to
    prefix the message with the class name, which is what tells the user the
    fault is in their bench script's output rather than in gymrat's git or
    config handling. Adapters raise this and nothing else for unparseable
    output.
    """


@dataclass(frozen=True, slots=True)
class MetricDefaults:
    """What an adapter knows about a metric from its name alone.

    Attributes:
        direction: Whether a lower or higher raw value is the better outcome.
        unit: The metric's physical unit, or ``None`` when the adapter cannot
            tell, in which case the report prints the raw value rather than
            scaling it.
        kind: The label grouping metrics an adapter emits for the same benchmark
            (a harness may emit both a ``time`` and a ``memory`` metric per
            benchmark), or ``None`` when the adapter cannot tell.
        short_name: The benchmark's name with the kind suffix stripped, or
            ``None`` when the adapter cannot tell, leaving the full metric name
            as the only thing the report can show.
    """

    direction: Direction
    unit: MetricUnit | None = None
    kind: str | None = None
    short_name: str | None = None


class Adapter(Protocol):
    """Turns a benchmark harness's stdout into gymrat's metric map.

    Conformance is structural: any object exposing ``name``, ``parse``, and
    ``defaults`` satisfies the protocol without inheriting from it.

    ``defaults`` is consulted once per metric name during config resolution, and
    only for fields the user's config does not override.

    Attributes:
        name: The adapter's identifier, as named by the ``adapter`` config key.
    """

    name: str

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse ``stdout`` into a metric map, routing complaints to ``warn``.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send complaints about individual unreadable lines.

        Returns:
            One value per metric name.

        Raises:
            AdapterError: When ``stdout`` yields no usable metric, or yields a
                metric name the adapter cannot accept (such as one carrying
                more than one ``#``); either aborts the whole parse.
        """
        ...

    def defaults(self, metric_name: str) -> MetricDefaults:
        """Return name-derived defaults for ``metric_name``."""
        ...


# ---------------------------------------------------------------------------
# name-based metric defaults
# ---------------------------------------------------------------------------


_METRIC_SUFFIXES: Final[tuple[tuple[str, MetricUnit, str], ...]] = (
    ("#time", "ns", "time"),
    ("#heap", "bytes", "memory"),
)
"""Suffix→(unit, kind) table walked by :func:`defaults_from_suffixes`; first match wins."""


def defaults_from_suffixes(metric_name: str) -> MetricDefaults:
    """Derive :class:`MetricDefaults` from a metric name by matching suffixes.

    Walks :data:`_METRIC_SUFFIXES` in order, so the first matching suffix wins.

    Args:
        metric_name: The metric name to match against the suffix table. It
            follows the metric name grammar, so a path precedes the suffix.

    Returns:
        Defaults for the first matching suffix, or direction-only defaults when
        no suffix matches.
    """
    for suffix, unit, kind in _METRIC_SUFFIXES:
        if metric_name.endswith(suffix):
            return MetricDefaults(
                direction="lower",
                unit=unit,
                kind=kind,
                short_name=metric_name.removesuffix(suffix),
            )
    return MetricDefaults(direction="lower")


# ---------------------------------------------------------------------------
# metric-lines adapter
# ---------------------------------------------------------------------------


_PREFIX = "METRIC"
_PREFIX_WITH_SPACE = "METRIC "

_LINE_SPLIT = re.compile(r"\r\n|[\n\r]")
"""Line boundary: CRLF, or a lone LF or CR.

Deliberately narrower than :meth:`str.splitlines`, which also breaks on U+000B,
U+000C, U+001C to U+001E, U+0085, U+2028, and U+2029. Those characters must stay
inside their line so that :meth:`parse` can reject a name holding one with
``LINE_TERMINATORS`` and report the whole line, instead of splitting one METRIC
line into fragments.
"""

_RADIX_NUMBER = re.compile(r"0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+")
"""Unsigned hex/octal/binary literal, matching what JS ``Number()`` accepts.

JS rejects a sign on these forms (``Number("-0x10")`` is NaN), so the pattern
carries no sign and the parser routes only sign-free tokens here.
"""


def _js_number(raw: str) -> float | None:
    """Parse ``raw`` with JavaScript ``Number()`` semantics.

    The result matches what ``Number(raw)`` would yield in JS. Where that differs
    from Python's ``float`` matters:

    - An empty or whitespace-only token is ``None`` here (JS ``Number("")`` is
      ``0``), so an unset shell variable is not read as a genuine zero.
    - ``0x``/``0o``/``0b`` literals are accepted the way JS accepts them.
    - Underscore separators, and the words ``inf``/``infinity``/``nan`` that
      Python's ``float`` accepts, are rejected because JS ``Number`` rejects them
      (or yields a non-finite value, which is dropped).

    Args:
        raw: The value token after a METRIC line's last ``=``.

    Returns:
        The finite float, or ``None`` when the token is empty, non-numeric, or
        non-finite.
    """
    token = raw.strip()
    # float("1_0") succeeds and is finite, so the finite check below cannot
    # catch underscore separators — reject them explicitly, alongside the
    # empty-token case, before either try block runs.
    if token == "" or "_" in token:
        return None
    if _RADIX_NUMBER.fullmatch(token):
        try:
            return float(int(token, 0))
        except OverflowError:
            return None
    try:
        value = float(token)
    except ValueError:
        return None
    return finite_or_none(value)


def _follows_grammar(metric_name: str, warn: WarnSink) -> bool:
    """Tell whether a METRIC line's name follows the metric name grammar, warning when not.

    Args:
        metric_name: The name read from a METRIC line; it holds no line terminator.
        warn: The sink that receives the warning about a name to skip.

    Returns:
        ``False`` when the line is to be skipped.

    Raises:
        AdapterError: When the name carries more than one ``#``.
    """
    try:
        parse_metric_name(metric_name)
    except MultipleHashesError as exc:
        msg = (
            f'Metric name "{metric_name}" contains {exc.flaw}; '
            "only a single '#' is allowed as the metric-type separator"
        )
        raise AdapterError(msg) from exc
    except MetricNameError as exc:
        warn(f'Skipping METRIC line with {exc.flaw} in its metric name: "{metric_name}"')
        return False
    return True


class _MetricLinesAdapter:
    """Adapter that reads ``METRIC name=value`` lines from a bench script's stdout."""

    name = "metric-lines"
    defaults = staticmethod(defaults_from_suffixes)

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse ``METRIC`` lines from ``stdout`` into a median-per-name metric map.

        Splits ``stdout`` into lines and reads each ``METRIC <name>=<value>`` line.
        A line whose name has an empty path segment or an empty kind is skipped
        with a warning, like a line that cannot be read at all.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send a complaint about an unreadable line; defaults to
                stderr.

        Returns:
            One value per metric name: the median of that name's samples.

        Raises:
            AdapterError: When no line yields a usable metric, or a metric name
                carries more than one ``#``.
        """
        samples: dict[str, list[float]] = {}

        for line in _LINE_SPLIT.split(stdout):
            trimmed = line.strip()
            if not trimmed.startswith(_PREFIX):
                continue
            parse_failure = f"Failed to parse METRIC line: {trimmed}"
            if not trimmed.startswith(_PREFIX_WITH_SPACE):
                warn(parse_failure)
                continue

            after = trimmed[len(_PREFIX) :].strip()
            last_eq = after.rfind("=")
            if last_eq <= 0:
                warn(parse_failure)
                continue

            metric_name = after[:last_eq]
            # CR and LF never reach a name (the input is split on them above), so this
            # rejects the other boundaries str.splitlines breaks on.
            if LINE_TERMINATORS.search(metric_name):
                warn(parse_failure)
                continue

            if not _follows_grammar(metric_name, warn):
                continue

            value = _js_number(after[last_eq + 1 :])
            if value is None:
                warn(parse_failure)
                continue

            if _PREFIX_WITH_SPACE in metric_name:
                warn(
                    f'Parsed metric name "{metric_name}" embeds the METRIC token '
                    "— the line may carry a duplicate METRIC prefix"
                )

            samples.setdefault(metric_name, []).append(value)

        if not samples:
            msg = "No valid METRIC lines found"
            raise AdapterError(msg)

        return {metric_name: statistics.median(values) for metric_name, values in samples.items()}


metric_lines_adapter = _MetricLinesAdapter()
"""The singleton ``metric-lines`` adapter instance callers register and invoke."""


# ---------------------------------------------------------------------------
# mitata adapter
# ---------------------------------------------------------------------------


_JSON_DECODER = json.JSONDecoder()


def _scan_json_objects(text: str) -> tuple[list[dict[str, object]], str | None]:
    """Scan ``text`` for JSON objects in a single pass.

    Each ``{`` in ``text`` is tried via ``raw_decode``, so unbalanced braces and
    banner text like ``cpu: {model}`` are rejected by the JSON decoder itself.
    Successes become candidates; the first failure is captured so
    ``_extract_benchmarks`` can surface an actionable diagnostic without a second
    scan.

    Args:
        text: The bench output to scan.

    Returns:
        A ``(candidates, first_failure)`` pair: the objects that parsed, and the
        error message from the first failed attempt (or ``None``).
    """
    candidates: list[dict[str, object]] = []
    first_failure: str | None = None
    pos = text.find("{")
    while pos != -1:
        try:
            # Every attempt starts at a ``{``, so a successful raw_decode can only
            # have produced a JSON object — no non-dict shape check is needed.
            parsed, end = _JSON_DECODER.raw_decode(text, pos)
        except (json.JSONDecodeError, RecursionError) as exc:
            if first_failure is None:
                first_failure = (
                    str(exc)
                    if isinstance(exc, json.JSONDecodeError)
                    else f"Exceeded maximum recursion depth while parsing at position {pos}"
                )
            pos = text.find("{", pos + 1)
            continue
        candidates.append(parsed)
        pos = text.find("{", end)
    return candidates, first_failure


def _extract_benchmarks(stdout: str) -> list[object]:
    """Find mitata's ``benchmarks`` array using :func:`_scan_json_objects`.

    A candidate carrying a ``benchmarks`` list wins over any earlier record that
    does not — a decoy object printed before mitata's own output must not shadow
    the real payload.

    When no candidate has a ``benchmarks`` list but a decode failure exists
    alongside a non-benchmarks record, the failure diagnostic takes priority —
    the real payload was likely truncated or malformed, and the parse error is
    more actionable than a generic "missing benchmarks array".

    When :func:`_scan_json_objects` returns no candidates, every ``{`` in
    ``stdout`` failed to start a valid JSON object. The diagnostic names the
    failure of the first attempt — the ``{`` spanning the most remaining text is
    most likely to be the real payload.

    Args:
        stdout: The bench command's captured stdout.

    Returns:
        The non-empty ``benchmarks`` list of the first JSON object carrying one.

    Raises:
        AdapterError: When no JSON object is found, the most promising candidate
            failed to parse, or the ``benchmarks`` array is missing or empty.
    """
    candidates, first_failure = _scan_json_objects(stdout)

    for candidate in candidates:
        benchmarks = candidate.get("benchmarks")
        if isinstance(benchmarks, list):
            if not benchmarks:
                msg = "benchmarks array is empty"
                raise AdapterError(msg)
            return benchmarks

    if first_failure is not None:
        msg = f"Failed to parse JSON: {first_failure}"
    elif candidates:
        msg = "JSON missing benchmarks array"
    else:
        msg = "No JSON object found in stdout"
    raise AdapterError(msg)


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
    if isinstance(value, float) and value.is_integer():
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
    detail = expected_got(phrase, value)
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
    key = describe_key((*prefix, *error["loc"]))
    if error["type"] == "missing":
        return f" with missing {key}"
    return _invalid_value_tail(key, phrase_for_error(error), error["input"])


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
    try:
        parse_metric_name(prefix)
    except MetricNameError as exc:
        warn(f"Skipping run with {exc.flaw} in its metric name: {json.dumps(prefix)}")
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
        _warn_skip(warn, subject, _first_problem(exc, ("stats", "heap")))
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
    defaults = staticmethod(defaults_from_suffixes)

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse mitata's JSON output into a metric map.

        Each run yields ``<alias>#time`` from ``stats.p50`` and, when mitata
        measured it, ``<alias>#heap`` from ``stats.heap.avg``. Runs that errored,
        reported a non-finite ``p50``, resolved to a metric name carrying a line
        terminator or an empty path segment, or carried a malformed
        ``args``/``stats`` shape are skipped rather than failing the parse — a
        single bad argument combination should not discard the rest of the run —
        and likewise for a benchmark whose ``alias``/``runs`` shape is
        malformed. Every skip warns through ``warn`` rather than vanishing
        silently, as does a collision between two runs landing on one metric
        name; on a collision the last run still wins.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send a complaint about a skipped run or benchmark;
                defaults to stderr.

        Returns:
            One value per metric name.

        Raises:
            AdapterError: When no JSON object is found, the JSON is malformed, the
                ``benchmarks`` array is missing or empty, a substituted metric
                prefix contains ``#`` (reserved as the metric-type separator), or
                no run yields a usable metric.
        """
        metrics: dict[str, float] = {}
        for benchmark in _extract_benchmarks(stdout):
            _extract_benchmark_metrics(benchmark, metrics, warn)

        if not metrics:
            msg = "No valid benchmark runs found"
            raise AdapterError(msg)

        return metrics


mitata_adapter = _MitataAdapter()
"""The singleton ``mitata`` adapter instance callers register and invoke."""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


_ADAPTERS: dict[str, Adapter] = {
    metric_lines_adapter.name: metric_lines_adapter,
    mitata_adapter.name: mitata_adapter,
}


def get_adapter(name: str) -> Adapter:
    """Return the built-in adapter registered under ``name``.

    Args:
        name: The adapter name to look up, e.g. ``"metric-lines"`` or ``"mitata"``.

    Returns:
        The singleton :class:`Adapter` for ``name``.

    Raises:
        GymratError: When ``name`` is not a built-in adapter. The message names
            the offending value and the ``hint`` lists every valid name.
    """
    try:
        return _ADAPTERS[name]
    except KeyError:
        msg = f'Unknown adapter: "{name}".'
        hint = f"valid adapters are: {', '.join(sorted(_ADAPTERS))}"
        raise GymratError(msg, hint=hint) from None
