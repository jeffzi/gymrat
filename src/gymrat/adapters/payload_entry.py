"""What the adapters for harnesses that print a JSON payload share.

``mitata`` and ``tinybench`` each print their results as a JSON object holding one
array. The payload is located by attempting ``json.JSONDecoder().raw_decode`` at
each ``{`` position in stdout, so banner text a harness prints around it —
including unbalanced braces like ``cpu: {model`` — does not prevent the real
payload from being found. :class:`JsonArrayAdapter` hands every array entry to a
per-harness extractor, which validates it with pydantic, skips a malformed entry
with a warning, and checks each metric name it builds with
:func:`accepts_metric_prefix`.

The extractors validate each entry, word the warning that skips a malformed one,
and store the metrics that survive through the helpers here, so both harnesses
phrase a skipped entry the same way.
"""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from gymrat.adapters.contract import AdapterError, defaults_from_suffixes
from gymrat.metric_name import MetricNameError
from gymrat.metric_name import parse as parse_metric_name
from gymrat.utils import (
    LINE_TERMINATORS,
    WarnSink,
    describe_key,
    drop_prefix_errors,
    expected_got,
    phrase_for_error,
    warn_to_stderr,
)

_JSON_DECODER = json.JSONDecoder()


def warn_skip(warn: WarnSink, subject: str, tail: str) -> None:
    """Send a "Skipping <subject><tail>" warning.

    Args:
        warn: The sink that receives the warning.
        subject: What is being skipped, e.g. ``'run of "my-bench"'``. Any alias or
            task name it names is JSON-escaped, so a line terminator in it cannot
            split the warning.
        tail: The reason, from :func:`~gymrat.adapters.payload_entry.first_problem`
            or written by the caller.
    """
    warn(f"warning: skipping {subject}{tail}")


def _scan_json_objects(text: str) -> tuple[list[dict[str, object]], str | None]:
    """Scan ``text`` for JSON objects in a single pass.

    Each ``{`` in ``text`` is tried via ``raw_decode``, so unbalanced braces and
    banner text like ``cpu: {model}`` are rejected by the JSON decoder itself.
    Successes become candidates; the first failure is captured so
    ``_extract_array`` can surface an actionable diagnostic without a second
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
        # ValueError, not just JSONDecodeError: an integer literal past the
        # interpreter's int-string digit limit raises a bare ValueError mid-decode.
        except ValueError as exc:
            failure = str(exc)
        except RecursionError:
            failure = f"Exceeded maximum recursion depth while parsing at position {pos}"
        else:
            candidates.append(parsed)
            pos = text.find("{", end)
            continue
        if first_failure is None:
            first_failure = failure
        pos = text.find("{", pos + 1)
    return candidates, first_failure


def _extract_array(stdout: str, key: str, *, last: bool) -> list[object]:
    """Find a harness's result array under ``key`` using :func:`_scan_json_objects`.

    A candidate carrying a ``key`` list wins over any record that does not, so a
    decoy object printed around the harness's own output cannot shadow the real
    payload. Mitata prints its payload first, while a tinybench script prints it
    after every task function has run, so an object a task logs comes before it.

    When no candidate has a ``key`` list but a decode failure exists alongside a
    record that lacks it, the failure diagnostic takes priority — the real
    payload was likely truncated or malformed, and the parse error is more
    actionable than a generic "missing array".

    When :func:`_scan_json_objects` returns no candidates, every ``{`` in
    ``stdout`` failed to start a valid JSON object. The diagnostic names the
    failure of the first attempt — the ``{`` spanning the most remaining text is
    most likely to be the real payload.

    Args:
        stdout: The bench command's captured stdout.
        key: The top-level key holding the result array, named in the messages.
        last: Whether the last carrier of a ``key`` list wins rather than the first.

    Returns:
        The non-empty ``key`` list of the winning JSON object.

    Raises:
        AdapterError: When no JSON object is found, the most promising candidate
            failed to parse, or the winning ``key`` array is missing or empty.
    """
    candidates, first_failure = _scan_json_objects(stdout)

    arrays = [value for candidate in candidates if isinstance(value := candidate.get(key), list)]
    if arrays:
        array = arrays[-1 if last else 0]
        if not array:
            msg = f"{key} array is empty"
            raise AdapterError(msg)
        return array

    if first_failure is not None:
        msg = f"Failed to parse JSON: {first_failure}"
    elif candidates:
        msg = f"JSON missing {key} array"
    else:
        msg = "No JSON object found in stdout"
    raise AdapterError(msg)


@dataclass(frozen=True, slots=True)
class JsonArrayAdapter:
    """Adapter for a harness that prints its results as a JSON array under one key.

    Each instance documents what its harness's entries yield and when one is skipped.

    Attributes:
        name: The adapter's identifier, as named by the ``adapter`` config key.
        key: The top-level key holding the result array.
        last: Whether the last carrier of a ``key`` list wins rather than the first.
        extract: Adds one array entry's metrics to the map, warning on a skip.
        empty_msg: The error message when no entry yields a metric.
    """

    name: str
    key: str
    last: bool
    extract: Callable[[object, dict[str, float], WarnSink], None]
    empty_msg: str
    defaults = staticmethod(defaults_from_suffixes)

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse the ``key`` array in ``stdout`` into a metric map.

        Every array entry goes through ``extract``, which skips a malformed entry
        with a warning rather than failing the parse.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send a complaint about a skipped entry; defaults to
                stderr.

        Returns:
            One value per metric name.

        Raises:
            AdapterError: When no JSON object is found, the JSON is malformed, the
                ``key`` array is missing or empty, a metric name contains more
                than one ``#``, or no entry yields a usable metric.
        """
        metrics: dict[str, float] = {}
        for entry in _extract_array(stdout, self.key, last=self.last):
            self.extract(entry, metrics, warn)

        if not metrics:
            raise AdapterError(self.empty_msg)

        return metrics


@dataclass(frozen=True, slots=True)
class NameWording:
    """How one adapter words the problems :func:`accepts_metric_prefix` reports.

    Each message names what the user wrote in their harness: a mitata alias, a
    tinybench task name.

    Attributes:
        skipped: What a skip warning names as skipped, e.g. ``"run"``.
        label: How the reserved-``#`` error introduces the prefix, e.g.
            ``"Metric prefix"``.
        origin: The JSON-escaped source text the line-terminator warning names.
            Escaped with ``ensure_ascii`` on, so every ``LINE_TERMINATORS`` entry
            is escaped and the warning stays on one line.
        hash_note: Text closing the reserved-``#`` error.
        terminator_note: Text closing the line-terminator warning.
    """

    skipped: str
    label: str
    origin: str
    hash_note: str = ""
    terminator_note: str = ""


def accepts_metric_prefix(prefix: str, warn: WarnSink, wording: NameWording) -> bool:
    """Check a metric-name prefix built from harness output, warning when it is skipped.

    Args:
        prefix: The metric name before its ``#`` kind.
        warn: The sink that receives the warning about a prefix to skip.
        wording: How the calling adapter words each problem.

    Returns:
        ``True`` when ``prefix`` is usable, ``False`` when the entry is to be
        skipped.

    Raises:
        AdapterError: When ``prefix`` contains ``#``, which the adapter appends
            itself as the metric-type separator.
    """
    if "#" in prefix:
        msg = (
            f"{wording.label} {json.dumps(prefix)} contains '#', which is reserved as the "
            f"metric-type separator{wording.hash_note}"
        )
        raise AdapterError(msg)
    if LINE_TERMINATORS.search(prefix):
        # Check every entry of LINE_TERMINATORS, not just ``\n`` and ``\r``: a
        # harness's JSON can carry any of them inside a name.
        warn_skip(
            warn,
            wording.skipped,
            " with a line terminator in its metric name: "
            f"{wording.origin}{wording.terminator_note}",
        )
        return False
    try:
        parse_metric_name(prefix)
    except MetricNameError as exc:
        warn_skip(
            warn,
            wording.skipped,
            f" with {exc.flaw} in its metric name: {json.dumps(prefix)}",
        )
        return False
    return True


def record_metric(
    metrics: dict[str, float], name: str, value: float, warn: WarnSink, advice: str
) -> None:
    """Store ``value`` under ``name``, warning when it displaces an earlier reading.

    A collision means two readings resolved to one metric name, so the report
    would otherwise silently show only the last one's value.

    Args:
        metrics: The readings collected so far, updated in place.
        name: The resolved metric name.
        value: The reading to store.
        warn: The sink that receives the collision warning.
        advice: How the user separates the colliding readings in their harness,
            closing the warning.
    """
    if name in metrics:
        warn(f"warning: duplicate metric name: {name} (keeping the last value; {advice})")
    metrics[name] = value


def _escape_line_terminator(match: re.Match[str]) -> str:
    return json.dumps(match.group())[1:-1]


_EMPTY_ERRORS: Final[tuple[object, ...]] = (None, "", {})
"""Error values that carry no detail.

``JSON.stringify`` writes an ``Error`` as ``{}``, so an errored entry reads as
``{}`` unless the bench script put something readable in its place.
"""


def error_tail(error: object) -> str:
    r"""Phrase a harness's error value as the tail of an errored entry's skip warning.

    The value (mitata's ``run.error``, tinybench's ``result.error``) is read from
    parsed JSON, so it can be any JSON value, not just a string: a non-string is
    rendered with ``json.dumps``, which cannot fail since the value round-trips
    from ``json.loads``, instead of a Python repr. A string is written as-is so a
    plain message reads naturally, except that each character of
    ``LINE_TERMINATORS`` becomes its JSON escape (``\n``, ``\u000b``, ``\f``,
    ``\r``, ``\u001c`` to ``\u001e``, ``\u0085``, ``\u2028``, ``\u2029``): the
    warning must stay on one line. ``json.dumps`` keeps ``ensure_ascii`` on, so a
    non-string value is already escaped the same way.

    Args:
        error: The error value as parsed from JSON.

    Returns:
        The rendered error in parentheses, after a space, or an empty string when
        the value carries no detail.
    """
    if error in _EMPTY_ERRORS:
        return ""
    if isinstance(error, str):
        return f" ({LINE_TERMINATORS.sub(_escape_line_terminator, error)})"
    return f" ({json.dumps(error)})"


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


def first_problem(exc: ValidationError, prefix: tuple[str, ...] = ()) -> str:
    """Phrase the first problem pydantic found as the tail of a skip warning.

    Args:
        exc: The validation failure.
        prefix: The path of the validated value inside the entry (a mitata run,
            a tinybench task), prepended to each error location.

    Returns:
        The text that follows the skipped entry's name in the warning.
    """
    error = drop_prefix_errors(exc.errors())[0]
    key = describe_key((*prefix, *error["loc"]))
    if error["type"] == "missing":
        return f" with missing {key}"
    return _invalid_value_tail(key, phrase_for_error(error), error["input"])


def is_finite_or_warn(value: float, warn: WarnSink, subject: str, key: str) -> bool:
    """Tell whether ``value`` is finite, sending a skip warning for ``subject`` when not.

    Args:
        value: The reading to check.
        warn: The sink that receives the warning.
        subject: What is being skipped, as for :func:`warn_skip`.
        key: The dotted path of ``value``, named in the warning.

    Returns:
        ``False`` when ``value`` is not finite, or is an integer too large to
        convert to a float, and the entry is to be skipped.
    """
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if finite:
        return True
    warn_skip(warn, subject, _invalid_value_tail(key, "a finite number", value))
    return False


def named_subject(kind: str, entry: object, key: str) -> str:
    """Name a skipped entry by its ``key`` field when that field is a string.

    Args:
        kind: What the entry is, e.g. ``"benchmark"``.
        entry: The raw entry as parsed from JSON.
        key: The field naming the entry, e.g. ``"alias"``.

    Returns:
        ``kind`` followed by the JSON-escaped name, or ``kind`` alone.
    """
    name = entry.get(key) if isinstance(entry, dict) else None
    return f"{kind} {json.dumps(name)}" if isinstance(name, str) else kind
