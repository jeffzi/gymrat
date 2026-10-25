"""What the adapters for harnesses that print a JSON payload share.

``mitata`` and ``tinybench`` each print their results as a JSON object holding one
array. The payload is located by attempting ``json.JSONDecoder().raw_decode`` at
each ``{`` position in stdout, so banner text a harness prints around it —
including unbalanced braces like ``cpu: {model`` — does not prevent the real
payload from being found. :class:`JsonArrayAdapter` hands every array entry to a
per-harness extractor, which validates it with pydantic, skips a malformed entry
with a warning, and checks each metric name it builds with
:func:`accepts_metric_prefix`.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from gymrat.adapters.contract import AdapterError, defaults_from_suffixes
from gymrat.metric_name import MetricNameError
from gymrat.metric_name import parse as parse_metric_name
from gymrat.utils import LINE_TERMINATORS, WarnSink, warn_skip, warn_to_stderr

_JSON_DECODER = json.JSONDecoder()


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
        except json.JSONDecodeError as exc:
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
