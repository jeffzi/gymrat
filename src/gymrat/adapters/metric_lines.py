"""The ``metric-lines`` adapter, which reads ``METRIC name=value`` lines.

A bench script prints one such line per sample; the adapter warns about every
line it cannot read and reduces repeated names to their median, so a run with
several samples yields one value per name.
"""

import re

from gymrat.adapters.contract import AdapterError, defaults_from_suffixes
from gymrat.metric_name import MetricNameError, MultipleHashesError
from gymrat.metric_name import parse as parse_metric_name
from gymrat.utils import (
    LINE_TERMINATORS,
    WarnSink,
    finite_or_none,
    medians_by_name,
    warn_to_stderr,
)

_PREFIX = "METRIC"
_PREFIX_WITH_SPACE = f"{_PREFIX} "

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
    - Underscore separators, non-ASCII digits (Arabic-Indic, full-width, and
      other Unicode decimal digits), and the words ``inf``/``infinity``/``nan``
      that Python's ``float`` accepts, are rejected because JS ``Number`` rejects
      them (or yields a non-finite value, which is dropped).

    Args:
        raw: The value token after a METRIC line's last ``=``.

    Returns:
        The finite float, or ``None`` when the token is empty, non-numeric, or
        non-finite.
    """
    token = raw.strip()
    # float("1_0") succeeds and is finite, as does float() on non-ASCII Unicode
    # digits, so the finite check below cannot catch underscore separators or
    # non-ASCII digits — reject them explicitly, alongside the empty-token case,
    # before either try block runs.
    if token == "" or "_" in token or not token.isascii():
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
        readings: list[dict[str, float]] = []

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

            readings.append({metric_name: value})

        if not readings:
            msg = "No valid METRIC lines found"
            raise AdapterError(msg)

        return medians_by_name(readings)


metric_lines_adapter = _MetricLinesAdapter()
"""Repeated names reduce to their median.

An unreadable line is skipped with a warning. Parsing raises
:class:`~gymrat.adapters.contract.AdapterError` when no valid ``METRIC`` line is
found or a name has more than one ``#``.
"""
