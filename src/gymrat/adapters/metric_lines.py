"""The ``metric-lines`` adapter: reads ``METRIC name=value`` lines from stdout.

A bench script prints one ``METRIC <name>=<value>`` line per sample. This adapter
collects them, warns about every line it cannot read, and reduces repeated names
to their median so a benchmark run with several samples yields one value per name.
"""

import math
import re

from gymrat.adapters.defaults import SuffixDefaultsMixin
from gymrat.adapters.types import AdapterError, WarnSink, warn_to_stderr
from gymrat.errors import GymratError
from gymrat.stats.descriptive import compute_median

_PREFIX = "METRIC"
_PREFIX_WITH_SPACE = "METRIC "

_LINE_SPLIT = re.compile(r"\r\n|[\n\r]")
"""Line boundary: CRLF, or a lone LF or CR.

Deliberately narrower than :meth:`str.splitlines`, which also breaks on U+000B,
U+000C, U+001C to U+001E, U+0085, U+2028, and U+2029 — splitting on those would
change which text forms a line and let a name carry a separator the caller's JSON
layer cannot represent.
"""

_FORBIDDEN_NAME_CHAR = re.compile("[\\u2028\\u2029]")
"""Line and paragraph separators a metric name may not contain.

CR and LF cannot reach a name — the input is already split on them — so only the
JSON-illegal separators U+2028 and U+2029 remain to reject. Written as escapes so
the source stays plain ASCII.
"""

_RADIX_NUMBER = re.compile(r"0[xX][0-9a-fA-F]+$|0[oO][0-7]+$|0[bB][01]+$")
"""Unsigned hex/octal/binary literal, matching what JS ``Number()`` accepts.

JS rejects a sign on these forms (``Number("-0x10")`` is NaN), so the pattern
carries no sign and the parser routes only sign-free tokens here.
"""


def _js_number(raw: str) -> float | None:
    """Parse ``raw`` with JavaScript ``Number()`` semantics.

    Returns the finite float ``Number(raw)`` would yield, or ``None`` when JS
    would produce ``NaN`` or a non-finite value. The distinction from Python's
    ``float`` matters:

    - An empty or whitespace-only token is ``None`` here (JS ``Number("")`` is
      ``0``), so an unset shell variable is not read as a genuine zero.
    - ``0x``/``0o``/``0b`` literals are accepted the way JS accepts them.
    - Underscore separators, and the words ``inf``/``infinity``/``nan`` that
      Python's ``float`` accepts, are rejected because JS ``Number`` rejects them
      (or yields a non-finite value that fails the ``isfinite`` guard).
    """
    token = raw.strip()
    # float("1_0") succeeds and is finite, so the isfinite guard below cannot
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
    return value if math.isfinite(value) else None


class _MetricLinesAdapter(SuffixDefaultsMixin):
    """Adapter that reads ``METRIC name=value`` lines from a bench script's stdout."""

    name = "metric-lines"

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse ``METRIC`` lines from ``stdout`` into a median-per-name metric map.

        Splits ``stdout`` into lines, reads each ``METRIC <name>=<value>`` line,
        and warns through ``warn`` about any line it cannot read. Repeated names
        collapse to their median.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send a complaint about an unreadable line; defaults to
                stderr.

        Returns:
            One median value per metric name.

        Raises:
            AdapterError: When no line yields a usable metric.
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
            if _FORBIDDEN_NAME_CHAR.search(metric_name):
                warn(parse_failure)
                continue

            if metric_name.count("#") > 1:
                msg = (
                    f"Metric name \"{metric_name}\" contains more than one '#'; "
                    "only a single '#' is allowed as the metric-type separator"
                )
                raise GymratError(msg)

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

        return {metric_name: compute_median(values) for metric_name, values in samples.items()}


metric_lines_adapter = _MetricLinesAdapter()
"""The singleton ``metric-lines`` adapter instance callers register and invoke."""
