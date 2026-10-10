"""Generic helpers that depend on the standard library alone.

Nothing here imports a project module or names a project type: each helper
would make sense pasted unchanged into an unrelated repository.
"""

import contextlib
import errno
import io
import json
import math
import os
import re
import secrets
import stat
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, NamedTuple, Protocol, Self

#: Shared by every module that converts between nanoseconds, milliseconds and
#: clock tiers, so the conversion factors are declared once.
SECONDS_PER_MINUTE = 60
MS_PER_SECOND = 1000
NS_PER_MS = 1_000_000
_MS_PER_MINUTE = SECONDS_PER_MINUTE * MS_PER_SECOND

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
"""Environment variable every tracing entry point reads the OTLP endpoint from."""

# The budget is measured in bytes, not characters: downstream consumers size
# their buffers in bytes, so a multi-byte-heavy string that "looks short" can
# still blow past the limit. Keep this module-private: the byte budget is an
# internal knob, not something a caller should tune.
_OUTPUT_LIMIT_BYTES = 8192

# ``[0-9]`` rather than ``\d``: ``\d`` matches every Unicode decimal digit, and
# ``float`` converts those too, so a non-ASCII digit would parse as a number.
ASCII_DECIMAL_PATTERN = r"[0-9]+(?:\.[0-9]+)?"
"""An unsigned decimal written in ASCII digits, with an optional fractional part."""

SHORT_SHA_LENGTH = 7
"""How many leading characters of a commit SHA an abbreviation keeps."""

LINE_TERMINATORS = re.compile("[\\n\\v\\f\\r\\x1c-\\x1e\\x85\\u2028\\u2029]")
"""Every line boundary :meth:`str.splitlines` breaks on.

That is LF, VT (U+000B), FF (U+000C), CR, U+001C to U+001E, U+0085, U+2028, and
U+2029. Text holding one splits across lines in any message or record line that
carries it, and LF, CR, U+2028, and U+2029 are line terminators to a JavaScript
regular-expression engine, so an anchored check on a record field could never
match such a value. Written as escapes so the source stays plain ASCII.
"""

MISSING_DELTA = "—"
"""What a progress display prints in place of a missing or non-finite delta."""

type WarnSink = Callable[[str], None]
"""Where a caller sends a complaint about output it could not read.

The caller owns the destination so a warning can be interleaved with whatever
else is on the terminal — the CLI's progress line, for one — instead of landing
on stderr wherever the cursor happens to be.
"""


class StyledSegment[R](NamedTuple):
    """One run of text together with the style role it renders under.

    Text built away from the rich view layer carries a role rather than a theme
    style; the view maps each role to a style, and a plain renderer joins the
    texts.

    Attributes:
        text: The run's text.
        role: The style role the text renders under.
    """

    text: str
    role: R


def abbreviate_home(path: str) -> str:
    """Shorten a path under the user's home directory to a ``~`` prefix.

    Args:
        path: The path to abbreviate.

    Returns:
        The ``~``-prefixed path, or *path* unchanged when it is not under home.
    """
    try:
        rel = Path(path).relative_to(Path.home()).as_posix()
    except (ValueError, RuntimeError):
        return path
    return "~" if rel == "." else f"~/{rel}"


def pluralize(count: int, noun: str) -> str:
    """Inflect ``noun`` for ``count`` by appending ``s``.

    A count of one keeps ``noun`` as given; any other count — zero and
    negatives included — takes the plural form. No other suffix rule is
    applied, so ``noun`` must be one whose plural is a plain ``s``.

    Args:
        count: The count that determines singular vs. plural form.
        noun: The singular form of the noun.

    Returns:
        The count followed by the inflected noun.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def format_cost(usd: float) -> str:
    """Format a USD amount as a two-decimal dollar string."""
    return f"${usd:.2f}"


def first_line(text: str) -> str:
    """The first line of ``text``, discarding the rest."""
    return text.split("\n", maxsplit=1)[0]


UNICODE_LINE_BREAKS = str.maketrans({
    "\N{NEXT LINE}": "\\u0085",
    "\N{LINE SEPARATOR}": "\\u2028",
    "\N{PARAGRAPH SEPARATOR}": "\\u2029",
})
"""Translation table for the non-ASCII characters ``str.splitlines`` breaks on.

Inside a JSON document they occur only within strings, so translating a
serialized line through this table is lossless and keeps one document on one
line for any reader that splits with ``str.splitlines``.
"""


def expected_got(phrase: str, value: object) -> str:
    """Word a rejected value against the shape that was expected of it.

    The value is shown as JSON, or through ``repr`` when JSON cannot encode it.
    An integer with more decimal digits than the interpreter converts to text
    is described by its bit length, and a container holding one by its type, so
    the wording never raises and never grows with the integer.

    Args:
        phrase: What was expected, such as ``"a number"``.
        value: The rejected value.

    Returns:
        The text ``expected {phrase}, got {value}``.
    """
    return f"expected {phrase}, got {_displayed(value)}"


def _displayed(value: object) -> str:
    with contextlib.suppress(TypeError, ValueError):
        return json.dumps(value)
    with contextlib.suppress(ValueError):
        return repr(value)
    if isinstance(value, int):
        return f"a {value.bit_length()}-bit integer"
    return f"a {type(value).__name__} too large to display"


def finite_or_none(value: float) -> float | None:
    """Swap a float that is not a finite number for ``None``.

    JSON serialization writes any non-finite float as ``null`` whatever the
    writer intended. Making the substitution in memory keeps the value a caller
    holds identical to the one read back from a serialized copy, and never lets
    a number stand where there was no measurement.

    Args:
        value: The float to check, possibly ``NaN`` or infinite.

    Returns:
        ``value`` itself, or ``None`` when it is ``NaN`` or either infinity.
    """
    return value if math.isfinite(value) else None


def fraction_of_median(numerator: float, median: float, scale: float = 1.0) -> float | None:
    """A value as a scaled fraction of a median's magnitude.

    The scale is applied after the division so a large numerator alone cannot
    overflow.

    Args:
        numerator: The value to express as a fraction of the median.
        median: The median whose magnitude is the denominator.
        scale: The factor applied to the fraction.

    Returns:
        The scaled fraction, or ``None`` when ``median`` is zero or the scaled
        fraction is not finite.
    """
    if median == 0:
        return None
    fraction = (numerator / abs(median)) * scale
    return finite_or_none(fraction)


def medians_by_name(rounds: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """The median each name came to over the rounds that reported it.

    A round that omits a name contributes nothing to that name's median rather
    than a zero, and a name no round reported has no entry at all.

    Args:
        rounds: One mapping of name to value per round.

    Returns:
        Each reported name mapped to its median, in the order the rounds first
        named them.
    """
    readings: dict[str, list[float]] = {}
    for round_ in rounds:
        for name, value in round_.items():
            readings.setdefault(name, []).append(value)
    return {name: statistics.median(values) for name, values in readings.items()}


def own_values(samples: Sequence[dict[str, float]], name: str) -> list[float]:
    """Collect the values a side reported for ``name``, skipping rounds without it.

    Args:
        samples: One metric record per round.
        name: The metric to extract.

    Returns:
        The reported values for ``name``, in round order.
    """
    return [record[name] for record in samples if name in record]


def coerce_integer(value: object) -> object:
    """Fold an integral float into ``int`` so it satisfies strict integer validation.

    Only the fold happens here; accepting or rejecting the value stays the
    model's job.

    Args:
        value: The value to coerce.

    Returns:
        The coerced ``int`` when ``value`` is an integral float, otherwise
        ``value`` unchanged.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def limit_output(text: str) -> str:
    """Return at most ``_OUTPUT_LIMIT_BYTES`` bytes of ``text`` (UTF-8).

    Decoding the cut with ``errors="ignore"`` drops the trailing bytes of a
    character the cut split, so a multi-byte character is never severed and no
    U+FFFD replacement character is emitted.

    Args:
        text: The text to cap.

    Returns:
        The original text when it fits the budget. Otherwise the prefix up to
        the last newline inside the first ``_OUTPUT_LIMIT_BYTES`` bytes, with
        that newline dropped; when no usable newline exists (a single long
        line, or the only newline at byte 0), the prefix up to the last whole
        character.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= _OUTPUT_LIMIT_BYTES:
        return text

    head = encoded[:_OUTPUT_LIMIT_BYTES]
    last_newline = head.rfind(b"\n")
    # Require the newline past byte 0 so a leading-newline single line still
    # relays its content instead of collapsing to an empty string.
    if last_newline > 0:
        return head[:last_newline].decode("utf-8")

    return head.decode("utf-8", errors="ignore")


def stderr_text_of(error: object) -> str:
    """The diagnostics a failed child process wrote to its output streams.

    ``subprocess.CalledProcessError`` and ``TimeoutExpired`` attach the child's
    captured output separately from the exception message, which carries
    ``Command '...' returned non-zero exit status`` noise. Preferring the raw
    stderr keeps the child's own diagnostics — the text a caller classifies
    the failure on — instead of that wrapper noise.

    Falls back to the captured stdout when stderr is blank: some tools explain
    a failure on stdout instead (git's "nothing to commit", a commit hook's
    rejection message), leaving stderr an empty or whitespace-only string. Only
    when neither stream carries text does the argv-shaped ``str(error)`` stand
    in.

    Args:
        error: The caught value.

    Returns:
        The trimmed stderr when it is present and non-blank, else the trimmed
        stdout on the same terms, else the message.
    """
    for stream_name in ("stderr", "stdout"):
        stream = getattr(error, stream_name, None)
        if isinstance(stream, bytes):
            # Captured streams are bytes unless the call ran in text mode; replace
            # undecodable sequences so diagnostics never raise here.
            stream = stream.decode("utf-8", errors="replace")
        if isinstance(stream, str) and stream.strip():
            return stream.strip()
    return str(error)


def otlp_endpoint(value: str | None) -> str | None:
    """Apply the one trimming rule every reader of the OTLP endpoint shares.

    Args:
        value: A raw endpoint, from a flag or ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        The endpoint with surrounding whitespace trimmed, or ``None`` when
        nothing is left, which means "no endpoint".
    """
    return (value or "").strip() or None


def otlp_endpoint_from_env(name: str = ENDPOINT_ENV) -> str | None:
    """Read an OTLP endpoint variable from the environment, trimmed by :func:`otlp_endpoint`.

    Args:
        name: The environment variable that holds the endpoint.

    Returns:
        The trimmed endpoint, or ``None`` when the variable is unset or blank.
    """
    return otlp_endpoint(os.environ.get(name))


def color_from_env() -> bool | None:
    """The color preference the environment declares, or ``None`` to defer to the caller.

    One precedence rule, shared by every color surface so they never disagree:
    ``FORCE_COLOR`` (any value but ``0``/``false``/empty) forces color on even
    when ``NO_COLOR`` is also present; ``FORCE_COLOR`` set to a rejected value
    (``0``/``false``/empty) explicitly disables color — this must override
    Rich's presence-based detection which treats any ``FORCE_COLOR`` as "on";
    ``NO_COLOR`` (present, any value) then forces it off; with neither the
    answer is ``None`` so the caller decides from the stream's own TTY state.

    Returns:
        ``True`` for forced color, ``False`` for suppressed color, or ``None``
        when neither environment variable is set.
    """
    force_color = os.environ.get("FORCE_COLOR")
    if force_color is not None:
        return force_color.lower() not in {"", "0", "false"}
    if "NO_COLOR" in os.environ:
        return False
    return None


def is_tty(stream: object) -> bool:
    """Whether ``stream`` reports itself as an interactive terminal."""
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def stream_color_from_env(stream: object) -> bool:
    """Whether ``stream`` carries color when no flag decides it.

    Args:
        stream: The output stream whose TTY status is the fallback.

    Returns:
        What :func:`color_from_env` declares, or the stream's own TTY status
        when the environment declares nothing.
    """
    declared = color_from_env()
    return declared if declared is not None else is_tty(stream)


class _WritableStream(Protocol):
    """A text stream error and progress output is written to."""

    def write(self, data: str, /) -> object: ...

    def flush(self) -> object: ...


def write_and_flush(stream: _WritableStream, data: str) -> None:
    """Write ``data`` to ``stream`` and flush it so an immediate exit cannot truncate it."""
    stream.write(data)
    stream.flush()


def is_broken_pipe(error: BaseException) -> bool:
    """Whether ``error`` is a write to a pipe whose reading end has closed.

    POSIX reports it as ``BrokenPipeError``. Windows reports it as a plain
    ``OSError`` with ``EINVAL``: the C runtime maps the ``ERROR_NO_DATA`` a write
    to a closed pipe fails with onto that errno, so no ``BrokenPipeError`` is
    ever raised there.

    Args:
        error: The exception a stream write or flush raised.

    Returns:
        ``True`` when ``error`` means the pipe's reader is gone.
    """
    if isinstance(error, BrokenPipeError):
        return True
    return sys.platform == "win32" and isinstance(error, OSError) and error.errno == errno.EINVAL


def point_stream_at_devnull(stream: IO[str]) -> None:
    """Redirect ``stream``'s file descriptor to devnull; a stream without one is left alone.

    The interpreter flushes a stream's unwritten buffer at shutdown. After a
    failed write that flush would fail again and turn the exit status into 120;
    a devnull descriptor lets it succeed.

    Args:
        stream: The stream whose descriptor is redirected.
    """
    try:
        fd = stream.fileno()
    except io.UnsupportedOperation:
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
    finally:
        os.close(devnull)


def warn_to_stderr(message: str) -> None:
    """Default :data:`WarnSink` for callers called without an explicit one.

    Args:
        message: The text to write to stderr, followed by a newline.
    """
    sys.stderr.write(f"{message}\n")


def fan_out[E](
    subscribers: Iterable[Callable[[E], None]],
    on_error: Callable[[Exception], None],
) -> Callable[[E], None]:
    """Build a callback that dispatches each event to every subscriber in order.

    Every subscriber receives the identical event object. A subscriber that
    raises never silences the others: its exception goes to ``on_error`` and
    dispatch continues with the next subscriber. With no subscribers the
    callback is a no-op.

    Args:
        subscribers: The callbacks to dispatch each event to, in call order.
            They are snapshotted when ``fan_out`` is called.
        on_error: The sink that receives each exception a subscriber raises.

    Returns:
        A callback that dispatches each event to all subscribers.
    """
    subs = tuple(subscribers)

    def dispatch(event: E) -> None:
        for subscriber in subs:
            try:
                subscriber(event)
            except Exception as error:  # noqa: BLE001 - reported through on_error; one failure must not break the chain
                on_error(error)

    return dispatch


def pair_value[V](pairs: tuple[tuple[str, V], ...], key: str) -> V | None:
    """Return the value paired with `key`, or None if `key` is not present."""
    return next((value for name, value in pairs if name == key), None)


def write_text_atomic(path: Path, text: str) -> None:
    """Replace *path* with *text*, encoded as UTF-8, in one atomic step.

    The text goes to a temporary sibling that is flushed and fsynced before it
    is renamed over *path*, so a reader sees either the previous content or the
    full new content. On failure *path* is left untouched and the temporary
    file is removed.

    The file ends up with the mode a plain write would leave: an existing
    *path* keeps its mode, and a new one gets ``0o666`` filtered by the umask.
    The temporary file is never more permissive than an existing *path*, so
    the new content is not readable by anyone the old content was closed to.

    Args:
        path: The file to write. Its directory must already exist.
        text: The content to write.

    Raises:
        OSError: When the temporary file cannot be created, written, synced,
            or renamed over *path*.
    """
    try:
        target_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        target_mode = None
    create_mode = 0o666 if target_mode is None else target_mode
    tmp_path = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")

    # Opened directly rather than through tempfile, which forces 0o600: here the
    # kernel applies the umask to the requested mode, and reading the umask any
    # other way means changing it for the whole process. Requesting the target's
    # own mode at creation is what keeps the content out of a wider file; a
    # chmod after the write would leave it exposed until then.
    def create_with_mode(name: str, flags: int) -> int:
        return os.open(name, flags, create_mode)

    # Exclusive creation stays outside the cleanup below: a file that already
    # holds the temporary name is not this call's to remove.
    with open(tmp_path, "xb", opener=create_with_mode):
        pass
    try:
        with tmp_path.open("wb") as tmp_file:
            tmp_file.write(text.encode("utf-8"))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        if target_mode is not None:
            # Restores the bits the umask removed from the requested mode.
            tmp_path.chmod(target_mode)
        tmp_path.replace(path)
    except BaseException:
        # The original error is what the caller needs; a failed cleanup must not mask it.
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Durations and ETAs
# ---------------------------------------------------------------------------

# The formatters are deliberately not unified. format_duration reports elapsed
# time and floors to whole seconds while always keeping a zero remainder
# ("1m 0s"); format_clock renders the media-player clock a progress bar ticks
# through ("07:45", "1:07:45"). Both build on the tier splits below, which the
# CLI's timestamp and ETA formatters share.


@dataclass(frozen=True, slots=True)
class SamplingEta:
    """Remaining-time estimate built from the average of finished passes.

    A pure value: it accumulates finished-pass durations and divides them over
    the work that is left.

    Attributes:
        total: Passes the run expects in all.
        completed: Passes finished so far; every one of them contributed a
            sample to ``total_time_ms``.
        total_time_ms: Sum of the sampled durations, in milliseconds.
    """

    total: int
    completed: int = 0
    total_time_ms: float = 0.0

    @property
    def eta_ms(self) -> float | None:
        """Milliseconds of work left, or None when no estimate can be made.

        Returns:
            The average finished-pass duration multiplied by the passes that
            remain, or None while no pass has finished and once nothing remains.
        """
        remaining = self.total - self.completed
        if remaining <= 0 or self.completed == 0:
            return None
        return (self.total_time_ms / self.completed) * remaining

    def advanced(self, duration_ms: float) -> Self:
        """Return a copy that counts one more finished pass.

        Args:
            duration_ms: Duration of the pass that just finished, in milliseconds.

        Returns:
            A new estimate; this one is left unchanged.
        """
        return replace(
            self,
            completed=self.completed + 1,
            total_time_ms=self.total_time_ms + duration_ms,
        )


def hours_minutes_seconds(total_seconds: int) -> tuple[int, int, int]:
    """Split a whole-second count into an hours/minutes/seconds tier."""
    minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    hours, minutes = divmod(minutes, SECONDS_PER_MINUTE)
    return hours, minutes, seconds


def floored_clock_tiers(ms: float) -> tuple[int, int, int]:
    """Clamp a negative elapsed input to zero, floor to whole seconds, and split into tiers."""
    return hours_minutes_seconds(math.floor(max(0.0, ms) / MS_PER_SECOND))


def minutes_to_ms(minutes: float) -> int:
    """Convert minutes to milliseconds, truncating toward zero via ``int()``."""
    return int(minutes * _MS_PER_MINUTE)


def ms_to_minutes(ms: float) -> float:
    """Convert milliseconds to minutes, keeping fractional precision."""
    return ms / _MS_PER_MINUTE


def format_duration(ms: float) -> str:
    """Format an elapsed duration, flooring to whole seconds.

    Uses at most two tiers and always shows a zero remainder in the lower tier
    (``60_000`` renders ``"1m 0s"``, ``3_600_000`` renders ``"1h 00m"``).  The
    hour tier zero-pads the minute remainder to two digits.

    Args:
        ms: The elapsed duration in milliseconds.

    Returns:
        The formatted duration string (e.g. ``"5s"``, ``"1m 0s"``, ``"1h 00m"``).
    """
    hours, minutes, seconds = floored_clock_tiers(ms)

    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_time_left(remaining_ms: float, max_minutes: float) -> str:
    """Format how much of a session's time budget is left, as the agent reads it.

    Args:
        remaining_ms: Milliseconds left until the deadline.
        max_minutes: The budget's total, in minutes.

    Returns:
        The ``"<remaining> left of <total>m"`` line.
    """
    return f"{format_duration(remaining_ms)} left of {max_minutes:g}m"


def format_clock(ms: float) -> str:
    """Format a duration as a media-player clock, flooring to whole seconds.

    Minutes always take two digits (``"00:09"``, ``"07:45"``); the hour tier
    appears only when there are whole hours (``"1:07:45"``).

    Args:
        ms: The duration in milliseconds.

    Returns:
        The clock string, e.g. ``"07:45"`` or ``"1:07:45"``.
    """
    hours, minutes, seconds = floored_clock_tiers(ms)

    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"
