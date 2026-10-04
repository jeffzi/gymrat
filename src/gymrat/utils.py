"""Generic helpers that depend on the standard library alone.

Nothing here imports a project module or names a project type: each helper
would make sense pasted unchanged into an unrelated repository.
"""

import contextlib
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

#: Shared by every module that converts between nanoseconds, milliseconds and
#: clock tiers, so the conversion factors are declared once.
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
MS_PER_SECOND = 1000
NS_PER_MS = 1_000_000

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
"""Environment variable every tracing entry point reads the OTLP endpoint from."""

# The budget is measured in bytes, not characters: downstream consumers size
# their buffers in bytes, so a multi-byte-heavy string that "looks short" can
# still blow past the limit. Keep this module-private: the byte budget is an
# internal knob, not something a caller should tune.
_OUTPUT_LIMIT_BYTES = 8192

_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")

_VOWELS = "aeiou"  # cspell:ignore aeiou

type WarnSink = Callable[[str], None]
"""Where a caller sends a complaint about output it could not read.

The caller owns the destination so a warning can be interleaved with whatever
else is on the terminal — the CLI's progress line, for one — instead of landing
on stderr wherever the cursor happens to be.
"""


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


def _regular_plural(noun: str) -> str:
    """``noun`` under the regular English suffix rules.

    A sibilant ending takes ``-es``; a consonant followed by ``y`` takes
    ``-ies``; everything else takes ``-s``. Multi-word nouns inflect on their
    last word, which the suffix tests already look at.

    Args:
        noun: The singular noun.

    Returns:
        The pluralized noun.
    """
    if noun.endswith(_SIBILANT_ENDINGS):
        return f"{noun}es"
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in _VOWELS:
        return f"{noun[:-1]}ies"
    return f"{noun}s"


def pluralize(count: int, noun: str) -> str:
    """Inflect ``noun`` for ``count`` under the regular English suffix rules.

    A count of one keeps ``noun`` as given; any other count — zero and
    negatives included — takes the plural form.

    Args:
        count: The count that determines singular vs. plural form.
        noun: The singular form of the noun.

    Returns:
        The count followed by the correctly inflected noun.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {_regular_plural(noun)}"


def first_line(text: str) -> str:
    """The first line of ``text``, discarding the rest."""
    return text.split("\n", maxsplit=1)[0]


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


def write_text_atomic(path: Path, text: str) -> None:
    """Replace *path* with *text*, encoded as UTF-8, in one atomic step.

    The text goes to a temporary sibling that is flushed and fsynced before it
    is renamed over *path*, so a reader sees either the previous content or the
    full new content. On failure *path* is left untouched and the temporary
    file is removed.

    Args:
        path: The file to write. Its directory must already exist.
        text: The content to write.

    Raises:
        OSError: When the temporary file cannot be created, written, synced,
            or renamed over *path*.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(text.encode("utf-8"))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
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
# ("1m 0s"); format_eta reports a forward estimate, rounds to whole seconds,
# clamps to at least one second, and drops a zero remainder ("~1m left");
# format_clock renders the media-player clock a progress bar ticks through
# ("07:45", "1:07:45").


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


def _hours_minutes_seconds(total_seconds: int) -> tuple[int, int, int]:
    """Split a whole-second count into an hours/minutes/seconds tier."""
    minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    hours, minutes = divmod(minutes, SECONDS_PER_MINUTE)
    return hours, minutes, seconds


def _floored_clock_tiers(ms: float) -> tuple[int, int, int]:
    """Clamp a negative elapsed input to zero, floor to whole seconds, and split into tiers."""
    return _hours_minutes_seconds(math.floor(max(0.0, ms) / MS_PER_SECOND))


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
    hours, minutes, seconds = _floored_clock_tiers(ms)

    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_timestamp(at_ms: float, run_start_ms: float | None) -> str:
    """Format an elapsed timestamp as ``[HH:MM:SS]`` since ``run_start_ms``.

    Each caller anchors its own run start, so an unanchored run reads as zero
    elapsed rather than as an error.

    Args:
        at_ms: The timestamp in milliseconds to format.
        run_start_ms: The run's start timestamp in milliseconds, or ``None``
            when the run is not yet anchored, which renders zero elapsed.

    Returns:
        A bracketed timestamp string, e.g. ``"[00:07:45]"``.
    """
    elapsed_ms = 0 if run_start_ms is None else at_ms - run_start_ms
    hours, minutes, seconds = _floored_clock_tiers(elapsed_ms)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"


def format_clock(ms: float) -> str:
    """Format a duration as a media-player clock, flooring to whole seconds.

    Minutes always take two digits (``"00:09"``, ``"07:45"``); the hour tier
    appears only when there are whole hours (``"1:07:45"``).

    Args:
        ms: The duration in milliseconds.

    Returns:
        The clock string, e.g. ``"07:45"`` or ``"1:07:45"``.
    """
    hours, minutes, seconds = _floored_clock_tiers(ms)

    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_eta(ms: float) -> str:
    """Format a forward time estimate, rounding to whole seconds.

    Clamps to at least one second and drops a zero remainder in the lower tier
    (``60_000`` renders ``"~1m left"``, not ``"~1m 0s left"``).

    Args:
        ms: The forward time estimate in milliseconds.

    Returns:
        The ETA string, e.g. ``"~5s left"`` or ``"~1m left"``.
    """
    total_seconds = max(1, round(ms / MS_PER_SECOND))
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)

    if total_seconds < SECONDS_PER_MINUTE:
        return f"~{seconds}s left"
    if total_seconds < SECONDS_PER_HOUR:
        if seconds > 0:
            return f"~{minutes}m {seconds}s left"
        return f"~{minutes}m left"
    if minutes > 0:
        return f"~{hours}h {minutes:02d}m left"
    return f"~{hours}h left"
