"""Generic helpers that depend on the standard library alone.

Nothing here imports a project module or names a project type: each helper
would make sense pasted unchanged into an unrelated repository.
"""

import contextlib
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

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


def pluralize(count: int, noun: str, plural: str | None = None) -> str:
    """Inflect ``noun`` for ``count``, or use the explicit ``plural`` instead.

    A count of one keeps ``noun`` as given; any other count — zero and
    negatives included — takes the plural form.

    Args:
        count: The count that determines singular vs. plural form.
        noun: The singular form of the noun.
        plural: An explicit plural form to use instead of the regular inflection.

    Returns:
        The count followed by the correctly inflected noun.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {plural if plural is not None else _regular_plural(noun)}"


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
