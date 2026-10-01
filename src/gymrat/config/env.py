"""Read ``GYMRAT_*`` environment variables into ``CliFlags``-shaped overrides.

Each reader returns an :class:`EnvResult` rather than raising, so the caller in
:mod:`gymrat.config` decides whether a problem throws (the CLI path) or is
collected. An unset variable yields an empty result so the next source in the
precedence chain -- config file, then built-in default -- can supply the value.
"""

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

MAX_TIMEOUT_SECONDS = 2_147_483
"""Largest ``timeout_seconds`` a 32-bit millisecond timer can represent."""

MAX_SAFE_INTEGER = 2**53 - 1
"""Largest ``samples`` count, matching JavaScript's ``Number.MAX_SAFE_INTEGER``.

Every source that can supply ``samples`` -- the ``--samples`` flag,
``GYMRAT_SAMPLES``, and the config file -- shares this ceiling, so the same input
is accepted or rejected no matter which one it arrives through.
"""


@dataclass(frozen=True, slots=True)
class EnvResult:
    """The outcome of reading one env var: a value, a problem, or neither.

    ``value`` and ``problem`` are mutually exclusive; both are ``None`` when the
    variable is unset.
    """

    value: str | int | None = None
    problem: str | None = None


def env_string_result(env_var: str) -> EnvResult:
    """Read a ``GYMRAT_*`` string env var, returning its value or a problem.

    A whitespace-only value is rejected alongside the empty string: these vars
    name work to do -- a command to run, or a config path to load -- and a blank
    value would run as a no-op shell or resolve to a meaningless path.

    Args:
        env_var: Name of the ``GYMRAT_*`` environment variable to read.

    Returns:
        An :class:`EnvResult` with the raw string value, a problem, or neither
        when the variable is unset.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return EnvResult()
    if raw.strip() == "":
        got = json.dumps(raw)
        return EnvResult(
            problem=f"Invalid value for {env_var}: expected a non-empty string, got {got}"
        )
    return EnvResult(value=raw)


def is_positive_integer(raw: str) -> bool:
    """Whether ``raw`` names a positive integer.

    Every source of a positive-integer setting -- the ``--samples`` and
    ``--timeout`` flags and their ``GYMRAT_*`` env vars -- applies this one rule,
    so the same input is accepted or rejected no matter which one it arrives
    through. ``int`` alone is too lenient: it accepts surrounding whitespace, a
    sign, ``_`` separators, and non-ASCII digits.

    Args:
        raw: The text as the user wrote it.

    Returns:
        ``True`` when ``raw`` is only ASCII digits and not all of them are zero.
    """
    return raw.isascii() and raw.isdigit() and raw.strip("0") != ""


def env_positive_int_result(env_var: str, maximum: int | None = None) -> EnvResult:
    """Read a ``GYMRAT_*`` positive-integer env var, returning its value or a problem.

    The value must satisfy :func:`is_positive_integer`.

    Args:
        env_var: Name of the ``GYMRAT_*`` environment variable to read.
        maximum: Largest accepted value, or ``None`` for no upper bound.

    Returns:
        An :class:`EnvResult` with the parsed integer, a problem, or neither
        when the variable is unset.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return EnvResult()
    value: int | None = None
    if is_positive_integer(raw):
        # A digit run past the interpreter's int-conversion limit raises here;
        # it is past every ceiling too, so it is reported like any bad value.
        with contextlib.suppress(ValueError):
            value = int(raw)
    if value is not None and (maximum is None or value <= maximum):
        return EnvResult(value=value)
    phrase = "a positive integer"
    got = json.dumps(raw)
    return EnvResult(problem=f"Invalid value for {env_var}: expected {phrase}, got {got}")


#: Each ``GYMRAT_*`` string field's ``(CliFlags field, env var)`` association.
STRING_ENV_FIELDS: tuple[tuple[str, str], ...] = (
    ("bench", "GYMRAT_BENCH"),
    ("prepare", "GYMRAT_PREPARE"),
    ("adapter", "GYMRAT_ADAPTER"),
)

_NumberReader = Callable[[str], EnvResult]

#: Each ``GYMRAT_*`` numeric field's ``(CliFlags field, env var, reader)`` association.
NUMBER_ENV_FIELDS: tuple[tuple[str, str, _NumberReader], ...] = (
    ("samples", "GYMRAT_SAMPLES", partial(env_positive_int_result, maximum=MAX_SAFE_INTEGER)),
    ("timeout", "GYMRAT_TIMEOUT", partial(env_positive_int_result, maximum=MAX_TIMEOUT_SECONDS)),
)
