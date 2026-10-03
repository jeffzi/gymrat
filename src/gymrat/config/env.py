"""Read ``GYMRAT_*`` environment variables into ``CliFlags``-shaped overrides.

Each reader returns an :class:`EnvResult` rather than raising, so the caller in
:mod:`gymrat.config.resolve` decides whether a problem throws (the CLI path) or is
collected. An unset variable yields an empty result so the next source in the
precedence chain -- config file, then built-in default -- can supply the value.
"""

import json
import os
from dataclasses import dataclass

MAX_TIMEOUT_SECONDS = 2_147_483
"""Largest ``timeout_seconds`` a 32-bit millisecond timer can represent."""

MAX_SAFE_INTEGER = 2**53 - 1
"""Largest ``samples`` count, matching JavaScript's ``Number.MAX_SAFE_INTEGER``.

Every source that can supply ``samples`` -- the ``--samples`` flag,
``GYMRAT_SAMPLES``, and the config file -- shares this ceiling, so the same input
is accepted or rejected no matter which one it arrives through.
"""


@dataclass(frozen=True, slots=True)
class EnvResult[T]:
    """The outcome of reading one env var: a value, a problem, or neither.

    ``value`` and ``problem`` are mutually exclusive; both are ``None`` when the
    variable is unset.
    """

    value: T | None = None
    problem: str | None = None


def _env_problem(env_var: str, phrase: str, raw: str) -> str:
    return f"Invalid value for {env_var}: expected {phrase}, got {json.dumps(raw)}"


def env_string_result(env_var: str) -> EnvResult[str]:
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
        return EnvResult(problem=_env_problem(env_var, "a non-empty string", raw))
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


def parse_bounded_positive_int(raw: str, maximum: int) -> int | None:
    """Parse a positive integer no larger than ``maximum``.

    Args:
        raw: The text as the user wrote it.
        maximum: Largest accepted value.

    Returns:
        The integer, or ``None`` when ``raw`` fails :func:`is_positive_integer`
        or names a value above ``maximum``.
    """
    if not is_positive_integer(raw):
        return None
    # More significant digits than ``maximum`` means above it; comparing lengths
    # first keeps ``int`` away from a run past the interpreter's conversion limit.
    digits = raw.lstrip("0")
    if len(digits) > len(str(maximum)) or int(digits) > maximum:
        return None
    return int(digits)


def env_positive_int_result(env_var: str, maximum: int) -> EnvResult[int]:
    """Read a ``GYMRAT_*`` positive-integer env var, returning its value or a problem.

    Args:
        env_var: Name of the ``GYMRAT_*`` environment variable to read.
        maximum: Largest accepted value.

    Returns:
        An :class:`EnvResult` with the parsed integer, a problem when the value
        fails :func:`parse_bounded_positive_int`, or neither when the variable
        is unset.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return EnvResult()
    value = parse_bounded_positive_int(raw, maximum)
    if value is None:
        return EnvResult(problem=_env_problem(env_var, "a positive integer", raw))
    return EnvResult(value=value)


#: Each ``GYMRAT_*`` string field's ``(CliFlags field, env var)`` association.
STRING_ENV_FIELDS: tuple[tuple[str, str], ...] = (
    ("bench", "GYMRAT_BENCH"),
    ("prepare", "GYMRAT_PREPARE"),
    ("adapter", "GYMRAT_ADAPTER"),
)

#: Each ``GYMRAT_*`` numeric field's ``(CliFlags field, env var, maximum)`` association.
NUMBER_ENV_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("samples", "GYMRAT_SAMPLES", MAX_SAFE_INTEGER),
    ("timeout", "GYMRAT_TIMEOUT", MAX_TIMEOUT_SECONDS),
)
