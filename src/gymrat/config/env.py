"""Read ``GYMRAT_*`` environment variables into ``CliFlags``-shaped overrides.

Each reader returns an :class:`EnvResult` rather than raising, so the caller in
:mod:`gymrat.config` decides whether a problem throws (the CLI path) or is
collected. An unset variable yields an empty result so the next source in the
precedence chain -- config file, then built-in default -- can supply the value.
"""

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


def env_positive_int_result(env_var: str, maximum: int | None = None) -> EnvResult:
    r"""Read a ``GYMRAT_*`` positive-integer env var, returning its value or a problem.

    The ``isascii() and isdigit()`` check rejects sign, decimal point,
    exponent, and hex notation, so only a bare run of ASCII digits parses.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return EnvResult()
    try:
        valid = (
            raw.isascii()
            and raw.isdigit()
            and int(raw) >= 1
            and (maximum is None or int(raw) <= maximum)
        )
    except ValueError:
        valid = False
    if valid:
        return EnvResult(value=int(raw))
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
