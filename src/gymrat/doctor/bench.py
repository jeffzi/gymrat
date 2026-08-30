"""The doctor bench section.

Validates bench configuration without executing the bench command: the adapter
name resolves, a bench command is set, and the command's executable is on PATH.
"""

import re
import shlex
import shutil

from gymrat.adapters import get_adapter
from gymrat.doctor.checks import Check, CheckSection
from gymrat.errors import GymratError, hint_of

_NO_BENCH_HINT = 'Set the bench command with --bench or the "bench" config key'
_TITLE = "Bench"

_SHELL_OPERATOR_RE = re.compile(r"[;&|(){}<>]")


def _first_command_word(bench: str) -> str | None:
    """Extract the first real executable from a shell command string.

    Skips env-var assignments (``VAR=val``) and returns ``None`` when the command
    contains shell metacharacters — the PATH check is meaningless for compound
    shell expressions.
    """
    if _SHELL_OPERATOR_RE.search(bench):
        return None

    try:
        tokens = shlex.split(bench)
    except ValueError:
        return None

    for token in tokens:
        if "=" in token:
            continue
        return token
    return None


def build_bench_section(
    *, bench: str | None, adapter: str, config_problems: bool = False
) -> CheckSection:
    """Build the "Bench" section by validating config, without running anything.

    When ``config_problems`` is True and ``bench`` is None the section collapses
    to a single skip placeholder — the bench value was never resolved, so a FAIL
    would be misleading.
    """
    if config_problems and bench is None:
        return CheckSection(
            title=_TITLE,
            checks=[Check(name="bench", status="ok", detail="Skipped — fix config errors first")],
        )

    try:
        get_adapter(adapter)
    except GymratError as error:
        return CheckSection(
            title=_TITLE,
            checks=[Check(name="adapter", status="fail", detail=str(error), hint=hint_of(error))],
        )
    checks: list[Check] = [Check(name="adapter", status="ok", detail=f"adapter: {adapter}")]

    if bench is None:
        checks.append(
            Check(
                name="bench",
                status="fail",
                detail="No bench command configured",
                hint=_NO_BENCH_HINT,
            )
        )
        return CheckSection(title=_TITLE, checks=checks)

    checks.append(Check(name="bench", status="ok", detail=f"bench: {bench}"))

    executable = _first_command_word(bench)
    if executable is not None:
        if shutil.which(executable) is not None:
            checks.append(
                Check(name="executable", status="ok", detail=f"{executable} is available on PATH")
            )
        else:
            checks.append(
                Check(
                    name="executable",
                    status="warn",
                    detail=f"{executable} was not found on PATH",
                )
            )

    return CheckSection(title=_TITLE, checks=checks)
