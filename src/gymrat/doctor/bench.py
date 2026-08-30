"""The doctor bench section.

Validates bench configuration without executing the bench command: the adapter
name resolves, a bench command is set, and the command's executable is on PATH.
"""

import shutil

from gymrat.adapters import get_adapter
from gymrat.doctor.checks import Check, CheckSection
from gymrat.errors import GymratError, hint_of

_NO_BENCH_HINT = 'Set the bench command with --bench or the "bench" config key'
_TITLE = "Bench"


def build_bench_section(*, bench: str | None, adapter: str) -> CheckSection:
    """Build the "Bench" section by validating config, without running anything."""
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

    executable = bench.split()[0]
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
