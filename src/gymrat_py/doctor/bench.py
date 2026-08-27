"""The doctor bench section.

Validates bench configuration without executing the bench command: the adapter
name resolves, a bench command is set, and the command's executable is on PATH.
"""

import shutil

from gymrat_py.adapters import get_adapter
from gymrat_py.doctor.checks import Check, CheckSection
from gymrat_py.errors import GymratError, hint_of

_NO_BENCH_HINT = 'Set the bench command with --bench or the "bench" config key'


def build_bench_section(*, bench: str | None, adapter: str) -> CheckSection:
    """Build the "Bench" section by validating config, without running anything."""
    checks: list[Check] = []

    try:
        get_adapter(adapter)
    except GymratError as error:
        return _bench_section(
            [Check(name="adapter", status="fail", detail=str(error), hint=hint_of(error))]
        )
    checks.append(Check(name="adapter", status="ok", detail=f"adapter: {adapter}"))

    if bench is None:
        checks.append(
            Check(
                name="bench",
                status="fail",
                detail="No bench command configured",
                hint=_NO_BENCH_HINT,
            )
        )
        return _bench_section(checks)

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

    return _bench_section(checks)


def _bench_section(checks: list[Check]) -> CheckSection:
    return CheckSection(title="Bench", checks=checks)
