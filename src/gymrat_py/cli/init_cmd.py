"""The ``init`` command: scaffold a gymrat.json, skill file, and runbook.

Resolves the base directory to the repository root when inside one (so a run from
a subdirectory still scaffolds at the root) and the process cwd otherwise; git is
not required. An existing ``gymrat.json`` at that base is refused before the
wizard runs. The wizard prompts on stderr and reads stdin, and the artifact
summary is written to stdout so the two channels stay separable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from gymrat_py.cli.shared import (
    MAX_SAFE_INTEGER,
    DebugOption,
    exit_with_error,
    parse_positive_integer_up_to,
    parse_stop_target_value,
    set_debug_mode,
    write_and_flush,
)
from gymrat_py.config import CONFIG_FILENAME, find_implicit_base
from gymrat_py.errors import GymratError
from gymrat_py.init.scaffold import ScaffoldArtifact, ScaffoldResult, scaffold
from gymrat_py.init.wizard import WizardOptions, run_wizard
from gymrat_py.report.style import RENDER_WIDTH, highlight_inline_code, render_lines

_BenchOption = Annotated[str | None, typer.Option("--bench", help="bench command")]
_AdapterOption = Annotated[str | None, typer.Option("--adapter", help="adapter type")]
_ChecksOption = Annotated[str | None, typer.Option("--checks", help="checks command")]
_StopTargetOption = Annotated[
    float | None,
    typer.Option("--stop-target", parser=parse_stop_target_value, help="stop target value"),
]
_StopMaxIterationsOption = Annotated[
    int | None,
    typer.Option(
        "--stop-max-iterations",
        parser=parse_positive_integer_up_to(MAX_SAFE_INTEGER),
        help="stop max iterations",
    ),
]
_PrimaryOption = Annotated[str | None, typer.Option("--primary", help="primary metric name")]
_RunbookOption = Annotated[str | None, typer.Option("--runbook", help="create the runbook at PATH")]
_NoRunbookOption = Annotated[bool, typer.Option("--no-runbook", help="skip the runbook")]
_SkillOption = Annotated[
    bool | None, typer.Option("--skill/--no-skill", help="install (or skip) the skill file")
]
_YesOption = Annotated[bool, typer.Option("--yes", "-y", help="non-interactive mode")]


def _resolve_runbook_flag(runbook: str | None, *, no_runbook: bool) -> str | bool | None:
    """Fold the ``--runbook``/``--no-runbook`` pair into the wizard's tri-state value."""
    if no_runbook:
        return False
    return runbook


def _format_artifact(label: str, artifact: ScaffoldArtifact) -> str:
    if artifact.status == "declined":
        return f"  {label} declined"
    verb = "created at" if artifact.status == "created" else "already exists at"
    return f"  {label} {verb} {artifact.path}"


def _format_summary(result: ScaffoldResult) -> str:
    doc = "\n".join(
        [
            escape(_format_artifact("Config:", result.config)),
            escape(_format_artifact("Runbook:", result.runbook)),
            escape(_format_artifact("Skill:", result.skill)),
            "",
            highlight_inline_code("Run `gymrat doctor` to verify the setup."),
        ]
    )
    return render_lines(doc, color=None, width=RENDER_WIDTH)


def init_command(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the scaffold surface
    *,
    bench: _BenchOption = None,
    adapter: _AdapterOption = None,
    checks: _ChecksOption = None,
    stop_target: _StopTargetOption = None,
    stop_max_iterations: _StopMaxIterationsOption = None,
    primary: _PrimaryOption = None,
    runbook: _RunbookOption = None,
    no_runbook: _NoRunbookOption = False,
    skill: _SkillOption = None,
    yes: _YesOption = False,
    debug: DebugOption = False,
) -> None:
    """Scaffold a gymrat.json, skill file, and runbook."""
    set_debug_mode(debug)
    base_dir = find_implicit_base()
    config_path = Path(base_dir) / CONFIG_FILENAME
    if config_path.exists():
        message = f"{config_path} already exists."
        hint = "Edit it directly, or run `gymrat doctor` to verify the setup."
        exit_with_error(GymratError(message, hint=hint))

    runbook_option = _resolve_runbook_flag(runbook, no_runbook=no_runbook)
    try:
        wizard_result = run_wizard(
            WizardOptions(
                input=sys.stdin,
                output=sys.stderr,
                bench=bench,
                adapter=adapter,
                checks=checks,
                stop_target=stop_target,
                stop_max_iterations=stop_max_iterations,
                primary=primary,
                runbook=runbook_option,
                skill=skill,
                yes=yes,
            )
        )
        result = scaffold(base_dir, wizard_result)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)

    write_and_flush(sys.stdout, _format_summary(result) + "\n")
