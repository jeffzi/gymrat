"""The ``init`` command: scaffold a gymrat.toml, skill file, and runbook.

Resolves the base directory to the repository root when inside one (so a run from
a subdirectory still scaffolds at the root) and the process cwd otherwise; git is
not required. An existing ``gymrat.toml`` at that base is refused before the
scaffold runs. The ``--bench`` flag is required; ``--no-runbook`` and
``--no-skill`` suppress those artifacts. The artifact summary is written to
stdout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from gymrat_py.cli.shared import (
    DebugOption,
    NoColorOption,
    color_override_of,
    exit_with_error,
    resolve_stream_color,
    set_debug_mode,
    write_and_flush,
)
from gymrat_py.config import CONFIG_FILENAME, find_implicit_base
from gymrat_py.errors import GymratError
from gymrat_py.init.scaffold import (
    ScaffoldArtifact,
    ScaffoldRequest,
    ScaffoldResult,
    scaffold,
)
from gymrat_py.report.style import RENDER_WIDTH, highlight_inline_code, render_lines

_BenchOption = Annotated[str | None, typer.Option("--bench", help="bench command")]
_NoRunbookOption = Annotated[bool, typer.Option("--no-runbook", help="skip the runbook")]
_NoSkillOption = Annotated[bool, typer.Option("--no-skill", help="skip the skill file")]


def _display_path(base_dir: str, relative: str) -> str:
    """Return a path navigable from the user's cwd, not from the project root."""
    return os.path.relpath(str(Path(base_dir) / relative))


def _format_artifact(label: str, artifact: ScaffoldArtifact, base_dir: str) -> str:
    if artifact.status == "declined":
        return f"  {label} declined"
    display = _display_path(base_dir, artifact.path)
    if artifact.status == "is a directory":
        return f"  {label} is a directory at {display}"
    verb = "created at" if artifact.status == "created" else "already exists at"
    return f"  {label} {verb} {display}"


def _format_summary(result: ScaffoldResult, base_dir: str, *, color: bool | None = None) -> str:
    doc = "\n".join(
        [
            escape(_format_artifact("Config:", result.config, base_dir)),
            escape(_format_artifact("Runbook:", result.runbook, base_dir)),
            escape(_format_artifact("Skill:", result.skill, base_dir)),
            "",
            highlight_inline_code("Run `gymrat doctor` to verify the setup."),
        ]
    )
    return render_lines(doc, color=color, width=RENDER_WIDTH)


def init_command(
    *,
    bench: _BenchOption = None,
    no_runbook: _NoRunbookOption = False,
    no_skill: _NoSkillOption = False,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Scaffold a gymrat.toml, skill file, and runbook."""
    if debug:
        set_debug_mode(True)
    if bench is None:
        exit_with_error(GymratError("Missing --bench flag."))

    color_override = color_override_of(not no_color)
    resolved_color = resolve_stream_color(color_override, sys.stdout)

    base_dir = find_implicit_base()
    config_path = Path(base_dir) / CONFIG_FILENAME
    if config_path.exists():
        message = f"{config_path} already exists."
        hint = "Edit it directly, or run `gymrat doctor` to verify the setup."
        exit_with_error(GymratError(message, hint=hint))

    try:
        request = ScaffoldRequest(
            bench=bench,
            runbook=not no_runbook,
            install_skill=not no_skill,
        )
        result = scaffold(base_dir, request)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)

    write_and_flush(sys.stdout, _format_summary(result, base_dir, color=resolved_color) + "\n")
