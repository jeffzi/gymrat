"""The ``gymrat supervise`` flag surface: typer annotations and the parsed options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast

import typer

from gymrat.cli.shared import parse_max_minutes, parse_positive_number
from gymrat.config import EFFORT_LEVELS, EFFORT_PHRASE, Effort

PromptArgument = Annotated[
    str | None,
    typer.Argument(metavar="[PROMPT]", help="optimization prompt for the agent"),
]
MaxMinutesOption = Annotated[
    float,
    typer.Option(
        "--max-minutes",
        parser=parse_max_minutes,
        metavar="<float>",
        help="wall-clock cap in minutes, counted from when the baseline is recorded",
    ),
]
MaxUsdOption = Annotated[
    float | None,
    typer.Option(
        "--max-usd", parser=parse_positive_number, metavar="<float>", help="spend cap in USD"
    ),
]
LogOption = Annotated[str | None, typer.Option("--log", help="path for the JSONL event log")]
ModelOption = Annotated[
    str | None, typer.Option("--model", help="model to use for the agent session")
]
AllowDirtyOption = Annotated[
    bool, typer.Option("--allow-dirty", help="allow launching with uncommitted changes")
]
ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        help="launch even when the cap cannot fit one iteration or a stop condition is already met",
    ),
]
NoFinalizeOption = Annotated[
    bool,
    typer.Option("--no-finalize", help="leave the session open instead of finalizing it on exit"),
]


def _parse_effort(value: str) -> Effort:
    if value not in EFFORT_LEVELS:
        raise typer.BadParameter(EFFORT_PHRASE)
    return cast("Effort", value)


EffortOption = Annotated[
    Effort | None,
    typer.Option("--effort", parser=_parse_effort, metavar="<level>", help="effort level"),
]


@dataclass(frozen=True, slots=True)
class Options:
    """The parsed flag surface, gathered so the run helpers take one argument."""

    prompt: str | None
    max_minutes: float
    max_usd: float | None
    log: str | None
    baseline: str | None
    model: str | None
    effort: Effort | None
    allow_dirty: bool
    force: bool
    color: bool | None
    finalize: bool
