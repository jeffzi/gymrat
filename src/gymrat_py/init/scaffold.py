"""Write the gymrat config, runbook stub, and skill file for ``init``.

A ``ScaffoldRequest`` carries the three user choices (bench command, runbook
flag, skill-install flag). The config is serialized as hand-written TOML —
``json.dumps`` escapes the bench string, which round-trips through any TOML
parser because every JSON basic-string escape is valid TOML. Validation runs
before anything is written, so an invalid config leaves no files behind. The
bundled skill is read next (only when installing), so a broken install fails
before the runbook stub is created. Among the writes the config lands last and
with an exclusive create, so a failure partway through never leaves a
``gymrat.toml`` pointing at artifacts that were not written, and an existing
config is never silently overwritten.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat_py.bundled_skill import read_bundled_skill
from gymrat_py.config import CONFIG_FILENAME, validate_config_dict

#: Path, relative to the project root, where init writes and doctor checks for the skill.
SKILL_RELATIVE_PATH = ".claude/skills/gymrat/SKILL.md"

DEFAULT_RUNBOOK_PATH = "gymrat-runbook.md"

type ArtifactStatus = Literal["created", "exists", "declined"]

_RUNBOOK_STUB = """# Optimization Runbook

## Goal

<!-- Describe the optimization goal here. -->

## Gating metrics

<!-- List the metrics that must not regress. -->

## Constraints

<!-- List any constraints on the optimization. -->

## Approaches to try

<!-- List strategies for the agent to explore. -->

`gymrat supervise` injects this file into the agent's instructions.
"""


@dataclass(frozen=True, slots=True)
class ScaffoldRequest:
    """User choices that drive the scaffold: bench command, runbook, and skill."""

    bench: str
    runbook: bool = True
    install_skill: bool = True


@dataclass(frozen=True, slots=True)
class ScaffoldArtifact:
    """The outcome of one scaffold artifact: its path and whether it was written."""

    path: str
    status: ArtifactStatus


@dataclass(frozen=True, slots=True)
class ScaffoldResult:
    """One entry per artifact :func:`scaffold` may write."""

    config: ScaffoldArtifact
    runbook: ScaffoldArtifact
    skill: ScaffoldArtifact


def _serialize_config(config: dict[str, object]) -> str:
    """Hand-written TOML: one key per line, trailing newline.

    ``json.dumps`` escapes the bench value — every JSON basic-string escape is a
    valid TOML basic-string escape, so the result round-trips through any TOML
    parser.
    """
    lines = [f"bench = {json.dumps(config['bench'])}"]
    if "runbook" in config:
        lines.append(f'runbook = "{config["runbook"]}"')
    return "\n".join(lines) + "\n"


def _write_runbook(base_dir: str, *, runbook: bool) -> ScaffoldArtifact:
    if not runbook:
        return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="declined")

    full_path = Path(base_dir) / DEFAULT_RUNBOOK_PATH
    if full_path.exists():
        return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="exists")

    full_path.write_text(_RUNBOOK_STUB, encoding="utf-8")
    return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="created")


def _write_skill(base_dir: str, content: str) -> ScaffoldArtifact:
    full_path = Path(base_dir) / SKILL_RELATIVE_PATH
    if full_path.exists():
        return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")


def _write_config(base_dir: str, content: str) -> ScaffoldArtifact:
    full_path = Path(base_dir) / CONFIG_FILENAME
    with full_path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return ScaffoldArtifact(path=CONFIG_FILENAME, status="created")


def scaffold(base_dir: str, request: ScaffoldRequest) -> ScaffoldResult:
    """Write the config, runbook stub, and skill file for ``base_dir``.

    Returns a :class:`ScaffoldResult` describing each artifact. Raises a
    :class:`GymratError` when the config fails validation or the bundled skill
    cannot be read, and :class:`FileExistsError` when ``gymrat.toml`` already
    exists — in every failure case no partial scaffold is left behind.
    """
    config_dict: dict[str, object] = {"bench": request.bench}
    if request.runbook:
        config_dict["runbook"] = DEFAULT_RUNBOOK_PATH
    validate_config_dict(config_dict)

    config_content = _serialize_config(config_dict)

    skill_content = read_bundled_skill() if request.install_skill else None

    runbook_artifact = _write_runbook(base_dir, runbook=request.runbook)
    skill_artifact = (
        ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="declined")
        if skill_content is None
        else _write_skill(base_dir, skill_content)
    )
    config_artifact = _write_config(base_dir, config_content)

    return ScaffoldResult(config=config_artifact, runbook=runbook_artifact, skill=skill_artifact)
