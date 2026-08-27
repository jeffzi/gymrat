"""Write the gymrat config, runbook stub, and skill file for ``init``.

A ``ScaffoldRequest`` carries the three user choices (bench command, runbook
flag, skill-install flag). The config is serialized as hand-written TOML —
``json.dumps`` escapes the bench string, which round-trips through any TOML
parser because every JSON basic-string escape is valid TOML. Validation and
the bundled-skill read run before any file is written, so a broken install
leaves nothing behind. The config is then written first with an exclusive
create, so an existing file raises ``FileExistsError`` before runbook or skill
writes start. If a later artifact write fails, the config is removed so no
partial scaffold is left.
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

type ArtifactStatus = Literal["created", "exists", "declined", "is a directory"]

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
        if not full_path.is_file():
            return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="is a directory")
        return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="exists")

    full_path.write_text(_RUNBOOK_STUB, encoding="utf-8")
    return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="created")


def _write_skill(base_dir: str, content: str) -> ScaffoldArtifact:
    full_path = Path(base_dir) / SKILL_RELATIVE_PATH
    if full_path.exists():
        if not full_path.is_file():
            return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="is a directory")
        return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")


def _write_config(base_dir: str, content: str) -> ScaffoldArtifact:
    full_path = Path(base_dir) / CONFIG_FILENAME
    with full_path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return ScaffoldArtifact(path=CONFIG_FILENAME, status="created")


def _path_blocked(base_dir: str, relative: str) -> bool:
    """True when a non-regular file (directory, symlink to dir, etc.) occupies ``relative``."""
    full = Path(base_dir) / relative
    return full.exists() and not full.is_file()


def scaffold(base_dir: str, request: ScaffoldRequest) -> ScaffoldResult:
    """Write the config, runbook stub, and skill file for ``base_dir``.

    Returns a :class:`ScaffoldResult` describing each artifact. Raises a
    :class:`GymratError` when the config fails validation or the bundled skill
    cannot be read, and :class:`FileExistsError` when ``gymrat.toml`` already
    exists — in every failure case no partial scaffold is left behind.

    When a non-regular file (e.g. a directory) sits at the runbook or skill path,
    the scaffold returns early with no config written — the config would point at
    something unusable.
    """
    config_dict: dict[str, object] = {"bench": request.bench}
    if request.runbook:
        config_dict["runbook"] = DEFAULT_RUNBOOK_PATH
    validate_config_dict(config_dict)

    config_content = _serialize_config(config_dict)

    skill_content = read_bundled_skill() if request.install_skill else None

    # Detect non-regular files at artifact paths before any writes.
    runbook_blocked = request.runbook and _path_blocked(base_dir, DEFAULT_RUNBOOK_PATH)
    skill_blocked = request.install_skill and _path_blocked(base_dir, SKILL_RELATIVE_PATH)
    if runbook_blocked or skill_blocked:
        return ScaffoldResult(
            config=ScaffoldArtifact(path=CONFIG_FILENAME, status="declined"),
            runbook=(
                ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="is a directory")
                if runbook_blocked
                else ScaffoldArtifact(
                    path=DEFAULT_RUNBOOK_PATH,
                    status="declined" if not request.runbook else "exists",
                )
            ),
            skill=(
                ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="is a directory")
                if skill_blocked
                else ScaffoldArtifact(
                    path=SKILL_RELATIVE_PATH,
                    status="declined" if skill_content is None else "exists",
                )
            ),
        )

    # Config first: the exclusive create fails atomically when the file exists,
    # so runbook and skill are never orphaned.
    config_artifact = _write_config(base_dir, config_content)
    try:
        runbook_artifact = _write_runbook(base_dir, runbook=request.runbook)
        skill_artifact = (
            ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="declined")
            if skill_content is None
            else _write_skill(base_dir, skill_content)
        )
    except BaseException:
        (Path(base_dir) / CONFIG_FILENAME).unlink(missing_ok=True)
        raise

    return ScaffoldResult(config=config_artifact, runbook=runbook_artifact, skill=skill_artifact)
