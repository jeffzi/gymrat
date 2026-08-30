"""Write the gymrat config, runbook stub, and skill file for ``init``.

A ``ScaffoldRequest`` carries the three user choices (bench command, runbook
flag, skill-install flag). The config is serialized as hand-written TOML —
``json.dumps`` escapes the bench string, which round-trips through any TOML
parser because every JSON basic-string escape is valid TOML. Validation and
the bundled-skill read run before any file is written, so a broken install
leaves nothing behind.

The scaffold is re-runnable: an existing ``gymrat.toml`` is left byte-identical
and reported as ``exists``, while the runbook and skill are still filled in.
``bench`` is therefore only required when the config has to be written. If a
later artifact write fails, a config this run created is removed so no partial
scaffold is left — one that was already there is never touched.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat.bundled_skill import read_bundled_skill
from gymrat.config import CONFIG_FILENAME, validate_config_dict
from gymrat.errors import GymratError

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

    bench: str | None = None
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

    try:
        full_path.write_text(_RUNBOOK_STUB, encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot write {DEFAULT_RUNBOOK_PATH} in {base_dir}"
        raise GymratError(msg, hint=str(exc)) from exc
    return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="created")


def _write_skill(base_dir: str, content: str) -> ScaffoldArtifact:
    full_path = Path(base_dir) / SKILL_RELATIVE_PATH
    if full_path.exists():
        if not full_path.is_file():
            return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="is a directory")
        return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot write {SKILL_RELATIVE_PATH} in {base_dir}"
        raise GymratError(msg, hint=str(exc)) from exc
    return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")


def _prepare_config(request: ScaffoldRequest) -> str:
    """Validate the requested config and render it as TOML."""
    config_dict: dict[str, object] = {"bench": request.bench}
    if request.runbook:
        config_dict["runbook"] = DEFAULT_RUNBOOK_PATH
    validate_config_dict(config_dict)
    return _serialize_config(config_dict)


def _write_config(base_dir: str, content: str) -> ScaffoldArtifact:
    """Write the config via a temp file + ``os.replace`` for atomicity.

    A crash or permission error mid-write never leaves a truncated
    ``gymrat.toml`` — the temp file is cleaned up on failure.
    """
    full_path = Path(base_dir) / CONFIG_FILENAME
    fd = -1
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="gymrat.toml.", dir=base_dir, suffix=".tmp")
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = -1
        Path(tmp_path).replace(full_path)
        tmp_path = None
    except OSError as exc:
        msg = f"Cannot write {CONFIG_FILENAME} in {base_dir}"
        raise GymratError(msg, hint=str(exc)) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
    return ScaffoldArtifact(path=CONFIG_FILENAME, status="created")


def _path_blocked(base_dir: str, relative: str) -> bool:
    """True when a non-regular file occupies ``relative``.

    Symlinks (including dangling ones) are always blocked — writing through a
    symlink would place the content at a location the user did not choose.
    Directories are blocked because they cannot be opened as regular files.
    """
    full = Path(base_dir) / relative
    if full.is_symlink():
        return True
    return full.exists() and not full.is_file()


def _blocked_paths(base_dir: str, request: ScaffoldRequest) -> list[str]:
    """Collect artifact paths that are blocked by a non-regular file."""
    blocked: list[str] = []
    if _path_blocked(base_dir, CONFIG_FILENAME):
        blocked.append(CONFIG_FILENAME)
    if request.runbook and _path_blocked(base_dir, DEFAULT_RUNBOOK_PATH):
        blocked.append(DEFAULT_RUNBOOK_PATH)
    if request.install_skill and _path_blocked(base_dir, SKILL_RELATIVE_PATH):
        blocked.append(SKILL_RELATIVE_PATH)
    return blocked


def scaffold(base_dir: str, request: ScaffoldRequest) -> ScaffoldResult:
    """Write the config, runbook stub, and skill file for ``base_dir``.

    Returns a :class:`ScaffoldResult` describing each artifact. An existing
    ``gymrat.toml`` is reported as ``exists`` and left byte-identical; the
    remaining artifacts are still created, which makes a re-run the way to
    restore a deleted runbook or skill.

    Raises :class:`GymratError` when:

    - The config fails validation or the bundled skill cannot be read.
    - A non-regular file (directory, symlink) occupies an artifact path.
    - A filesystem error (permission denied, read-only FS) prevents writing.

    In every failure case no partial scaffold is left behind.
    """
    config_path = Path(base_dir) / CONFIG_FILENAME
    try:
        config_exists = config_path.exists()
    except OSError as error:
        msg = f"Cannot access {config_path}: {error}"
        raise GymratError(msg, hint="Check directory permissions.") from error

    config_content = None if config_exists else _prepare_config(request)
    skill_content = read_bundled_skill() if request.install_skill else None

    blocked = _blocked_paths(base_dir, request)
    if blocked:
        paths = ", ".join(blocked)
        msg = f"Blocked path: {paths}"
        raise GymratError(msg, hint="Remove or rename the blocking entry and re-run.")

    config_artifact = (
        ScaffoldArtifact(path=CONFIG_FILENAME, status="exists")
        if config_content is None
        else _write_config(base_dir, config_content)
    )
    try:
        runbook_artifact = _write_runbook(base_dir, runbook=request.runbook)
        skill_artifact = (
            ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="declined")
            if skill_content is None
            else _write_skill(base_dir, skill_content)
        )
    except BaseException:
        # Only roll back a config this run created; a pre-existing one is the user's.
        if config_content is not None:
            config_path.unlink(missing_ok=True)
        raise

    return ScaffoldResult(config=config_artifact, runbook=runbook_artifact, skill=skill_artifact)
