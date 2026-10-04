"""Write the gymrat config, runbook stub, and skill file for ``init``.

A ``ScaffoldRequest`` carries the three user choices (bench command, runbook
flag, skill-install flag). The config is serialized as hand-written TOML.
Validation and the bundled-skill read run before any file is written, so a
broken install leaves nothing behind.

The scaffold is re-runnable: an existing ``gymrat.toml`` is left byte-identical
and reported as ``exists``, while the runbook and skill are still filled in.
``bench`` is therefore only required when the config has to be written. If a
later artifact write fails, a config this run created is removed so no partial
scaffold is left — one that was already there is never touched.
"""

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat.bundled_skill import read_bundled_skill
from gymrat.config import CONFIG_FILENAME, validate_config_dict
from gymrat.errors import GymratError
from gymrat.utils import write_text_atomic

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
    """User choices that drive the scaffold.

    ``bench`` is required only when the config must be written; ``runbook``
    and ``install_skill`` gate whether those artifacts are created.
    """

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


def _write_artifact(base_dir: Path, relative: str, content: str) -> ScaffoldArtifact:
    """Write ``content`` to ``relative`` unless a file is already there.

    Args:
        base_dir: The project root the artifact is written into.
        relative: The artifact's path, relative to ``base_dir``.
        content: The text to write.

    Returns:
        The artifact, reported as ``exists`` when a file already occupied the
        path and as ``created`` otherwise.

    Raises:
        GymratError: When the file cannot be written.
    """
    full_path = base_dir / relative
    if full_path.is_file():
        return ScaffoldArtifact(path=relative, status="exists")

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot write {relative} in {base_dir}"
        raise GymratError(msg, hint=str(exc)) from exc
    return ScaffoldArtifact(path=relative, status="created")


def _prepare_config(request: ScaffoldRequest) -> str:
    """Validate the scaffold config and serialize it as hand-written TOML.

    One key per line, with a trailing newline. ``json.dumps`` escapes each
    value — every JSON basic-string escape is a valid TOML basic-string escape,
    so the result round-trips through any TOML parser.

    Args:
        request: The user choices; ``bench`` goes into the config and
            ``runbook`` decides whether the runbook path does.

    Returns:
        The TOML string with a trailing newline.

    Raises:
        GymratError: When the config fails validation.
    """
    config_dict: dict[str, object] = {"bench": request.bench}
    if request.runbook:
        config_dict["runbook"] = DEFAULT_RUNBOOK_PATH
    validate_config_dict(config_dict)
    return "".join(f"{key} = {json.dumps(value)}\n" for key, value in config_dict.items())


def _write_config(base_dir: Path, content: str) -> ScaffoldArtifact:
    """Write the config atomically, so a failed write never leaves a truncated ``gymrat.toml``.

    Args:
        base_dir: The project root the config is written into.
        content: The serialized TOML text.

    Returns:
        The scaffold artifact describing the written config file.

    Raises:
        GymratError: When the file cannot be written.
    """
    try:
        write_text_atomic(base_dir / CONFIG_FILENAME, content)
    except OSError as exc:
        msg = f"Cannot write {CONFIG_FILENAME} in {base_dir}"
        raise GymratError(msg, hint=str(exc)) from exc
    return ScaffoldArtifact(path=CONFIG_FILENAME, status="created")


def _blocked_paths(base_dir: Path, request: ScaffoldRequest) -> list[str]:
    """The requested artifact paths a non-regular file occupies.

    Symlinks (including dangling ones) are always blocked — writing through a
    symlink would place the content at a location the user did not choose.
    Directories are blocked because they cannot be opened as regular files.

    Args:
        base_dir: The base directory the artifact paths are resolved against.
        request: The user choices that decide which artifacts are written.

    Returns:
        Each requested path holding a symlink, directory, or other non-regular
        file, in write order.
    """
    requested = [
        (CONFIG_FILENAME, True),
        (DEFAULT_RUNBOOK_PATH, request.runbook),
        (SKILL_RELATIVE_PATH, request.install_skill),
    ]
    return [
        relative
        for relative, wanted in requested
        if wanted
        and ((full := base_dir / relative).is_symlink() or (full.exists() and not full.is_file()))
    ]


def scaffold(base_dir: str | Path, request: ScaffoldRequest) -> ScaffoldResult:
    """Write the config, runbook stub, and skill file for ``base_dir``.

    An existing ``gymrat.toml`` is reported as ``exists`` and left
    byte-identical; the remaining artifacts are still created, which makes a
    re-run the way to restore a deleted runbook or skill. In every failure
    case no partial scaffold is left behind.

    Args:
        base_dir: The project root to scaffold into.
        request: Which artifacts to create and the bench command to embed in
            the config.

    Returns:
        A :class:`ScaffoldResult` describing each artifact.

    Raises:
        GymratError: When the config fails validation, the bundled skill cannot
            be read, a non-regular file (directory, symlink) occupies an artifact
            path, or a filesystem error prevents writing.
    """
    base_dir = Path(base_dir)
    config_path = base_dir / CONFIG_FILENAME
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
        runbook_artifact = (
            _write_artifact(base_dir, DEFAULT_RUNBOOK_PATH, _RUNBOOK_STUB)
            if request.runbook
            else ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="declined")
        )
        skill_artifact = (
            ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="declined")
            if skill_content is None
            else _write_artifact(base_dir, SKILL_RELATIVE_PATH, skill_content)
        )
    except BaseException:
        # Only roll back a config this run created; a pre-existing one is the user's.
        if config_content is not None:
            with contextlib.suppress(OSError):
                config_path.unlink(missing_ok=True)
        raise

    return ScaffoldResult(config=config_artifact, runbook=runbook_artifact, skill=skill_artifact)
