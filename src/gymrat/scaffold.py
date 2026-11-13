"""Write the gymrat config, runbook stub, and skill file for ``init``.

A ``ScaffoldRequest`` carries the three user choices (bench command, runbook
flag, skill-install flag). The config is serialized as hand-written TOML.
Validation and the bundled-skill read run before any file is written, so a
broken install leaves nothing behind.

The scaffold is re-runnable: an existing ``gymrat.toml`` is left byte-identical
and reported as ``exists``, while the runbook and skill are still filled in.
``bench`` is therefore only required when the config has to be written. Every
artifact is written atomically and never over a file that appears mid-run,
which is reported as ``exists`` and kept. If a write fails, each artifact this run
created is removed so no partial scaffold is left — one that was already there
is never touched. The directories this run created go with them, unless
something else has put a file in one since: a directory that is no longer
empty is left, along with its ancestors.
"""

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat.bundled_skill import read_bundled_skill
from gymrat.config import CONFIG_FILENAME, validate_config_dict
from gymrat.errors import GymratError
from gymrat.utils import create_text_atomic

SKILL_RELATIVE_PATH = ".claude/skills/gymrat/SKILL.md"
"""Path, relative to the project root, where init writes and doctor checks for the skill."""

DEFAULT_RUNBOOK_PATH = "gymrat-runbook.md"
"""Path, relative to the project root, where init writes the runbook stub.

The config names the runbook by this path.
"""

type ArtifactStatus = Literal["created", "exists", "declined"]
"""How init reports each artifact.

``created`` when it wrote the file this run, ``exists`` when a file already occupied the path and
was left untouched, and ``declined`` when the request opted out of it.
"""

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

    Attributes:
        bench: The bench command written into the config; required only when
            the config must be written.
        runbook: Whether to write the runbook stub and name it in the config.
        install_skill: Whether to install the skill file.
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


def _make_directory(directory: Path) -> bool:
    try:
        directory.mkdir()
    except OSError:
        # A directory that is already there belongs to whoever made it, even
        # when it appeared after this run found it missing.
        if not directory.is_dir():
            raise
        return False
    return True


def _make_directories(directory: Path, base_dir: Path, created: list[Path]) -> None:
    """Create ``directory`` and its missing ancestors below ``base_dir``, recording each one made.

    ``base_dir`` itself is never created: a project root that does not exist is
    a mistyped path, so the write under it fails instead. A directory is
    recorded the moment it is made, so a failure part-way through still leaves
    ``created`` naming everything there is to roll back.

    Args:
        directory: The directory that must exist afterwards.
        base_dir: The project root, which must already exist.
        created: Extended with each directory this call created, an ancestor
            before its descendants.

    Raises:
        OSError: When a directory cannot be created, ``base_dir`` included
            when it is missing.
    """
    if directory == base_dir:
        return
    try:
        made = _make_directory(directory)
    except FileNotFoundError:
        _make_directories(directory.parent, base_dir, created)
        made = _make_directory(directory)
    if made:
        created.append(directory)


def _write_artifact(
    base_dir: Path, relative: str, content: str, created_directories: list[Path]
) -> ScaffoldArtifact:
    """Write ``content`` to ``relative`` unless a file is already there.

    The write is atomic, so a failure never leaves a truncated file that a
    later run would take for an existing artifact, and it never replaces a
    file that appears after the existence check.

    Args:
        base_dir: The project root the artifact is written into.
        relative: The artifact's path, relative to ``base_dir``.
        content: The text to write.
        created_directories: Extended with each directory created to hold the
            artifact, an ancestor before its descendants, including when the
            write then fails.

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
        _make_directories(full_path.parent, base_dir, created_directories)
        create_text_atomic(full_path, content)
    except FileExistsError:
        return ScaffoldArtifact(path=relative, status="exists")
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


def _is_blocked(base_dir: Path, relative: str) -> bool:
    full = base_dir / relative
    if full.is_symlink() or (full.exists() and not full.is_file()):
        return True
    # A file or dangling symlink where a directory of the path belongs leaves
    # nowhere to write; ``is_file`` reports such a path as missing, not blocked.
    return any(
        (ancestor := base_dir / parent).exists(follow_symlinks=False) and not ancestor.is_dir()
        for parent in Path(relative).parents
        if parent != Path()
    )


def _blocked_paths(base_dir: Path, request: ScaffoldRequest) -> list[str]:
    """The requested artifact paths a non-regular file occupies or cuts off.

    Symlinks (including dangling ones) are always blocked — writing through a
    symlink would place the content at a location the user did not choose.
    Directories are blocked because they cannot be opened as regular files. A
    path is also blocked when one of its directories below ``base_dir`` is a
    file or a symlink that does not lead to a directory.

    Args:
        base_dir: The base directory the artifact paths are resolved against.
        request: The user choices that decide which artifacts are written.

    Returns:
        Each requested path holding a symlink, directory, or other non-regular
        file, or lying under a non-directory, in write order.
    """
    requested = [
        (CONFIG_FILENAME, True),
        (DEFAULT_RUNBOOK_PATH, request.runbook),
        (SKILL_RELATIVE_PATH, request.install_skill),
    ]
    return [
        relative for relative, wanted in requested if wanted and _is_blocked(base_dir, relative)
    ]


def _roll_back(
    base_dir: Path, artifacts: list[ScaffoldArtifact], created_directories: list[Path]
) -> None:
    """Remove the files and directories this run created, and nothing else.

    A removal that fails is skipped, so the error that triggered the rollback
    is the one that propagates. That is also what keeps a directory someone
    else has since put a file in: removing a non-empty directory fails.

    Args:
        base_dir: The project root the artifact paths are relative to.
        artifacts: The artifacts handled so far; only those with status
            ``created`` are removed.
        created_directories: The directories this run created, an ancestor
            before its descendants.
    """
    for artifact in artifacts:
        if artifact.status == "created":
            with contextlib.suppress(OSError):
                (base_dir / artifact.path).unlink(missing_ok=True)
    for directory in reversed(created_directories):
        with contextlib.suppress(OSError):
            directory.rmdir()


def scaffold(base_dir: str | Path, request: ScaffoldRequest) -> ScaffoldResult:
    """Write the config, runbook stub, and skill file for ``base_dir``.

    An existing ``gymrat.toml`` is reported as ``exists`` and left
    byte-identical; the remaining artifacts are still created, which makes a
    re-run the way to restore a deleted runbook or skill. A failure leaves no
    partial scaffold behind: the files and the directories this run created
    are removed, except a directory that is no longer empty.

    Args:
        base_dir: The project root to scaffold into.
        request: Which artifacts to create and the bench command to embed in
            the config.

    Returns:
        A :class:`ScaffoldResult` describing each artifact.

    Raises:
        GymratError: When no config exists and ``request.bench`` is None, the
            config fails validation, the bundled skill cannot be read, a
            non-regular file (directory, symlink) occupies an artifact path, a
            file or dangling symlink stands where one of its directories
            belongs, or a filesystem error prevents writing.
    """
    base_dir = Path(base_dir)
    config_path = base_dir / CONFIG_FILENAME
    try:
        config_exists = config_path.exists()
    except OSError as error:
        msg = f"Cannot access {config_path}: {error}"
        raise GymratError(msg, hint="Check directory permissions.") from error

    # An existing config is kept as-is, so its bench command stands in for the flag.
    if not config_exists and request.bench is None:
        msg = "Missing --bench flag."
        raise GymratError(msg)
    config_content = None if config_exists else _prepare_config(request)
    skill_content = read_bundled_skill() if request.install_skill else None

    blocked = _blocked_paths(base_dir, request)
    if blocked:
        paths = ", ".join(blocked)
        msg = f"Blocked path: {paths}"
        raise GymratError(msg, hint="Remove or rename the blocking entry and re-run.")

    artifacts: list[ScaffoldArtifact] = []
    created_directories: list[Path] = []
    try:
        artifacts.append(
            ScaffoldArtifact(path=CONFIG_FILENAME, status="exists")
            if config_content is None
            else _write_artifact(base_dir, CONFIG_FILENAME, config_content, created_directories)
        )
        artifacts.append(
            _write_artifact(base_dir, DEFAULT_RUNBOOK_PATH, _RUNBOOK_STUB, created_directories)
            if request.runbook
            else ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="declined")
        )
        artifacts.append(
            ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="declined")
            if skill_content is None
            else _write_artifact(base_dir, SKILL_RELATIVE_PATH, skill_content, created_directories)
        )
    except BaseException:
        _roll_back(base_dir, artifacts, created_directories)
        raise

    config_artifact, runbook_artifact, skill_artifact = artifacts
    return ScaffoldResult(config=config_artifact, runbook=runbook_artifact, skill=skill_artifact)
