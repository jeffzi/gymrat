"""Confine a supervised agent's file edits to the experiment worktree.

:func:`check_file_edit` inspects one Claude Code ``PreToolUse`` payload for an
editing tool (Edit, Write, MultiEdit, NotebookEdit) and decides whether the
edited path may be written. The path is resolved through symlinks, relative
paths against the payload's ``cwd`` (or the repository root when it has none),
and then checked in order:

1. Under the experiment worktree — allowed.
2. Anywhere else in the repository, including the main tree, the baseline
   worktree, and the rest of the session directory — denied.
3. Under a scratch root (the system temp directory, ``/tmp``, or ``%TEMP%``
   on Windows) — allowed.
4. Anywhere else — denied.

The repository check precedes the scratch check because a repository may itself
live under a scratch root. Tools the rule does not know, such as Read or Bash,
are always allowed.
"""

import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from gymrat.session.paths import experiment_worktree_dir
from gymrat.supervisor.events import FILE_PATH_TOOLS

_RULE = "edits belong in the experiment worktree"

# Read shares the path-key map but must never be restricted by this rule.
_EDITING_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

# Differs from tempfile.gettempdir() on macOS, where $TMPDIR is per-user.
_POSIX_TMP = Path("/tmp")  # noqa: S108 -- only recognized as a scratch root, never written to


def check_file_edit(hook_input: Mapping[str, object], root: Path) -> str | None:
    """Decide whether a file-editing tool call may write its target path.

    Args:
        hook_input: The ``PreToolUse`` payload, carrying ``tool_name``,
            ``tool_input``, and optionally ``cwd``.
        root: The repository root the agent is supervised in.

    Returns:
        A one-line denial reason, or ``None`` when the call is allowed.
    """
    tool_name = hook_input.get("tool_name")
    if not isinstance(tool_name, str) or tool_name not in _EDITING_TOOLS:
        return None
    tool_input = hook_input.get("tool_input")
    raw = tool_input.get(FILE_PATH_TOOLS[tool_name]) if isinstance(tool_input, Mapping) else None
    if not isinstance(raw, str) or not raw:
        return f"{_RULE}: the edit's path is missing or empty"
    cwd = hook_input.get("cwd")
    base = Path(cwd) if isinstance(cwd, str) else root
    candidate = str(base / raw)
    # No filesystem accepts a NUL, but realpath disagrees across platforms:
    # POSIX raises ValueError, while Windows' non-strict mode returns the path
    # unresolved, which would then be judged by where it merely appears to be.
    if "\0" in candidate:
        return f"{_RULE}: {_printable(raw)} cannot be resolved"
    try:
        resolved = os.path.realpath(candidate)
    except (ValueError, OSError):
        return f"{_RULE}: {_printable(raw)} cannot be resolved"
    if _may_edit(resolved, root):
        return None
    return f"{_RULE}: {_printable(raw)} is outside it"


def _printable(path: str) -> str:
    # A denial reason must stay on one line; escaping only non-printable
    # characters leaves Windows backslashes readable as written.
    return "".join(char if char.isprintable() else repr(char)[1:-1] for char in path)


def _may_edit(resolved: str, root: Path) -> bool:
    # The repository check precedes the scratch check: a repository under a
    # scratch root must still confine edits to its experiment worktree.
    if _is_within(resolved, os.path.realpath(experiment_worktree_dir(str(root)))):
        return True
    if _is_within(resolved, os.path.realpath(root)):
        return False
    return any(_is_within(resolved, scratch) for scratch in _scratch_roots())


def _is_within(path: str, directory: str) -> bool:
    return Path(os.path.normcase(path)).is_relative_to(os.path.normcase(directory))


def _scratch_roots() -> list[str]:
    roots = [os.path.realpath(tempfile.gettempdir())]
    if _POSIX_TMP.is_dir():
        roots.append(os.path.realpath(_POSIX_TMP))
    windows_temp = os.environ.get("TEMP")
    if sys.platform == "win32" and windows_temp:
        roots.append(os.path.realpath(windows_temp))
    return roots
