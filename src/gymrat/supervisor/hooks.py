"""PreToolUse hooks that fence a supervised Claude session.

Two rules are registered, each behind its own matcher:

- Edit, Write, MultiEdit, and NotebookEdit calls go through
  :func:`check_file_edit`, which keeps file edits inside the experiment
  worktree.
- Bash calls go through :func:`check_background_gymrat`, which refuses to run
  a gymrat command in the background.

:func:`check_file_edit` resolves the edited path through symlinks, relative
paths against the payload's ``cwd`` (or the repository root when it has none),
and then checks it in order:

1. Under the experiment worktree — allowed.
2. Anywhere else in the repository, including the main tree, the baseline
   worktree, and the rest of the session directory — denied.
3. Under a scratch root (the system temp directory, ``/tmp``, or ``%TEMP%``
   on Windows) — allowed.
4. Anywhere else — denied.

The repository check precedes the scratch check because a repository may itself
live under a scratch root. Each check recognizes its directory by file identity
(device and inode) among the path's existing ancestors, not by spelling, so a
differently cased name is the same directory exactly when the filesystem says
it is. A path whose ancestors cannot be examined is denied. Tools
:func:`check_file_edit` does not know, such as Read, are always allowed.

What the hooks do not do:

- They read no shell command beyond looking for the ``gymrat`` word. The
  command string is never tokenized or parsed, so any text, valid shell or
  not, is judged by that word alone.
- A Bash command that writes outside the worktree is not seen; only the
  dedicated editing tools are confined.

Hooks fire for subagent tool calls too, and nothing treats those differently.
The rules stay synchronous and free of the SDK; the callbacks are async
adapters around them, and the SDK is imported only when the mapping is built.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.session.paths import experiment_worktree_dir
from gymrat.supervisor.events import FILE_PATH_TOOLS

if TYPE_CHECKING:
    from claude_agent_sdk import HookCallback, HookContext, HookMatcher
    from claude_agent_sdk.types import HookEvent, HookInput, HookJSONOutput

type HooksFactory = Callable[[], dict[HookEvent, list[HookMatcher]]]


# ---------------------------------------------------------------------------
# File-edit rule
# ---------------------------------------------------------------------------

_FILE_EDIT_RULE = "edits belong in the experiment worktree"

# Read shares the path-key map but must never be restricted by this rule. The SDK file
# matcher is built from this tuple, so the tools the rule checks and the tools routed to it
# cannot drift apart.
_EDITING_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")

# Differs from tempfile.gettempdir() on macOS, where $TMPDIR is per-user.
_POSIX_TMP = Path("/tmp")  # noqa: S108 -- only recognized as a scratch root, never written to

# The Windows error code for a file name the system rejects, such as one holding a tab.
_ERROR_INVALID_NAME = 123


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
        return f"{_FILE_EDIT_RULE}: the edit's path is missing or empty"
    cwd = hook_input.get("cwd")
    base = Path(cwd) if isinstance(cwd, str) else root
    candidate = str(base / raw)
    # No filesystem accepts a NUL, but realpath disagrees across platforms:
    # POSIX raises ValueError, while Windows' non-strict mode returns the path
    # unresolved, which would then be judged by where it merely appears to be.
    if "\0" in candidate:
        return f"{_FILE_EDIT_RULE}: {_printable(raw)} cannot be resolved"
    try:
        allowed = _may_edit(os.path.realpath(candidate), root)
    except (ValueError, OSError):
        return f"{_FILE_EDIT_RULE}: {_printable(raw)} cannot be resolved"
    if allowed:
        return None
    return f"{_FILE_EDIT_RULE}: {_printable(raw)} is outside it"


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
    """Tell whether a resolved path is a directory or lies under it, by file identity.

    Spelling cannot decide this: a filesystem that ignores letter case gives
    one directory many spellings that symlink resolution leaves alone, and one
    that honors it keeps differently cased names apart. So the directory is
    recognized by its device and inode among the path's existing ancestors.
    Only the part of the directory that does not exist yet is matched by name.

    Args:
        path: The symlink-resolved path being judged.
        directory: The symlink-resolved directory it may lie under.

    Returns:
        Whether the path is the directory or lies under it.

    Raises:
        OSError: An existing ancestor of either path could not be examined.
    """
    target = Path(directory)
    candidate = Path(path)
    for anchor in (target, *target.parents):
        identity = _stat_if_present(anchor)
        if identity is None:
            continue
        missing = _folded_parts(target.relative_to(anchor))
        for ancestor in (candidate, *candidate.parents):
            found = _stat_if_present(ancestor)
            if found is not None and os.path.samestat(found, identity):
                below = _folded_parts(candidate.relative_to(ancestor))
                return below[: len(missing)] == missing
        return False
    # Nothing of the directory exists, not even its drive, so there is no
    # identity to compare and its name is all that is left.
    return Path(os.path.normcase(path)).is_relative_to(os.path.normcase(directory))


def _stat_if_present(path: Path) -> os.stat_result | None:
    # A name Windows rejects as invalid can never exist, so it is as absent as
    # a missing one. Every other OSError propagates: a path that cannot be
    # examined must be refused, never judged as if it were absent.
    try:
        return path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as error:
        if getattr(error, "winerror", None) == _ERROR_INVALID_NAME:
            return None
        raise


def _folded_parts(relative: Path) -> tuple[str, ...]:
    return tuple(os.path.normcase(part) for part in relative.parts)


def _scratch_roots() -> list[str]:
    roots = [os.path.realpath(tempfile.gettempdir())]
    if _POSIX_TMP.is_dir():
        roots.append(os.path.realpath(_POSIX_TMP))
    windows_temp = os.environ.get("TEMP")
    if sys.platform == "win32" and windows_temp:
        roots.append(os.path.realpath(windows_temp))
    return roots


# ---------------------------------------------------------------------------
# Background-command rule
# ---------------------------------------------------------------------------

_BACKGROUND_REASON = "never background a gymrat command; run it in the foreground"

# `\b` would treat `-` as a boundary and flag names like `my-gymrat`.
_GYMRAT_WORD = re.compile(r"(?<![A-Za-z0-9_-])gymrat(?![A-Za-z0-9_-])")


def check_background_gymrat(hook_input: Mapping[str, object]) -> str | None:
    """Decide whether a Bash call may run in the background.

    Only a ``run_in_background`` value of boolean ``True`` counts. The command
    is searched for ``gymrat`` as a whole word (no letter, digit, underscore,
    or hyphen on either side) and is otherwise never inspected.

    Args:
        hook_input: The ``PreToolUse`` payload, carrying ``tool_input``.

    Returns:
        A one-line denial reason, or ``None`` when the call is allowed.
    """
    tool_input = hook_input.get("tool_input")
    if not isinstance(tool_input, Mapping):
        return None
    if tool_input.get("run_in_background") is not True:
        return None
    command = tool_input.get("command")
    if isinstance(command, str) and _GYMRAT_WORD.search(command):
        return _BACKGROUND_REASON
    return None


# ---------------------------------------------------------------------------
# SDK hook adapters
# ---------------------------------------------------------------------------

_REFUSED_REASON = "gymrat could not evaluate this call, so it was refused"

_FILE_MATCHER = "|".join(_EDITING_TOOLS)


def _deny_on(rule: Callable[[HookInput], str | None]) -> HookCallback:
    """Adapt a rule to an SDK callback that denies on its reason, and when it raises."""

    async def callback(
        hook_input: HookInput, _tool_use_id: str | None, _context: HookContext
    ) -> HookJSONOutput:
        try:
            reason = rule(hook_input)
        except Exception:  # noqa: BLE001 -- any failure refuses the call rather than letting it through
            reason = _REFUSED_REASON
        if reason is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return callback


def supervise_hooks_factory(root: Path) -> HooksFactory:
    """Return a factory that builds the session's PreToolUse hooks on demand.

    The ``claude_agent_sdk`` import happens when the returned callable runs, so
    importing this module never loads the SDK.

    Args:
        root: The repository root; file edits are confined to its experiment
            worktree.

    Returns:
        A zero-argument callable returning the hooks mapping, keyed by
        ``"PreToolUse"``, with one matcher for the editing tools and one for
        Bash.
    """

    def factory() -> dict[HookEvent, list[HookMatcher]]:
        from claude_agent_sdk import (  # noqa: PLC0415 -- deferred to avoid import-time SDK load
            HookMatcher,
        )

        file_rule = _deny_on(lambda hook_input: check_file_edit(hook_input, root))
        return {
            "PreToolUse": [
                HookMatcher(matcher=_FILE_MATCHER, hooks=[file_rule]),
                HookMatcher(matcher="Bash", hooks=[_deny_on(check_background_gymrat)]),
            ]
        }

    return factory
