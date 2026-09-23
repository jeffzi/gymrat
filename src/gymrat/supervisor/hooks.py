"""PreToolUse hooks that fence a supervised Claude session.

Two rules are registered, each behind its own matcher:

- Edit, Write, MultiEdit, and NotebookEdit calls go through
  :func:`~gymrat.supervisor.hooks_files.check_file_edit`, which keeps file
  edits inside the experiment worktree.
- Bash calls go through :func:`check_background_gymrat`, which refuses to run
  a gymrat command in the background.

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

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from gymrat.supervisor.hooks_files import check_file_edit

if TYPE_CHECKING:
    from pathlib import Path

    from claude_agent_sdk import HookCallback, HookContext, HookMatcher
    from claude_agent_sdk.types import HookEvent, HookInput, HookJSONOutput

type HooksFactory = Callable[[], dict[HookEvent, list[HookMatcher]]]

_BACKGROUND_REASON = "never background a gymrat command; run it in the foreground"
_REFUSED_REASON = "gymrat could not evaluate this call, so it was refused"

_FILE_MATCHER = "Edit|Write|MultiEdit|NotebookEdit"
_BASH_MATCHER = "Bash"

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


def _decision(reason: str | None) -> HookJSONOutput:
    if reason is None:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _refuse_on_error(evaluate: Callable[[], str | None]) -> HookJSONOutput:
    """Run a rule and turn any exception it raises into a refusal."""
    try:
        reason = evaluate()
    except Exception:  # noqa: BLE001 -- any failure refuses the call rather than letting it through
        reason = _REFUSED_REASON
    return _decision(reason)


def _file_callback(root: Path) -> HookCallback:
    async def callback(
        hook_input: HookInput, _tool_use_id: str | None, _context: HookContext
    ) -> HookJSONOutput:
        return _refuse_on_error(lambda: check_file_edit(hook_input, root))

    return callback


async def _bash_callback(
    hook_input: HookInput, _tool_use_id: str | None, _context: HookContext
) -> HookJSONOutput:
    return _refuse_on_error(lambda: check_background_gymrat(hook_input))


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

        return {
            "PreToolUse": [
                HookMatcher(matcher=_FILE_MATCHER, hooks=[_file_callback(root)]),
                HookMatcher(matcher=_BASH_MATCHER, hooks=[_bash_callback]),
            ]
        }

    return factory
