"""Tests for the PreToolUse hooks registered on a supervised Claude session.

These cover the rule confining the agent's file edits to the experiment
worktree, the rule refusing a gymrat command run in the background, and the hooks
mapping the factory registers.
"""

import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast, override
from unittest.mock import create_autospec

import pytest
from claude_agent_sdk import (
    HookCallback,
    HookContext,
    HookInput,
    HookMatcher,
    PreToolUseHookInput,
)
from claude_agent_sdk.types import HookEvent

from gymrat.session.paths import experiment_worktree_dir
from gymrat.supervisor.hooks import (
    check_background_gymrat,
    check_file_edit,
    supervise_hooks_factory,
)
from tests._imports import loaded_under, modules_loaded_after
from tests._platform import needs_symlinks


def _verdict(reason: str | None, path: str) -> str:
    """Which branch decided the edit of ``path``: ``allowed``, ``outside`` or ``unresolvable``.

    A reason that names neither branch for ``path`` comes back unchanged, so a
    failing assertion shows the text the rule gave.
    """
    if reason is None:
        return "allowed"
    if path in reason and "outside" in reason:
        return "outside"
    if path in reason and "cannot be resolved" in reason:
        return "unresolvable"
    return reason


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A repository root holding a main-tree file and an experiment worktree."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "x.py").write_text("x = 1\n")
    Path(experiment_worktree_dir(str(repo))).mkdir(parents=True)
    return repo


@pytest.fixture
def worktree(root: Path) -> Path:
    """The experiment worktree directory of the test repository."""
    return Path(experiment_worktree_dir(str(root)))


# ---------------------------------------------------------------------------
# Path key per tool
# ---------------------------------------------------------------------------


def test_check_file_edit_when_path_in_worktree_does_allow(worktree: Path, root: Path):
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": str(worktree / "x.py")}}

    reason = check_file_edit(hook_input, root)

    assert reason is None


def test_check_file_edit_when_notebook_path_outside_and_file_path_inside_does_deny(
    root: Path, worktree: Path
):
    path = str(root / "nb.ipynb")
    hook_input = {
        "tool_name": "NotebookEdit",
        "tool_input": {"notebook_path": path, "file_path": str(worktree / "nb.ipynb")},
    }

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


# ---------------------------------------------------------------------------
# Worktree containment
# ---------------------------------------------------------------------------


def test_check_file_edit_when_path_is_worktree_dir_does_allow(root: Path, worktree: Path):
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(worktree)}}

    reason = check_file_edit(hook_input, root)

    assert reason is None


def test_check_file_edit_when_path_in_a_sibling_sharing_the_worktree_name_prefix_does_deny(
    root: Path,
):
    worktree = Path(experiment_worktree_dir(str(root)))
    path = str(worktree.parent / f"{worktree.name}-old" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


# ---------------------------------------------------------------------------
# Relative paths
# ---------------------------------------------------------------------------


def test_check_file_edit_when_relative_path_and_cwd_in_worktree_does_allow(
    root: Path, worktree: Path
):
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "cwd": str(worktree)}

    reason = check_file_edit(hook_input, root)

    assert reason is None


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="cwd-missing"),
        pytest.param({"cwd": 42}, id="cwd-not-string"),
    ],
)
def test_check_file_edit_when_relative_path_and_no_usable_cwd_does_resolve_against_root(
    root: Path, worktree: Path, extra: dict[str, object]
):
    relative = str(worktree.relative_to(root) / "x.py")
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": relative}, **extra}

    reason = check_file_edit(hook_input, root)

    assert reason is None


# ---------------------------------------------------------------------------
# Scratch roots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            str(Path(tempfile.gettempdir()) / "gymrat-scratch" / "notes.txt"), id="gettempdir"
        ),
        pytest.param(
            "/tmp/gymrat-scratch/notes.txt",
            id="slash-tmp",
            marks=pytest.mark.skipif(
                not Path("/tmp").is_dir(), reason="/tmp does not exist on this platform"
            ),
        ),
    ],
)
def test_check_file_edit_when_path_under_a_scratch_root_outside_repo_does_allow(
    root: Path, path: str
):
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert reason is None


def test_check_file_edit_when_posix_tmp_missing_does_deny_path_under_it(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    posix_tmp = Path("/tmp")
    real_is_dir = Path.is_dir

    def is_dir_without_posix_tmp(self: Path, *args: object, **kwargs: object) -> bool:
        return self != posix_tmp and real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(
        Path, "is_dir", create_autospec(real_is_dir, side_effect=is_dir_without_posix_tmp)
    )
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    path = str(posix_tmp / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


def _set_temp(monkeypatch: pytest.MonkeyPatch, temp: Path) -> None:
    monkeypatch.setenv("TEMP", str(temp))


def _unset_temp(monkeypatch: pytest.MonkeyPatch, _temp: Path) -> None:
    monkeypatch.delenv("TEMP", raising=False)


@pytest.mark.parametrize(
    ("platform", "arrange_temp", "expected"),
    [
        pytest.param("win32", _set_temp, "allowed", id="windows-under-temp-allowed"),
        pytest.param("win32", _unset_temp, "outside", id="windows-temp-unset-denied"),
        pytest.param("linux", _set_temp, "outside", id="not-windows-temp-ignored"),
    ],
)
def test_check_file_edit_when_path_under_the_temp_env_does_trust_it_only_on_windows(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    arrange_temp: Callable[[pytest.MonkeyPatch, Path], None],
    expected: str,
):
    temp = Path(root.anchor) / "banana"
    monkeypatch.setattr(sys, "platform", platform)
    arrange_temp(monkeypatch, temp)
    path = str(temp / "notes.txt")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == expected


@needs_symlinks
def test_check_file_edit_when_windows_and_temp_env_is_symlink_does_allow_its_target(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
):
    target = Path(root.anchor) / "banana"
    link = tmp_path / "temp-link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("TEMP", str(link))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(target / "x.py")}}

    reason = check_file_edit(hook_input, root)

    assert reason is None


# ---------------------------------------------------------------------------
# Symlinks
# ---------------------------------------------------------------------------


@needs_symlinks
def test_check_file_edit_when_worktree_symlink_points_at_main_tree_does_deny(
    root: Path, worktree: Path
):
    link = worktree / "link.py"
    link.symlink_to(root / "src" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(link)}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, str(link)) == "outside"


# ---------------------------------------------------------------------------
# Letter case
# ---------------------------------------------------------------------------


def _upper_root(repo: Path) -> Path:
    return repo.with_name(repo.name.upper())


def _worktree_dir(repo: Path) -> Path:
    return Path(experiment_worktree_dir(str(repo)))


def _worktree_relative(repo: Path) -> Path:
    return _worktree_dir(repo).relative_to(repo)


def _upper_worktree(repo: Path) -> Path:
    return repo / str(_worktree_relative(repo)).upper()


def _fail_stat(
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    fails: Callable[[str], bool],
    error: OSError,
) -> None:
    def patch(name: str) -> None:
        real = getattr(os, name)

        def fake(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if isinstance(path, (str, bytes, os.PathLike)) and fails(os.fsdecode(path)):
                raise error
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, name, create_autospec(real, side_effect=fake))

    for name in names:
        patch(name)


@pytest.fixture
def ignores_case(root: Path) -> None:
    """Skip unless the filesystem under the repository ignores letter case."""
    if not _upper_root(root).exists():
        pytest.skip("needs a filesystem that ignores letter case")


@pytest.fixture
def under_scratch_root(root: Path) -> None:
    """Skip unless the repository sits under the system temp directory, a scratch root."""
    if not root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        pytest.skip("needs the repository under the system temp directory")


@pytest.fixture
def honors_case(root: Path) -> None:
    """Skip unless the filesystem under the repository tells letter cases apart."""
    if _upper_root(root).exists():
        pytest.skip("needs a filesystem that tells letter cases apart")


@pytest.fixture
def case_sensitive_stat(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every upper-case spelling of a repository directory a missing path."""
    if sys.platform == "win32":
        pytest.skip(
            "the Windows path resolver does not go through os.stat or os.lstat, "
            "so patching them does not model a volume that tells letter cases apart"
        )
    upper_names = {root.name.upper(), *Path(str(_worktree_relative(root)).upper()).parts}

    def is_upper_spelling(path: str) -> bool:
        return not upper_names.isdisjoint(Path(path).parts)

    _fail_stat(monkeypatch, ("stat", "lstat"), is_upper_spelling, FileNotFoundError(2, "missing"))


@pytest.mark.usefixtures("ignores_case", "under_scratch_root")
@pytest.mark.parametrize(
    "tail",
    [
        pytest.param("src/x.py", id="existing-file"),
        pytest.param("fresh/note.py", id="missing-tail"),
    ],
)
def test_check_file_edit_when_case_variant_of_main_tree_under_scratch_root_does_deny(
    root: Path, tail: str
):
    path = str(_upper_root(root) / tail)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


@pytest.mark.usefixtures("ignores_case")
def test_check_file_edit_when_case_variant_of_worktree_does_allow(root: Path):
    hook_input = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(_upper_worktree(root) / "x.py")},
    }

    reason = check_file_edit(hook_input, root)

    assert reason is None


@pytest.mark.usefixtures("under_scratch_root", "honors_case")
def test_check_file_edit_when_sibling_dir_differs_by_case_does_allow_as_scratch(root: Path):
    sibling = _upper_root(root)
    (sibling / "src").mkdir(parents=True)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(sibling / "src" / "x.py")}}

    reason = check_file_edit(hook_input, root)

    assert reason is None


@pytest.mark.usefixtures("honors_case")
def test_check_file_edit_when_existing_dir_differs_from_worktree_by_case_does_deny(root: Path):
    lookalike = _upper_worktree(root)
    lookalike.mkdir(parents=True)
    path = str(lookalike / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


@pytest.mark.usefixtures("under_scratch_root", "case_sensitive_stat")
def test_check_file_edit_when_case_matters_and_repo_name_differs_by_case_does_allow_as_scratch(
    root: Path,
):
    path = str(_upper_root(root) / "src" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert reason is None


@pytest.mark.usefixtures("case_sensitive_stat")
def test_check_file_edit_when_case_matters_and_worktree_name_differs_by_case_does_deny(root: Path):
    path = str(_upper_worktree(root) / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


# ---------------------------------------------------------------------------
# Names the operating system rejects
# ---------------------------------------------------------------------------


class _WindowsError(OSError):
    def __init__(self, winerror: int) -> None:
        super().__init__(22, "rejected by the operating system")
        self.winerror = winerror


_ERROR_INVALID_NAME = 123


def _reject_name(monkeypatch: pytest.MonkeyPatch, rejected: Path, error: OSError) -> None:
    def is_under_rejected(path: str) -> bool:
        return path == str(rejected) or path.startswith(f"{rejected}{os.sep}")

    _fail_stat(monkeypatch, ("stat", "lstat"), is_under_rejected, error)


def test_check_file_edit_when_name_rejected_as_invalid_outside_worktree_does_deny_as_outside(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    rejected = Path(os.path.realpath(root)) / "bad.py"
    path = str(rejected)
    _reject_name(monkeypatch, rejected, _WindowsError(_ERROR_INVALID_NAME))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "outside"


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_WindowsError(5), id="access-denied"),
        pytest.param(PermissionError(13, "denied"), id="no-windows-code"),
    ],
)
def test_check_file_edit_when_examining_path_raises_other_error_does_deny_as_unresolvable(
    root: Path, worktree: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
):
    rejected = Path(os.path.realpath(worktree)) / "bad.py"
    path = str(rejected)
    _reject_name(monkeypatch, rejected, error)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, path) == "unresolvable"


# ---------------------------------------------------------------------------
# Unresolvable paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param({}, id="missing"),
        pytest.param({"file_path": 42}, id="not-string"),
        pytest.param({"file_path": ""}, id="empty"),
        pytest.param("src/x.py", id="input-is-a-string"),
    ],
)
def test_check_file_edit_when_path_missing_or_empty_does_deny_naming_the_rule(
    root: Path, tool_input: object
):
    hook_input = {"tool_name": "Edit", "tool_input": tool_input, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert reason is not None
    assert "missing" in reason


def test_check_file_edit_when_path_has_nul_does_deny_naming_the_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    # Windows' non-strict realpath hands a NUL-bearing path back as-is instead
    # of raising, which abspath reproduces on every host. A stat that reads the
    # path as absent, rather than raising, makes it look like a fresh file under
    # the main tree, so only the NUL guard can still call it unresolvable.
    monkeypatch.setattr(
        os.path, "realpath", create_autospec(os.path.realpath, side_effect=os.path.abspath)
    )
    _fail_stat(
        monkeypatch, ("stat", "lstat"), lambda path: "\0" in path, FileNotFoundError(2, "missing")
    )
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x\0.py"}, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, "src/x\\x00.py") == "unresolvable"


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("bad path"), id="value-error"),
        pytest.param(OSError("too many links"), id="os-error"),
    ],
)
def test_check_file_edit_when_realpath_raises_does_deny_naming_the_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    monkeypatch.setattr(os.path, "realpath", create_autospec(os.path.realpath, side_effect=error))
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x.py"}, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, "src/x.py") == "unresolvable"


# ---------------------------------------------------------------------------
# Echoed paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        pytest.param("\n", "\\n", id="newline"),
        pytest.param("\\", "\\", id="backslash"),
    ],
)
def test_check_file_edit_when_path_has_special_characters_does_echo_them_on_one_line(
    root: Path, raw: str, shown: str
):
    prefix = str(root / "a")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": f"{prefix}{raw}b.py"}}

    reason = check_file_edit(hook_input, root)

    assert _verdict(reason, f"{prefix}{shown}b.py") == "outside"


# ---------------------------------------------------------------------------
# Tools the rule does not govern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param("Read", {"file_path": "/banana/x.py"}, id="read"),
        pytest.param(["Edit"], {"file_path": "/banana/x.py"}, id="name-unhashable"),
    ],
)
def test_check_file_edit_when_tool_not_an_editing_tool_does_allow(
    root: Path, tool_name: object, tool_input: dict[str, object]
):
    hook_input = {"tool_name": tool_name, "tool_input": tool_input}

    reason = check_file_edit(hook_input, root)

    assert reason is None


# ---------------------------------------------------------------------------
# PreToolUse hook registration
# ---------------------------------------------------------------------------

_CONTEXT: HookContext = {"signal": None}


def _bash(command: object, **extra: object) -> dict[str, object]:
    return {"tool_name": "Bash", "tool_input": {"command": command, **extra}}


def _pre_tool_use(tool_name: str, tool_input: dict[str, object]) -> PreToolUseHookInput:
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "session",
        "transcript_path": "/transcript",
        "cwd": "/cwd",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": "tool-1",
    }


# ---------------------------------------------------------------------------
# Background gymrat rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "gymrat keep -m x",
        "uv run gymrat status",
        ".venv/bin/gymrat keep",
        "cat gymrat.toml",
    ],
)
def test_check_background_gymrat_when_in_background_gymrat_word_does_deny(command: str):
    hook_input = _bash(command, run_in_background=True)

    reason = check_background_gymrat(hook_input)

    assert reason is not None


@pytest.mark.parametrize(
    "command",
    [
        "cat gymrat-notes.txt",
        "ls my-gymrat",
        "tail gymrat_log",
        "echo gymrat2",
        "ls x_gymrat",
        "ls 2gymrat",
        "ls agymrat",  # cspell:disable-line
        "ls gymrats",  # cspell:disable-line
        "ls Xgymrat",  # cspell:disable-line
        "ls gymratX",  # cspell:disable-line
    ],
)
def test_check_background_gymrat_when_in_background_without_gymrat_word_does_allow(
    command: str,
):
    hook_input = _bash(command, run_in_background=True)

    reason = check_background_gymrat(hook_input)

    assert reason is None


@pytest.mark.parametrize(
    "hook_input",
    [
        pytest.param(_bash("gymrat keep -m x", run_in_background=False), id="background-false"),
        pytest.param(_bash("gymrat keep -m x"), id="background-missing"),
        pytest.param(_bash("gymrat keep -m x", run_in_background="true"), id="background-string"),
        pytest.param(
            {"tool_name": "Bash", "tool_input": {"run_in_background": True}},
            id="command-missing",
        ),
        pytest.param(_bash(["gymrat", "keep"], run_in_background=True), id="command-list"),
        pytest.param(
            {"tool_name": "Bash", "tool_input": "gymrat keep -m x"}, id="tool-input-string"
        ),
    ],
)
def test_check_background_gymrat_when_not_a_background_gymrat_command_does_allow(
    hook_input: dict[str, object],
):
    reason = check_background_gymrat(hook_input)

    assert reason is None


# ---------------------------------------------------------------------------
# Hooks mapping
# ---------------------------------------------------------------------------


def _matchers_for(mapping: dict[HookEvent, list[HookMatcher]], tool_name: str) -> list[HookMatcher]:
    """The ``PreToolUse`` matchers whose pattern routes ``tool_name`` to their hooks.

    Args:
        mapping: The hooks mapping a factory built.
        tool_name: The tool the agent calls.

    Returns:
        Every matcher whose pattern matches the whole tool name.
    """
    return [
        matcher
        for matcher in mapping["PreToolUse"]
        if matcher.matcher is not None and re.fullmatch(matcher.matcher, tool_name)
    ]


def _callback_for(root: Path, tool_name: str) -> HookCallback:
    """The one hook callback the factory's mapping routes ``tool_name`` to.

    Args:
        root: The repository root the factory confines edits to.
        tool_name: The tool the agent calls.

    Returns:
        The first hook of the single matcher that routes the tool.
    """
    (matcher,) = _matchers_for(supervise_hooks_factory(root)(), tool_name)
    return matcher.hooks[0]


def _decision(output: object) -> tuple[object, object]:
    """The ``(permissionDecision, permissionDecisionReason)`` a hook answered with.

    An answer that allows the call carries neither, so both come back ``None``.
    """
    specific = cast("dict[str, object]", output).get("hookSpecificOutput", {})
    fields = cast("dict[str, object]", specific)
    return fields.get("permissionDecision"), fields.get("permissionDecisionReason")


def _write_outside(root: Path, _worktree: Path) -> PreToolUseHookInput:
    return _pre_tool_use("Write", {"file_path": str(root / "x.py")})


def _write_inside(_root: Path, worktree: Path) -> PreToolUseHookInput:
    return _pre_tool_use("Write", {"file_path": str(worktree / "x.py")})


async def test_supervise_hooks_factory_when_a_rule_denies_the_call_does_answer_deny_with_its_reason(
    root: Path,
    worktree: Path,
):
    hook_input = _write_outside(root, worktree)
    callback = _callback_for(root, hook_input["tool_name"])

    output = await callback(hook_input, "tool-1", _CONTEXT)

    decision, reason = _decision(output)
    assert (decision, bool(reason)) == ("deny", True)


async def test_supervise_hooks_factory_when_the_rules_allow_the_call_does_answer_empty(
    root: Path, worktree: Path
):
    hook_input = _write_inside(root, worktree)
    callback = _callback_for(root, hook_input["tool_name"])

    output = await callback(hook_input, "tool-1", _CONTEXT)

    assert output == {}


class _UnreadableInput(dict[str, object]):
    """A hook payload whose every lookup fails, so any rule reading it raises."""

    @override
    def get(self, *_args: object) -> object:
        msg = "boom"
        raise RuntimeError(msg)


async def test_supervise_hooks_factory_when_the_rule_raises_does_deny_naming_the_error(
    root: Path,
):
    callback = _callback_for(root, "Write")

    output = await callback(cast("HookInput", _UnreadableInput()), "t", _CONTEXT)

    decision, reason = _decision(output)
    assert decision == "deny"
    assert "RuntimeError: boom" in str(reason)


def _create_factory_source(root: Path) -> str:
    return (
        "from pathlib import Path\n"
        "from gymrat.supervisor.hooks import supervise_hooks_factory\n"
        f"factory = supervise_hooks_factory(Path({str(root)!r}))"
    )


def test_supervise_hooks_factory_when_created_does_not_import_the_sdk(root: Path):
    source = _create_factory_source(root)

    loaded = modules_loaded_after(source)

    assert loaded_under(loaded, "claude_agent_sdk") == []
