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

from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.supervisor.hooks import (
    check_background_gymrat,
    check_file_edit,
    supervise_hooks_factory,
)
from tests._imports import loaded_under, modules_loaded_after

_needs_symlinks = pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks needs extra privileges on Windows"
)


def _outside(path: str) -> str:
    return f"edits belong in the experiment worktree: {path} is outside it"


_EDITING_TOOLS = [
    pytest.param("Edit", "file_path", id="edit"),
    pytest.param("Write", "file_path", id="write"),
    pytest.param("MultiEdit", "file_path", id="multi-edit"),
    pytest.param("NotebookEdit", "notebook_path", id="notebook-edit"),
]


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


@pytest.mark.parametrize(("tool_name", "path_key"), _EDITING_TOOLS)
def test_check_file_edit_when_path_in_worktree_does_allow(
    root: Path, worktree: Path, tool_name: str, path_key: str
):
    hook_input = {"tool_name": tool_name, "tool_input": {path_key: str(worktree / "x.py")}}

    assert check_file_edit(hook_input, root) is None


@pytest.mark.parametrize(("tool_name", "path_key"), _EDITING_TOOLS)
def test_check_file_edit_when_path_in_main_tree_does_deny(
    root: Path, tool_name: str, path_key: str
):
    path = str(root / "src" / "x.py")
    hook_input = {"tool_name": tool_name, "tool_input": {path_key: path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def test_check_file_edit_when_notebook_path_outside_and_file_path_inside_does_deny(
    root: Path, worktree: Path
):
    path = str(root / "nb.ipynb")
    hook_input = {
        "tool_name": "NotebookEdit",
        "tool_input": {"notebook_path": path, "file_path": str(worktree / "nb.ipynb")},
    }

    assert check_file_edit(hook_input, root) == _outside(path)


# ---------------------------------------------------------------------------
# Worktree containment
# ---------------------------------------------------------------------------


def test_check_file_edit_when_path_is_worktree_dir_does_allow(root: Path, worktree: Path):
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(worktree)}}

    assert check_file_edit(hook_input, root) is None


def test_check_file_edit_when_sibling_dir_shares_worktree_name_prefix_does_deny(
    root: Path, worktree: Path
):
    path = str(worktree.parent / f"{worktree.name}-old" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def _baseline_file(repo: Path) -> Path:
    return Path(baseline_worktree_dir(str(repo))) / "x.py"


def _session_dir_file(repo: Path) -> Path:
    return repo / ".gymrat" / "notes.txt"


@pytest.mark.parametrize(
    "locate",
    [
        pytest.param(_baseline_file, id="baseline-worktree"),
        pytest.param(_session_dir_file, id="session-dir"),
    ],
)
def test_check_file_edit_when_path_elsewhere_under_session_dir_does_deny(
    root: Path, locate: Callable[[Path], Path]
):
    path = str(locate(root))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


# ---------------------------------------------------------------------------
# Relative paths
# ---------------------------------------------------------------------------


def test_check_file_edit_when_relative_path_and_cwd_in_worktree_does_allow(
    root: Path, worktree: Path
):
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "cwd": str(worktree)}

    assert check_file_edit(hook_input, root) is None


def test_check_file_edit_when_relative_path_and_cwd_at_repo_root_does_deny(root: Path):
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x.py"}, "cwd": str(root)}

    assert check_file_edit(hook_input, root) == _outside("src/x.py")


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

    assert check_file_edit(hook_input, root) is None


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

    assert check_file_edit(hook_input, root) is None


def test_check_file_edit_when_path_outside_repo_and_scratch_does_deny(root: Path):
    path = str(Path(root.anchor) / "banana" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def test_check_file_edit_when_posix_tmp_missing_does_deny_path_under_it(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    posix_tmp = Path("/tmp")
    real_is_dir = Path.is_dir

    def is_dir_without_posix_tmp(self: Path, *args: object, **kwargs: object) -> bool:
        return self != posix_tmp and real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir_without_posix_tmp)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    path = str(posix_tmp / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def _allowed(_path: str) -> None:
    return None


def _set_temp(monkeypatch: pytest.MonkeyPatch, temp: Path) -> None:
    monkeypatch.setenv("TEMP", str(temp))


def _unset_temp(monkeypatch: pytest.MonkeyPatch, _temp: Path) -> None:
    monkeypatch.delenv("TEMP", raising=False)


@pytest.mark.parametrize(
    ("platform", "arrange_temp", "expected"),
    [
        pytest.param("win32", _set_temp, _allowed, id="windows-under-temp-allowed"),
        pytest.param("win32", _unset_temp, _outside, id="windows-temp-unset-denied"),
        pytest.param("linux", _set_temp, _outside, id="not-windows-temp-ignored"),
    ],
)
def test_check_file_edit_when_path_under_the_temp_env_does_trust_it_only_on_windows(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    arrange_temp: Callable[[pytest.MonkeyPatch, Path], None],
    expected: Callable[[str], str | None],
):
    temp = Path(root.anchor) / "banana"
    monkeypatch.setattr(sys, "platform", platform)
    arrange_temp(monkeypatch, temp)
    path = str(temp / "notes.txt")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == expected(path)


@_needs_symlinks
def test_check_file_edit_when_windows_and_temp_env_is_symlink_does_allow_its_target(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
):
    target = Path(root.anchor) / "banana"
    link = tmp_path / "temp-link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("TEMP", str(link))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(target / "x.py")}}

    assert check_file_edit(hook_input, root) is None


# ---------------------------------------------------------------------------
# Symlinks
# ---------------------------------------------------------------------------


@_needs_symlinks
def test_check_file_edit_when_worktree_symlink_points_at_main_tree_does_deny(
    root: Path, worktree: Path
):
    link = worktree / "link.py"
    link.symlink_to(root / "src" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(link)}}

    assert check_file_edit(hook_input, root) == _outside(str(link))


def _link_to_main_tree_file(link: Path, repo: Path) -> Path:
    link.symlink_to(repo / "src" / "x.py")
    return link


def _link_to_main_tree_dir(link: Path, repo: Path) -> Path:
    link.symlink_to(repo / "src", target_is_directory=True)
    return link / "fresh.py"


@_needs_symlinks
@pytest.mark.parametrize(
    "plant",
    [
        pytest.param(_link_to_main_tree_file, id="file-link"),
        pytest.param(_link_to_main_tree_dir, id="directory-link-and-missing-file"),
    ],
)
def test_check_file_edit_when_scratch_symlink_points_at_main_tree_does_deny(
    tmp_path: Path, root: Path, plant: Callable[[Path, Path], Path]
):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    path = str(plant(scratch / "link", root))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


@_needs_symlinks
def test_check_file_edit_when_symlinked_spelling_of_worktree_does_allow(root: Path, worktree: Path):
    alias = root / "alias"
    alias.symlink_to(worktree, target_is_directory=True)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(alias / "x.py")}}

    assert check_file_edit(hook_input, root) is None


@_needs_symlinks
def test_check_file_edit_when_root_is_symlinked_spelling_does_allow_real_worktree_path(
    tmp_path: Path, root: Path, worktree: Path
):
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(worktree / "x.py")}}

    assert check_file_edit(hook_input, link) is None


@_needs_symlinks
def test_check_file_edit_when_root_is_symlinked_spelling_does_deny_real_main_tree_path(
    tmp_path: Path, root: Path
):
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    path = str(root / "src" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, link) == _outside(path)


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


def _upper_root_worktree(repo: Path) -> Path:
    return _upper_root(repo) / _worktree_relative(repo)


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

        monkeypatch.setattr(os, name, fake)

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

    assert check_file_edit(hook_input, root) == _outside(path)


@pytest.mark.usefixtures("ignores_case")
@pytest.mark.parametrize(
    "locate",
    [
        pytest.param(_upper_worktree, id="worktree-part"),
        pytest.param(_upper_root_worktree, id="repository-part"),
    ],
)
def test_check_file_edit_when_case_variant_of_worktree_does_allow(
    root: Path, locate: Callable[[Path], Path]
):
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(locate(root) / "x.py")}}

    assert check_file_edit(hook_input, root) is None


@pytest.mark.usefixtures("under_scratch_root", "honors_case")
def test_check_file_edit_when_sibling_dir_differs_by_case_does_allow_as_scratch(root: Path):
    sibling = _upper_root(root)
    (sibling / "src").mkdir(parents=True)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(sibling / "src" / "x.py")}}

    assert check_file_edit(hook_input, root) is None


@pytest.mark.usefixtures("honors_case")
def test_check_file_edit_when_existing_dir_differs_from_worktree_by_case_does_deny(root: Path):
    lookalike = _upper_worktree(root)
    lookalike.mkdir(parents=True)
    path = str(lookalike / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


@pytest.mark.usefixtures("under_scratch_root", "case_sensitive_stat")
def test_check_file_edit_when_case_matters_and_repo_name_differs_by_case_does_allow_as_scratch(
    root: Path,
):
    path = str(_upper_root(root) / "src" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) is None


@pytest.mark.usefixtures("case_sensitive_stat")
def test_check_file_edit_when_case_matters_and_worktree_name_differs_by_case_does_deny(root: Path):
    path = str(_upper_worktree(root) / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def _scratch_dir(repo: Path) -> Path:
    scratch = repo.parent / "scratch"
    scratch.mkdir()
    return scratch


def _repo_root(repo: Path) -> Path:
    return repo


def _temp_dir(_repo: Path) -> Path:
    return Path(tempfile.gettempdir())


@pytest.mark.parametrize(
    "locate",
    [
        pytest.param(_worktree_dir, id="worktree"),
        pytest.param(_scratch_dir, id="scratch-dir"),
        pytest.param(_repo_root, id="repository-root"),
        pytest.param(_temp_dir, id="scratch-root"),
    ],
)
def test_check_file_edit_when_stat_of_existing_ancestor_fails_does_deny(
    root: Path, monkeypatch: pytest.MonkeyPatch, locate: Callable[[Path], Path]
):
    ancestor = os.path.realpath(locate(root))
    path = str(Path(ancestor) / "x.py")
    _fail_stat(monkeypatch, ("stat",), lambda path: path == ancestor, PermissionError(13, "denied"))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert reason == f"edits belong in the experiment worktree: {path} cannot be resolved"


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


_REJECTED_COMPONENTS = [
    pytest.param(("bad.py",), id="file-name"),
    pytest.param(("bad", "x.py"), id="directory-name"),
]


@pytest.mark.parametrize("parts", _REJECTED_COMPONENTS)
def test_check_file_edit_when_name_rejected_as_invalid_outside_worktree_does_deny_as_outside(
    root: Path, monkeypatch: pytest.MonkeyPatch, parts: tuple[str, ...]
):
    rejected = Path(os.path.realpath(root)) / parts[0]
    path = str(rejected.joinpath(*parts[1:]))
    _reject_name(monkeypatch, rejected, _WindowsError(_ERROR_INVALID_NAME))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    reason = check_file_edit(hook_input, root)

    assert reason == _outside(path)


@pytest.mark.parametrize("parts", _REJECTED_COMPONENTS)
def test_check_file_edit_when_name_rejected_as_invalid_inside_worktree_does_allow(
    root: Path, worktree: Path, monkeypatch: pytest.MonkeyPatch, parts: tuple[str, ...]
):
    rejected = Path(os.path.realpath(worktree)) / parts[0]
    path = str(rejected.joinpath(*parts[1:]))
    _reject_name(monkeypatch, rejected, _WindowsError(_ERROR_INVALID_NAME))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) is None


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

    assert reason == f"edits belong in the experiment worktree: {path} cannot be resolved"


# ---------------------------------------------------------------------------
# Unresolvable paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param({}, id="missing"),
        pytest.param({"file_path": 42}, id="not-string"),
        pytest.param({"file_path": ""}, id="empty"),
        pytest.param(None, id="input-not-a-mapping"),
        pytest.param("src/x.py", id="input-is-a-string"),
    ],
)
def test_check_file_edit_when_path_missing_or_empty_does_deny_naming_the_rule(
    root: Path, tool_input: object
):
    hook_input = {"tool_name": "Edit", "tool_input": tool_input, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert reason == "edits belong in the experiment worktree: the edit's path is missing or empty"


@pytest.mark.parametrize(
    "realpath_keeps_nul",
    [
        pytest.param(False, id="host-realpath"),
        # Windows' non-strict realpath hands a NUL-bearing path back as-is
        # instead of raising, which abspath reproduces on every host.
        pytest.param(True, id="realpath-keeps-nul"),
    ],
)
def test_check_file_edit_when_path_has_nul_does_deny_naming_the_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, realpath_keeps_nul: bool
):
    if realpath_keeps_nul:
        monkeypatch.setattr(
            os.path, "realpath", create_autospec(os.path.realpath, side_effect=os.path.abspath)
        )
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x\0.py"}, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert reason == "edits belong in the experiment worktree: src/x\\x00.py cannot be resolved"


@pytest.mark.parametrize("error", [ValueError("bad path"), OSError("too many links")])
def test_check_file_edit_when_realpath_raises_does_deny_naming_the_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    monkeypatch.setattr(os.path, "realpath", create_autospec(os.path.realpath, side_effect=error))
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x.py"}, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert reason == "edits belong in the experiment worktree: src/x.py cannot be resolved"


# ---------------------------------------------------------------------------
# Echoed paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        pytest.param("\n", "\\n", id="newline"),
        pytest.param("\r", "\\r", id="carriage-return"),
        pytest.param("\t", "\\t", id="tab"),
        pytest.param("\x1b", "\\x1b", id="escape"),
        pytest.param("\x85", "\\x85", id="next-line"),
        pytest.param("\u2028", "\\u2028", id="line-separator"),
        pytest.param("\u2029", "\\u2029", id="paragraph-separator"),
        pytest.param("\\", "\\", id="backslash"),
    ],
)
def test_check_file_edit_when_path_has_special_characters_does_echo_them_on_one_line(
    root: Path, raw: str, shown: str
):
    prefix = str(root / "a")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": f"{prefix}{raw}b.py"}}

    reason = check_file_edit(hook_input, root)

    assert reason == _outside(f"{prefix}{shown}b.py")


# ---------------------------------------------------------------------------
# Tools the rule does not govern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param("Read", {"file_path": "/banana/x.py"}, id="read"),
        pytest.param("Bash", {"command": "echo banana > /banana/x.py"}, id="bash"),
        pytest.param(None, {"file_path": "/banana/x.py"}, id="name-missing"),
        pytest.param(42, {"file_path": "/banana/x.py"}, id="name-not-a-string"),
        pytest.param(["Edit"], {"file_path": "/banana/x.py"}, id="name-unhashable"),
    ],
)
def test_check_file_edit_when_tool_not_an_editing_tool_does_allow(
    root: Path, tool_name: object, tool_input: dict[str, object]
):
    hook_input = {"tool_name": tool_name, "tool_input": tool_input}

    assert check_file_edit(hook_input, root) is None


# ---------------------------------------------------------------------------
# PreToolUse hook registration
# ---------------------------------------------------------------------------

_BACKGROUND_REASON = "never background a gymrat command; run it in the foreground"
_REFUSED_REASON = "gymrat could not evaluate this call, so it was refused"
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


def _bash_hook_input(command: object, **extra: object) -> PreToolUseHookInput:
    return _pre_tool_use("Bash", {"command": command, **extra})


def _deny(reason: str) -> dict[str, object]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


# ---------------------------------------------------------------------------
# Background gymrat rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "gymrat keep -m x",
        "uv run gymrat status",
        "cd x && gymrat measure .",
        ".venv/bin/gymrat keep",
        "python -m gymrat iterate",
        "cat gymrat.toml",
        pytest.param("echo 'unbalanced gymrat keep", id="unbalanced-quote"),
    ],
)
def test_check_background_gymrat_when_in_background_gymrat_word_does_deny(command: str):
    hook_input = _bash(command, run_in_background=True)

    assert check_background_gymrat(hook_input) == _BACKGROUND_REASON


@pytest.mark.parametrize(
    "command",
    [
        "sleep 10",
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
        pytest.param("echo 'unbalanced \"quotes", id="unbalanced-quote"),
        pytest.param('git commit -m "tune the parser\n\nkeeps the fast path"', id="multi-line"),
        pytest.param("git commit -m 'use `parse()` over `split()`'", id="backticks"),
    ],
)
def test_check_background_gymrat_when_in_background_without_gymrat_word_does_allow(
    command: str,
):
    hook_input = _bash(command, run_in_background=True)

    assert check_background_gymrat(hook_input) is None


@pytest.mark.parametrize(
    "hook_input",
    [
        pytest.param(_bash("gymrat keep -m x", run_in_background=False), id="background-false"),
        pytest.param(_bash("gymrat keep -m x"), id="background-missing"),
        pytest.param(_bash("gymrat keep -m x", run_in_background="true"), id="background-string"),
        pytest.param(_bash("gymrat keep -m x", run_in_background=1), id="background-int-one"),
        pytest.param(
            {"tool_name": "Bash", "tool_input": {"run_in_background": True}},
            id="command-missing",
        ),
        pytest.param(_bash(None, run_in_background=True), id="command-none"),
        pytest.param(_bash(["gymrat", "keep"], run_in_background=True), id="command-list"),
        pytest.param(
            {"tool_name": "Bash", "tool_input": "gymrat keep -m x"}, id="tool-input-string"
        ),
        pytest.param({"tool_name": "Bash", "tool_input": None}, id="tool-input-none"),
    ],
)
def test_check_background_gymrat_when_not_a_background_gymrat_command_does_allow(
    hook_input: dict[str, object],
):
    assert check_background_gymrat(hook_input) is None


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


def test_supervise_hooks_factory_when_built_does_route_each_guarded_tool_to_one_matcher(
    root: Path,
):
    mapping = supervise_hooks_factory(root)()

    routed = {
        tool: len(_matchers_for(mapping, tool))
        for tool in ("Edit", "Write", "MultiEdit", "NotebookEdit", "Bash", "Read")
    }

    assert list(mapping) == ["PreToolUse"]
    assert routed == {
        "Edit": 1,
        "Write": 1,
        "MultiEdit": 1,
        "NotebookEdit": 1,
        "Bash": 1,
        "Read": 0,
    }


#: The input a case hands the hook its tool routes to, and the answer expected.
_CallbackCase = tuple[PreToolUseHookInput, dict[str, object]]


def _write_outside(root: Path, _worktree: Path) -> _CallbackCase:
    path = str(root / "x.py")
    return (
        _pre_tool_use("Write", {"file_path": path}),
        _deny(f"edits belong in the experiment worktree: {path} is outside it"),
    )


def _write_inside(_root: Path, worktree: Path) -> _CallbackCase:
    return _pre_tool_use("Write", {"file_path": str(worktree / "x.py")}), {}


def _background_gymrat(_root: Path, _worktree: Path) -> _CallbackCase:
    return (
        _bash_hook_input("gymrat keep -m x", run_in_background=True),
        _deny(_BACKGROUND_REASON),
    )


def _foreground_gymrat(_root: Path, _worktree: Path) -> _CallbackCase:
    return _bash_hook_input("gymrat keep -m x"), {}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_write_outside, id="file-write-outside-worktree-denied"),
        pytest.param(_write_inside, id="file-write-inside-worktree-allowed"),
        pytest.param(_background_gymrat, id="bash-background-gymrat-denied"),
        pytest.param(_foreground_gymrat, id="bash-foreground-gymrat-allowed"),
    ],
)
async def test_supervise_hooks_factory_when_a_hook_is_called_does_answer_with_its_rule_decision(
    root: Path,
    worktree: Path,
    case: Callable[[Path, Path], _CallbackCase],
):
    hook_input, expected = case(root, worktree)
    callback = _callback_for(root, hook_input["tool_name"])

    output = await callback(hook_input, "tool-1", _CONTEXT)

    assert output == expected


class _UnreadableInput(dict[str, object]):
    """A hook payload whose every lookup fails, so any rule reading it raises."""

    @override
    def get(self, *_args: object) -> object:
        msg = "boom"
        raise RuntimeError(msg)


@pytest.mark.parametrize(
    "tool_name", [pytest.param("Write", id="file"), pytest.param("Bash", id="bash")]
)
async def test_supervise_hooks_factory_when_the_rule_raises_does_deny(root: Path, tool_name: str):
    callback = _callback_for(root, tool_name)

    output = await callback(cast("HookInput", _UnreadableInput()), "t", _CONTEXT)

    assert output == _deny(_REFUSED_REASON)


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


def test_supervise_hooks_factory_when_built_does_import_the_sdk(root: Path):
    source = f"{_create_factory_source(root)}\nfactory()"

    loaded = modules_loaded_after(source)

    assert loaded_under(loaded, "claude_agent_sdk") != []
