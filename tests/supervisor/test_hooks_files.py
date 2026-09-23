"""Tests for the rule confining a supervised agent's file edits to the worktree."""

import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.supervisor import hooks_files
from gymrat.supervisor.hooks_files import check_file_edit

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


def test_check_file_edit_when_repo_under_scratch_root_does_deny_main_tree(root: Path):
    scratch = Path(tempfile.gettempdir()).resolve()
    assert root.resolve().is_relative_to(scratch), "the repository must sit under a scratch root"
    path = str(root / "src" / "x.py")

    assert check_file_edit({"tool_name": "Write", "tool_input": {"file_path": path}}, root) == (
        _outside(path)
    )


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


def test_check_file_edit_when_path_under_gettempdir_outside_repo_does_allow(root: Path):
    path = str(Path(tempfile.gettempdir()) / "gymrat-scratch" / "notes.txt")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) is None


@pytest.mark.skipif(not Path("/tmp").is_dir(), reason="/tmp does not exist on this platform")
def test_check_file_edit_when_path_under_slash_tmp_outside_repo_does_allow(root: Path):
    hook_input = {
        "tool_name": "Write",
        "tool_input": {"file_path": "/tmp/gymrat-scratch/notes.txt"},
    }

    assert check_file_edit(hook_input, root) is None


def test_check_file_edit_when_path_outside_repo_and_scratch_does_deny(root: Path):
    path = str(Path(root.anchor) / "banana" / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def test_check_file_edit_when_posix_tmp_missing_does_deny_path_under_it(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    missing_tmp = Path(root.anchor) / "banana"
    monkeypatch.setattr(hooks_files, "_POSIX_TMP", missing_tmp)
    path = str(missing_tmp / "x.py")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def test_check_file_edit_when_windows_and_path_under_temp_env_does_allow(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    temp = Path(root.anchor) / "banana"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("TEMP", str(temp))
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": str(temp / "notes.txt")}}

    assert check_file_edit(hook_input, root) is None


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


def test_check_file_edit_when_windows_and_temp_env_unset_does_deny(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    path = str(Path(root.anchor) / "banana" / "notes.txt")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("TEMP", raising=False)
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


def test_check_file_edit_when_not_windows_and_path_under_temp_env_does_deny(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    temp = Path(root.anchor) / "banana"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("TEMP", str(temp))
    path = str(temp / "notes.txt")
    hook_input = {"tool_name": "Write", "tool_input": {"file_path": path}}

    assert check_file_edit(hook_input, root) == _outside(path)


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
# Unresolvable paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param({}, id="missing"),
        pytest.param({"file_path": 42}, id="not-string"),
        pytest.param({"file_path": ""}, id="empty"),
    ],
)
def test_check_file_edit_when_path_missing_or_empty_does_deny_naming_the_rule(
    root: Path, tool_input: dict[str, object]
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
        monkeypatch.setattr(os.path, "realpath", os.path.abspath)
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x\0.py"}, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert reason == "edits belong in the experiment worktree: src/x\\x00.py cannot be resolved"


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
    ],
)
def test_check_file_edit_when_tool_not_an_editing_tool_does_allow(
    root: Path, tool_name: str, tool_input: dict[str, object]
):
    hook_input = {"tool_name": tool_name, "tool_input": tool_input}

    assert check_file_edit(hook_input, root) is None
