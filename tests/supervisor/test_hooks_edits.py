"""Tests for the rule confining a supervised agent's file edits to the worktree."""

import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.supervisor import hooks
from gymrat.supervisor.hooks import check_file_edit

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
    monkeypatch.setattr(hooks, "_POSIX_TMP", missing_tmp)
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


@pytest.mark.usefixtures("ignores_case")
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
    scratch = Path(tempfile.gettempdir()).resolve()
    assert root.resolve().is_relative_to(scratch), "the repository must sit under a scratch root"
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


@pytest.mark.usefixtures("honors_case")
def test_check_file_edit_when_sibling_dir_differs_by_case_does_treat_it_as_outside_repo(root: Path):
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


@pytest.mark.usefixtures("case_sensitive_stat")
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
        monkeypatch.setattr(os.path, "realpath", os.path.abspath)
    hook_input = {"tool_name": "Edit", "tool_input": {"file_path": "src/x\0.py"}, "cwd": str(root)}

    reason = check_file_edit(hook_input, root)

    assert reason == "edits belong in the experiment worktree: src/x\\x00.py cannot be resolved"


@pytest.mark.parametrize("error", [ValueError("bad path"), OSError("too many links")])
def test_check_file_edit_when_realpath_raises_does_deny_naming_the_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    def failing_realpath(_path: str) -> str:
        raise error

    monkeypatch.setattr(os.path, "realpath", failing_realpath)
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
