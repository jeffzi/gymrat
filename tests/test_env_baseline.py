"""Tests for the environment baseline every test runs against and the scratch-repo teardown."""

import os
from pathlib import Path

import pytest

from tests._inner_pytest import REPO_ROOT, SINGLE_PROCESS_ARGS, inherited_env, run_inner_pytest
from tests.conftest import BASELINE_ENV_NAMES, BASELINE_ENV_PREFIXES, PINNED_ENV

#: What a developer's shell might export that must never reach a test.
SHELL_EXPORTS = {
    "FORCE_COLOR": "1",
    "NO_COLOR": "1",
    "TERM": "dumb",
    "COLUMNS": "40",
    "LINES": "5",
    "TTY_COMPATIBLE": "1",
    "TTY_INTERACTIVE": "1",
    "EDITOR": "vi",
    "GYMRAT_CONFIG": "/nowhere/gymrat.toml",
    "GYMRAT_COMMAND_ORIGIN": "tool",
    "TRACEPARENT": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    "PY_COLORS": "1",
    "GITHUB_ACTIONS": "true",
    "LANG": "fr_FR.UTF-8",
    "GIT_CONFIG_GLOBAL": "/nowhere/.gitconfig",
}

#: Allowlisted names the shell exports, which tests must see unchanged.
KEPT = {
    "USER": "banana",
    "UV_BANANA": "kept",
    "PYTEST_BANANA": "kept",
    "COVERAGE_BANANA": "kept",
}

#: Names the inner pytest sets for itself, so their values are not the shell's.
PYTEST_OWNED = ("PYTEST_CURRENT_TEST", "PYTEST_XDIST_")

#: Allowlisted names the shell does not export, which must stay absent.
ABSENT = ("CI", "LOGNAME")

#: Exported only inside the xdist worker, so only the worker's own rebuild
#: can remove it.
WORKER_ONLY = "GYMRAT_WORKER_ONLY"

#: Allowlisted and exported only inside the xdist worker, proving the worker
#: received its exports at all.
WORKER_KEPT = {"UV_WORKER_PROBE": "1"}

#: Runs under the project conftest in a child pytest; the tests run in file
#: order, so each one after the first sees what the one before it left behind.
INNER_TESTS = f"""
import json
import os
import subprocess
import sys

import pytest

SHELL_EXPORTS = {SHELL_EXPORTS!r}
ABSENT = {ABSENT!r}
PINNED = {PINNED_ENV!r}
HIDDEN = (set(SHELL_EXPORTS) - set(PINNED)) | {{{WORKER_ONLY!r}}}
SEEN_AT_TEARDOWN = {{}}


def _current():
    return {{k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}}


IMPORT_ENV = _current()


@pytest.fixture(autouse=True)
def observe_teardown(request):
    '''Record FORCE_COLOR as each test's teardown sees it.'''
    yield
    SEEN_AT_TEARDOWN[request.node.name] = os.environ.get("FORCE_COLOR")


@pytest.fixture
def patched(monkeypatch):
    '''Patch GYMRAT_PATCHED, then record GYMRAT_DIRECT as teardown sees it.'''
    monkeypatch.setenv("GYMRAT_PATCHED", "1")
    yield
    SEEN_AT_TEARDOWN["patched"] = os.environ.get("GYMRAT_DIRECT")


def test_shell_exports_are_absent_at_import():
    assert not HIDDEN & set(IMPORT_ENV)


def test_allowlisted_names_keep_shell_values_or_stay_absent():
    assert {{name: os.environ.get(name) for name in SHELL_KEPT}} == SHELL_KEPT
    assert not set(ABSENT) & set(os.environ)


def test_pinned_names_hold_pinned_values():
    assert {{name: os.environ.get(name) for name in PINNED}} == PINNED


def test_child_inherits_no_shell_export():
    script = "import json, os; print(json.dumps(sorted(os.environ)))"
    output = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    ).stdout

    assert not HIDDEN & set(json.loads(output))


def test_writes_environ_directly():
    os.environ["GYMRAT_LEAK"] = "direct"
    os.environ["LANG"] = "de_DE.UTF-8"
    del os.environ["USER"]


def test_direct_writes_are_undone():
    assert _current() == IMPORT_ENV


def test_monkeypatch_wins_during_test(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("LANG")


def test_writes_environ_under_monkeypatch_fixture(patched):
    os.environ["GYMRAT_DIRECT"] = "set"


def test_monkeypatch_undo_runs_before_module_cleanups_and_restore():
    assert SEEN_AT_TEARDOWN["test_monkeypatch_wins_during_test"] is None
    assert SEEN_AT_TEARDOWN["patched"] == "set"
    assert _current() == IMPORT_ENV
"""


def _is_allowed(name: str) -> bool:
    return (
        name in BASELINE_ENV_NAMES or name in PINNED_ENV or name.startswith(BASELINE_ENV_PREFIXES)
    )


@pytest.mark.parametrize(
    ("worker_args", "worker_kept"),
    [
        pytest.param(SINGLE_PROCESS_ARGS, {}, id="single-process"),
        pytest.param(
            [
                "--dist",
                "load",
                "--tx",
                "popen"
                + "".join(f"//env:{name}={value}" for name, value in WORKER_KEPT.items())
                + f"//env:{WORKER_ONLY}=1",
            ],
            WORKER_KEPT,
            id="xdist-worker",
        ),
    ],
)
def test_baseline_when_shell_exports_variables_does_isolate_each_test_from_them(
    tmp_path: Path, worker_args: list[str], worker_kept: dict[str, str]
):
    inherited = {name: value for name, value in os.environ.items() if name not in ABSENT}
    env = {
        **inherited,
        **SHELL_EXPORTS,
        **KEPT,
        "PYTHONPATH": str(REPO_ROOT),
    }
    shell_kept = {
        name: value
        for name, value in env.items()
        if _is_allowed(name) and name not in PINNED_ENV and not name.startswith(PYTEST_OWNED)
    } | worker_kept
    inner = f"{INNER_TESTS}\nSHELL_KEPT = {shell_kept!r}\n"

    result = run_inner_pytest(tmp_path, inner, env, worker_args)

    assert (result.returncode, "9 passed" in result.stdout) == (0, True), result.stdout


#: Runs under the project conftest in a child pytest: deletes one scratch repo
#: and strands a linked worktree of another, leaving both to the teardown. Git
#: writes its object files read-only, which Windows refuses to unlink until the
#: read-only attribute is cleared.
DELETED_REPO_INNER_TEST = """
import os
import shutil
import stat

from tests._git import run_git

WORKTREE = {worktree!r}


def _clear_read_only_and_retry(remove, path, _error):
    os.chmod(path, stat.S_IWRITE)
    remove(path)


def test_strands_worktree_after_deleting_a_repo(create_scratch_repo):
    deleted = create_scratch_repo()
    kept = create_scratch_repo()
    run_git(["worktree", "add", "--detach", WORKTREE], kept)

    shutil.rmtree(deleted, onexc=_clear_read_only_and_retry)
"""


def test_create_scratch_repo_when_a_repo_directory_is_deleted_does_still_remove_stranded_worktrees(
    tmp_path: Path,
):
    worktree = tmp_path / "worktree"
    inner = DELETED_REPO_INNER_TEST.format(worktree=str(worktree))

    result = run_inner_pytest(tmp_path, inner, inherited_env(), SINGLE_PROCESS_ARGS)

    assert (result.returncode, worktree.exists()) == (0, False), result.stdout
