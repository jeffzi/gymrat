"""Tests for the environment baseline every test runs against."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ALLOWED_NAMES = {
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "CI",
    "SYSTEMROOT",
    "USERPROFILE",
    "PATHEXT",
    "COMSPEC",
}
ALLOWED_PREFIXES = ("UV_", "PYTEST_", "COVERAGE_")
PINNED = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}

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
PINNED = {PINNED!r}
HIDDEN = (set(SHELL_EXPORTS) - set(PINNED)) | {{{WORKER_ONLY!r}}}
SEEN_AT_TEARDOWN = {{}}


def _current():
    return {{k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}}


IMPORT_ENV = _current()


@pytest.fixture(autouse=True)
def observe_teardown(request):
    yield
    SEEN_AT_TEARDOWN[request.node.name] = os.environ.get("FORCE_COLOR")


@pytest.fixture
def patched(monkeypatch):
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

    assert os.environ["FORCE_COLOR"] == "1"
    assert "LANG" not in os.environ


def test_monkeypatch_changes_are_undone():
    assert _current() == IMPORT_ENV


def test_writes_environ_under_monkeypatch_fixture(patched):
    os.environ["GYMRAT_DIRECT"] = "set"


def test_forces_typer_terminal_under_force_color(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", True)


def test_typer_terminal_detection_ignores_shell_and_earlier_tests():
    import typer.rich_utils

    assert typer.rich_utils.FORCE_TERMINAL is None


def test_git_tag_ignores_global_gpg_sign(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=a", "-c", "user.email=a@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init")
    git("commit", "--allow-empty", "-m", "initial")
    git("tag", "v1")


def test_monkeypatch_undo_runs_before_module_cleanups_and_restore():
    assert SEEN_AT_TEARDOWN["test_monkeypatch_wins_during_test"] is None
    assert SEEN_AT_TEARDOWN["patched"] == "set"
    assert _current() == IMPORT_ENV
"""


def _is_allowed(name: str) -> bool:
    return name in ALLOWED_NAMES or name in PINNED or name.startswith(ALLOWED_PREFIXES)


def test_environ_when_test_starts_does_hold_only_allowlisted_and_pinned_names():
    unexpected = sorted(name for name in os.environ if not _is_allowed(name))
    pinned = {name: os.environ.get(name) for name in PINNED}

    assert (unexpected, pinned) == ([], PINNED)


@pytest.mark.parametrize(
    ("worker_args", "worker_kept"),
    [
        pytest.param(["-p", "no:xdist"], {}, id="single-process"),
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
def test_baseline_when_shell_exports_variables_does_hide_them_and_restore_after_each_test(
    tmp_path: Path, worker_args: list[str], worker_kept: dict[str, str]
):
    repo_root = Path(__file__).resolve().parent.parent
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[tag]\n\tgpgSign = true\n", encoding="utf-8")
    inherited = {name: value for name, value in os.environ.items() if name not in ABSENT}
    env = {
        **inherited,
        **SHELL_EXPORTS,
        **KEPT,
        "GIT_CONFIG_GLOBAL": str(global_config),
        "PYTHONPATH": str(repo_root),
    }
    shell_kept = {
        name: value
        for name, value in env.items()
        if _is_allowed(name) and name not in PINNED and not name.startswith(PYTEST_OWNED)
    } | worker_kept
    inner = f"{INNER_TESTS}\nSHELL_KEPT = {shell_kept!r}\n"
    (tmp_path / "test_inner.py").write_text(inner, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.devnull,
        "--rootdir",
        str(tmp_path),
        "-p",
        "tests.conftest",
        "--import-mode=importlib",
        "-p",
        "no:randomly",
        *worker_args,
        "-q",
        str(tmp_path / "test_inner.py"),
    ]

    result = subprocess.run(  # noqa: S603
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )

    assert (result.returncode, "13 passed" in result.stdout) == (0, True), result.stdout
