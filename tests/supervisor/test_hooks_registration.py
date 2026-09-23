"""Tests for the PreToolUse hooks registered on a supervised Claude session."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from claude_agent_sdk import HookContext, PreToolUseHookInput

from gymrat.session.paths import experiment_worktree_dir
from gymrat.supervisor import hooks
from gymrat.supervisor.hooks import check_background_gymrat, supervise_hooks_factory

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


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A repository root holding an experiment worktree."""
    repo = tmp_path / "repo"
    Path(experiment_worktree_dir(str(repo))).mkdir(parents=True)
    return repo


@pytest.fixture
def worktree(root: Path) -> Path:
    """The experiment worktree directory of the test repository."""
    return Path(experiment_worktree_dir(str(root)))


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
        "ls mygymrat",  # cspell:disable-line
        "tail gymrat_log",
        "echo gymrat2",
        "ls x_gymrat",
        "ls 2gymrat",
        "ls agymrat",  # cspell:disable-line
        "ls gymrats",  # cspell:disable-line
        "ls gymrat9",
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
    "extra",
    [
        pytest.param({"run_in_background": False}, id="false"),
        pytest.param({}, id="missing"),
        pytest.param({"run_in_background": "true"}, id="string-true"),
        pytest.param({"run_in_background": 1}, id="int-one"),
    ],
)
def test_check_background_gymrat_when_not_boolean_true_background_does_allow(
    extra: dict[str, object],
):
    hook_input = _bash("gymrat keep -m x", **extra)

    assert check_background_gymrat(hook_input) is None


@pytest.mark.parametrize(
    "hook_input",
    [
        pytest.param(
            {"tool_name": "Bash", "tool_input": {"run_in_background": True}}, id="missing"
        ),
        pytest.param(_bash(None, run_in_background=True), id="none"),
        pytest.param(_bash(["gymrat", "keep"], run_in_background=True), id="list"),
    ],
)
def test_check_background_gymrat_when_command_not_string_does_allow(
    hook_input: dict[str, object],
):
    assert check_background_gymrat(hook_input) is None


@pytest.mark.parametrize("tool_input", ["gymrat keep -m x", None])
def test_check_background_gymrat_when_tool_input_not_mapping_does_allow(tool_input: object):
    hook_input = {"tool_name": "Bash", "tool_input": tool_input}

    assert check_background_gymrat(hook_input) is None


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "tune the parser\n\nkeeps the fast path"',
        "git commit -m 'use `parse()` over `split()`'",
        "gymrat keep -m x",
    ],
)
def test_check_background_gymrat_when_foreground_does_allow(command: str):
    hook_input = _bash(command, run_in_background=False)

    assert check_background_gymrat(hook_input) is None


# ---------------------------------------------------------------------------
# Hooks mapping
# ---------------------------------------------------------------------------


def test_supervise_hooks_factory_when_built_does_register_edit_and_bash_matchers(root: Path):
    mapping = supervise_hooks_factory(root)()

    assert list(mapping) == ["PreToolUse"]
    assert [matcher.matcher for matcher in mapping["PreToolUse"]] == [
        "Edit|Write|MultiEdit|NotebookEdit",
        "Bash",
    ]


def test_supervise_hooks_factory_when_built_does_use_sdk_hook_matchers(root: Path):
    from claude_agent_sdk import HookMatcher

    mapping = supervise_hooks_factory(root)()

    assert all(isinstance(matcher, HookMatcher) for matcher in mapping["PreToolUse"])


async def test_file_callback_when_write_outside_worktree_does_deny(root: Path):
    path = str(root / "x.py")
    callback = supervise_hooks_factory(root)()["PreToolUse"][0].hooks[0]
    hook_input = _pre_tool_use("Write", {"file_path": path})

    output = await callback(hook_input, "tool-1", _CONTEXT)

    assert output == _deny(f"edits belong in the experiment worktree: {path} is outside it")


async def test_file_callback_when_write_inside_worktree_does_return_empty(
    root: Path, worktree: Path
):
    callback = supervise_hooks_factory(root)()["PreToolUse"][0].hooks[0]
    hook_input = _pre_tool_use("Write", {"file_path": str(worktree / "x.py")})

    output = await callback(hook_input, "tool-1", _CONTEXT)

    assert output == {}


async def test_bash_callback_when_in_background_gymrat_does_deny(root: Path):
    callback = supervise_hooks_factory(root)()["PreToolUse"][1].hooks[0]

    output = await callback(
        _bash_hook_input("gymrat keep -m x", run_in_background=True), "t", _CONTEXT
    )

    assert output == _deny(_BACKGROUND_REASON)


async def test_bash_callback_when_allowed_does_return_empty(root: Path):
    callback = supervise_hooks_factory(root)()["PreToolUse"][1].hooks[0]

    output = await callback(_bash_hook_input("gymrat keep -m x"), "t", _CONTEXT)

    assert output == {}


@pytest.mark.parametrize(
    ("rule_name", "index"),
    [
        pytest.param("check_file_edit", 0, id="file"),
        pytest.param("check_background_gymrat", 1, id="bash"),
    ],
)
async def test_callback_when_rule_raises_does_deny(
    root: Path, monkeypatch: pytest.MonkeyPatch, rule_name: str, index: int
):
    def _explode(*_args: object) -> str | None:
        message = "boom"
        raise RuntimeError(message)

    monkeypatch.setattr(hooks, rule_name, _explode)
    callback = supervise_hooks_factory(root)()["PreToolUse"][index].hooks[0]

    output = await callback(_bash_hook_input("ls"), "t", _CONTEXT)

    assert output == _deny(_REFUSED_REASON)


def test_supervise_hooks_factory_when_building_does_import_sdk(root: Path):
    probe = f"""
import sys
from pathlib import Path
from gymrat.supervisor.hooks import supervise_hooks_factory
factory = supervise_hooks_factory(Path({str(root)!r}))
if 'claude_agent_sdk' in sys.modules:
    print('factory creation imported the SDK', file=sys.stderr)
    sys.exit(1)
factory()
if 'claude_agent_sdk' not in sys.modules:
    print('building the mapping did not import the SDK', file=sys.stderr)
    sys.exit(1)
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_supervise_hooks_factory_when_source_scanned_is_called_only_by_supervise():
    package = Path(hooks.__file__).parent.parent
    callers = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if re.search(r"(?<!def )\bsupervise_hooks_factory\(", path.read_text(encoding="utf-8"))
    }

    assert callers == {"cli/supervise/cmd.py"}
