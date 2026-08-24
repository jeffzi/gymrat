"""Tests for the reference-CLI oracle locator, builder, and runner."""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from tools.parity import oracle
from tools.parity.oracle import (
    PINNED_ORACLE_SHA,
    OracleRunner,
    PortRunner,
    RunResult,
    assert_pinned_sha,
    ensure_built,
    ts_repo_path,
)


def _init_git_repo(path: Path) -> str:
    """Create a one-commit git repo at ``path`` and return its HEAD SHA."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)  # noqa: S607
    subprocess.run(
        [  # noqa: S607
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "seed",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# ts_repo_path
# ---------------------------------------------------------------------------


def test_ts_repo_path_when_env_var_set_to_existing_dir_does_return_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GYMRAT_TS_REPO", str(tmp_path))

    assert ts_repo_path().resolve() == tmp_path.resolve()


def test_ts_repo_path_when_env_var_set_to_missing_dir_does_raise_naming_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GYMRAT_TS_REPO", str(tmp_path / "does-not-exist"))

    with pytest.raises(GymratError, match="GYMRAT_TS_REPO"):
        ts_repo_path()


def test_ts_repo_path_when_env_var_unset_does_return_default_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GYMRAT_TS_REPO", raising=False)

    result = ts_repo_path()

    assert result.parts[-2:] == ("ts", "gymrat")


# ---------------------------------------------------------------------------
# assert_pinned_sha
# ---------------------------------------------------------------------------


def test_assert_pinned_sha_when_head_differs_does_raise_naming_both_shas(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)

    with pytest.raises(GymratError) as excinfo:
        assert_pinned_sha(tmp_path)

    message = str(excinfo.value)
    assert head in message
    assert PINNED_ORACLE_SHA in message


def test_assert_pinned_sha_when_head_matches_pin_does_not_raise(requires_oracle: None) -> None:
    assert assert_pinned_sha(ts_repo_path()) is None


# ---------------------------------------------------------------------------
# ensure_built
# ---------------------------------------------------------------------------


def test_ensure_built_when_binary_exists_does_return_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_js = tmp_path / "dist" / "cli.js"
    cli_js.parent.mkdir(parents=True)
    cli_js.write_text("// built")

    def _boom(*args: object, **kwargs: object) -> RunResult:
        msg = "subprocess must not run when the binary already exists"
        raise AssertionError(msg)

    monkeypatch.setattr(oracle, "_run", _boom)

    assert ensure_built(tmp_path) == cli_js


def test_ensure_built_when_binary_absent_does_run_npm_build_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[list[str]] = []
    cli_js = tmp_path / "dist" / "cli.js"

    def _fake_run(cmd: Sequence[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
        recorded.append(list(cmd))
        cli_js.parent.mkdir(parents=True, exist_ok=True)
        cli_js.write_text("// built")
        return RunResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(oracle, "_run", _fake_run)

    result = ensure_built(tmp_path)

    assert result == cli_js
    assert recorded == [
        ["npm", "ci", "--ignore-scripts"],
        ["npm", "run", "build"],
    ]


def test_ensure_built_when_force_true_does_rebuild_despite_existing_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_js = tmp_path / "dist" / "cli.js"
    cli_js.parent.mkdir(parents=True)
    cli_js.write_text("// stale")
    recorded: list[list[str]] = []

    def _fake_run(cmd: Sequence[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
        recorded.append(list(cmd))
        return RunResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(oracle, "_run", _fake_run)

    result = ensure_built(tmp_path, force=True)

    assert result == cli_js
    assert recorded == [
        ["npm", "ci", "--ignore-scripts"],
        ["npm", "run", "build"],
    ]


# ---------------------------------------------------------------------------
# OracleRunner
# ---------------------------------------------------------------------------


def test_oracle_runner_run_when_invoked_with_help_does_return_usage_on_stdout(
    requires_oracle: None, tmp_path: Path
) -> None:
    cli_js = ensure_built(ts_repo_path())
    runner = OracleRunner(cli_js)

    result = runner.run(["compare", "--help"], cwd=tmp_path)

    assert result.exit_code == 0
    assert "Usage: gymrat compare" in result.stdout
    assert "Usage: gymrat compare" not in result.stderr


def test_oracle_runner_run_when_spawning_does_use_clean_color_env_and_node_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    captured: dict[str, object] = {}
    cli_js = tmp_path / "dist" / "cli.js"

    def _fake_run(cmd: Sequence[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env) if env is not None else None
        return RunResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(oracle, "_run", _fake_run)
    runner = OracleRunner(cli_js)

    runner.run(["compare", "--help"], cwd=tmp_path)

    assert captured["cmd"] == ["node", str(cli_js), "compare", "--help"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NO_COLOR"] == "1"
    assert "FORCE_COLOR" not in env


# ---------------------------------------------------------------------------
# PINNED_ORACLE_SHA
# ---------------------------------------------------------------------------


def test_pinned_oracle_sha_is_the_v0_8_reference_commit() -> None:
    assert PINNED_ORACLE_SHA == "17f4ea96bb37587dc536a882868a0791f9ecbbf7"


# ---------------------------------------------------------------------------
# PortRunner
# ---------------------------------------------------------------------------


def test_port_runner_run_when_spawning_does_use_clean_color_env_and_python_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    captured: dict[str, object] = {}

    def _fake_run(cmd: Sequence[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env) if env is not None else None
        return RunResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(oracle, "_run", _fake_run)
    runner = PortRunner()

    runner.run(["compare", "--help"], cwd=tmp_path)

    assert captured["cmd"] == [sys.executable, "-m", "gymrat_py.cli.app", "compare", "--help"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NO_COLOR"] == "1"
    assert "FORCE_COLOR" not in env


def test_port_runner_run_when_invoked_with_help_does_return_usage_on_stdout(
    tmp_path: Path,
) -> None:
    runner = PortRunner()

    result = runner.run(["compare", "--help"], cwd=tmp_path)

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "compare" in result.stdout
