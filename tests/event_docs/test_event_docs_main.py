"""Entry point error handling: main() renders GymratError like the CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.event_docs import main
from tests._imports import loaded_under, modules_imported_by

# ---------------------------------------------------------------------------
# Entry point isolation
# ---------------------------------------------------------------------------


def test_import_event_docs_when_loaded_does_not_import_the_cli_package():
    loaded = modules_imported_by("gymrat.event_docs")

    assert loaded_under(loaded, "gymrat.cli") == []


# ---------------------------------------------------------------------------
# main — GymratError with hint
# ---------------------------------------------------------------------------


def test_main_when_repo_root_raises_gymrat_error_with_hint_does_write_message_and_hint_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    error = GymratError("not a git repository", hint="run `git init` first")
    with (
        patch("gymrat.event_docs.repo_root", side_effect=error, autospec=True),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == TOOL_FAILURE_EXIT_CODE
    captured = capsys.readouterr()
    assert "not a git repository" in captured.err
    assert "run `git init` first" in captured.err


# ---------------------------------------------------------------------------
# main — GymratError without hint
# ---------------------------------------------------------------------------


def test_main_when_repo_root_raises_gymrat_error_without_hint_does_write_only_message_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    error = GymratError("not a git repository")
    with (
        patch("gymrat.event_docs.repo_root", side_effect=error, autospec=True),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == TOOL_FAILURE_EXIT_CODE
    captured = capsys.readouterr()
    assert "not a git repository" in captured.err
    assert "hint" not in captured.err.lower()


# ---------------------------------------------------------------------------
# main — success
# ---------------------------------------------------------------------------


def test_main_when_repo_root_succeeds_does_print_written_paths_to_stdout(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    fake_paths = [tmp_path / "schemas" / "a.json", tmp_path / "docs" / "b.md"]
    with (
        patch("gymrat.event_docs.repo_root", return_value=str(tmp_path), autospec=True),
        patch("gymrat.event_docs.write_all", return_value=fake_paths, autospec=True),
    ):
        main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert str(fake_paths[0]) in lines[0]
    assert str(fake_paths[1]) in lines[1]
