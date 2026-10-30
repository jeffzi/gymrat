"""Behavioral tests for starting or resuming a session (``start_session``).

Every test drives the real ``start_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the
worktree checkouts, resume recreation, and finalize archive-and-recreate only
reveal their behavior against real worktrees, and the assertions read commit SHAs
straight out of the worktrees git laid down.
"""

import itertools
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.config import HooksConfig, ResolvedConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.start import StartResult, start_session
from gymrat.session.paths import (
    archived_session_path,
    baseline_worktree_dir,
    experiment_worktree_dir,
    session_jsonl_path,
)
from gymrat.session.records import SessionConfig, SessionHooks
from gymrat.session.store import fold_session, read_records
from gymrat.session.workspace import BaselineRef, remove_worktrees
from gymrat.utils import now_ns
from tests._config import resolved_config
from tests._git import (
    commit_all,
    create_in_place_target_dir,
    head_of,
    install_git_hook,
    kill_git_during_worktree_add,
    list_worktree_dirs,
    session_branches,
)
from tests._mode_bits import needs_mode_bits
from tests._platform import needs_posix_kill
from tests.loop._settle import (
    commit_and_keep,
    keep_iteration,
)
from tests.session.records._fixtures import (
    append_records,
    finalize_record,
    log_records,
    session_header_of,
)

# The directory a hooked checkout leaves in a worktree, closed to deletion.
PINNED_DIR = "pinned"

HOOKS = HooksConfig(before="npm run warm-cache", after="npm run cool-down")

# A settled run config carrying both the keys the session header snapshots and
# keys it must leave out (``unstable_noise_pct``, ``stop``).
CONFIG_WITHOUT_HOOKS = resolved_config(
    prepare="npm run build",
    filter="npm run bench -- {names}",
    stop=StopConfig(max_iterations=20),
)

CONFIG = replace(CONFIG_WITHOUT_HOOKS, hooks=HOOKS)

# The subset of ``CONFIG_WITHOUT_HOOKS`` the header keeps as provenance.
CONFIG_SNAPSHOT_WITHOUT_HOOKS = SessionConfig(
    bench="npm run bench",
    adapter="metric-lines",
    samples=10,
    timeout_seconds=1800,
    primary="geomean",
    prepare="npm run build",
    filter="npm run bench -- {names}",
)

# The provenance the header keeps once hooks are configured.
CONFIG_SNAPSHOT = CONFIG_SNAPSHOT_WITHOUT_HOOKS.model_copy(
    update={"hooks": SessionHooks(before="npm run warm-cache", after="npm run cool-down")}
)


def _close_session_with_one_keep(root: str) -> str:
    """Keep one commit on the open session and close it.

    Mirrors what a real finalize leaves behind — worktrees off disk and a finalize
    record ending the log — by committing a keep, taking the worktrees down, and
    appending a finalize record, so the next ``start_session`` meets a settled,
    closed session.

    Args:
        root: The repository whose open session is closed.

    Returns:
        The id of the session it closed.
    """
    header = session_header_of(root)
    commit_and_keep(root, 1, "cache the regex")
    remove_worktrees(root, header.worktrees)
    append_records(root, finalize_record())
    return header.session_id


def _pin_worktree_contents(repo_dir: str) -> None:
    """Install a post-checkout hook that leaves each new worktree a directory nothing can empty."""
    install_git_hook(
        repo_dir,
        "post-checkout",
        f"mkdir {PINNED_DIR} && : > {PINNED_DIR}/file && chmod 500 {PINNED_DIR}\n",
    )


@pytest.fixture
def read_only_log_dir(repo: str) -> Iterator[Path]:
    """The session log's directory, refusing new files while worktrees still fit under it."""
    log_dir = Path(session_jsonl_path(repo)).parent
    # The worktrees' parent exists up front, so only the log itself cannot be created.
    Path(experiment_worktree_dir(repo)).parent.mkdir(parents=True)
    log_dir.chmod(0o500)
    yield log_dir

    log_dir.chmod(0o700)
    for worktree in (experiment_worktree_dir(repo), baseline_worktree_dir(repo)):
        pinned = Path(worktree) / PINNED_DIR
        if pinned.exists():
            pinned.chmod(0o700)


# ---------------------------------------------------------------------------
# when the repository holds no session yet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "snapshot"),
    [
        pytest.param(CONFIG, CONFIG_SNAPSHOT, id="hooks-configured"),
        pytest.param(CONFIG_WITHOUT_HOOKS, CONFIG_SNAPSHOT_WITHOUT_HOOKS, id="no-hooks"),
    ],
)
def test_start_session_when_no_session_yet_does_write_header_naming_baseline_and_config(
    repo: str, repo_head: str, config: ResolvedConfig, snapshot: SessionConfig
):
    result = start_session(repo, "main", config)

    header = session_header_of(repo)
    assert result == StartResult(session=header, state=fold_session([header]), resumed=False)
    assert log_records(repo) == [header]
    assert header.baseline == BaselineRef(ref="main", sha=repo_head)
    assert header.config == snapshot


def test_start_session_when_new_does_mint_the_session_id_from_the_instant_the_header_stamps(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # 2024-03-05T06:07:08.9Z in epoch nanoseconds, then a day later on every further read.
    instant_ns = 1_709_618_828_900_000_000
    day_ns = 86_400_000_000_000
    reads = itertools.count(instant_ns, day_ns)
    monkeypatch.setattr("gymrat.utils.now_ns", create_autospec(now_ns, side_effect=reads))

    start_session(repo, "main", CONFIG)

    header = session_header_of(repo)
    assert header.at == instant_ns
    assert header.session_id.startswith("20240305-060708-")


def test_start_session_when_no_ref_given_does_pin_the_baseline_at_head(repo: str, repo_head: str):
    result = start_session(repo, None, CONFIG)

    assert result.session.baseline == BaselineRef(ref="HEAD", sha=repo_head)


# ---------------------------------------------------------------------------
# when a session is already on disk
# ---------------------------------------------------------------------------


def test_start_session_when_session_on_disk_does_resume_returning_counts_without_appending(
    repo: str,
):
    created = start_session(repo, "main", CONFIG).session
    keep_iteration(repo, 1)

    result = start_session(repo, "main", CONFIG)

    assert result.session == created
    assert result.resumed is True
    assert result.state.iteration_count == 1
    assert len(log_records(repo)) == 3


# ---------------------------------------------------------------------------
# when a baseline ref is given to a session that already exists
# ---------------------------------------------------------------------------


def _open_a_session(repo: str) -> None:
    """Open a session at ``main`` that the next start resumes."""
    start_session(repo, "main", CONFIG)


def _open_and_finalize_a_session(repo: str) -> None:
    """Open a session and close it, so the next start archives it and opens a fresh one."""
    start_session(repo, "main", CONFIG)
    _close_session_with_one_keep(repo)


@pytest.mark.parametrize(
    ("arrange", "ref", "warning_count"),
    [
        pytest.param(_open_a_session, "main", 1, id="resumed-with-ref"),
        pytest.param(_open_a_session, None, 0, id="resumed-without-ref"),
        pytest.param(_open_and_finalize_a_session, "main", 0, id="reopened-after-finalize"),
    ],
)
def test_start_session_when_a_session_exists_does_warn_only_when_a_ref_meets_a_resumed_one(
    repo: str, arrange: Callable[[str], None], ref: str | None, warning_count: int
):
    arrange(repo)
    warnings: list[str] = []

    start_session(repo, ref, CONFIG, warn=warnings.append)

    assert len(warnings) == warning_count
    assert all("main" in warning for warning in warnings)


# ---------------------------------------------------------------------------
# when the sample count cannot balance the two sides
# ---------------------------------------------------------------------------


def _footprint(repo: str) -> tuple[list[tuple[str, bytes]], list[str], list[str]]:
    """The session directory's files and bytes, git's worktrees, and the session branches."""
    state_dir = Path(session_jsonl_path(repo)).parent
    files = sorted(
        (str(path.relative_to(state_dir)), path.read_bytes())
        for path in state_dir.rglob("*")
        if path.is_file()
    )
    return files, list_worktree_dirs(repo), session_branches(repo)


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_open_and_finalize_a_session, id="closed-session"),
    ],
)
def test_start_session_when_samples_odd_does_refuse_leaving_the_repo_untouched(
    repo: str, arrange: Callable[[str], None]
):
    arrange(repo)
    before = _footprint(repo)

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "main", replace(CONFIG, samples=5))

    assert excinfo.value.reason == "unbalanced-samples"
    assert _footprint(repo) == before


# ---------------------------------------------------------------------------
# when the session on disk was finalized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "close",
    [
        pytest.param(_close_session_with_one_keep, id="worktrees-taken-down"),
    ],
)
def test_start_session_when_finalized_does_reopen_at_the_pinned_baseline_with_the_old_log_archived(
    repo: str, repo_head: str, close: Callable[[str], str]
):
    start_session(repo, "main", CONFIG)
    closed = close(repo)
    closed_log = log_records(repo)

    result = start_session(repo, "main", CONFIG)

    assert read_records(archived_session_path(repo, closed)) == closed_log
    assert result.archived == closed
    assert result.archived_path == archived_session_path(repo, closed)
    assert (result.resumed, result.state.finalized) == (False, None)
    assert result.session.session_id != closed
    assert log_records(repo) == [result.session]
    assert head_of(experiment_worktree_dir(repo)) == repo_head
    assert head_of(baseline_worktree_dir(repo)) == repo_head


@needs_posix_kill
def test_start_session_when_fresh_workspace_after_finalize_dies_does_put_the_closed_log_back(
    repo: str,
):
    start_session(repo, "main", CONFIG)
    closed = _close_session_with_one_keep(repo)
    closed_log = log_records(repo)
    kill_git_during_worktree_add(repo)

    with pytest.raises(GymratError):
        start_session(repo, "main", CONFIG)

    assert log_records(repo) == closed_log
    assert not Path(archived_session_path(repo, closed)).exists()


@needs_posix_kill
def test_start_session_when_putting_the_closed_log_back_fails_does_warn_naming_both_paths_raising_the_start_failure(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    start_session(repo, "main", CONFIG)
    closed = _close_session_with_one_keep(repo)
    kill_git_during_worktree_add(repo)
    jsonl = Path(session_jsonl_path(repo))
    rename = Path.rename

    def refuse_rename_back(self: Path, target: str | Path) -> Path:
        if Path(target) == jsonl:
            raise PermissionError(13, "Permission denied")
        return rename(self, target)

    monkeypatch.setattr(
        Path, "rename", create_autospec(Path.rename, side_effect=refuse_rename_back)
    )

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "main", CONFIG)

    archived = archived_session_path(repo, closed)
    # The rollback's own failure never speaks for the start's.
    assert experiment_worktree_dir(repo) in str(excinfo.value)
    assert Path(archived).exists()
    warning = capsys.readouterr().err
    assert archived in warning
    assert str(jsonl) in warning


# ---------------------------------------------------------------------------
# when the baseline worktree went missing
# ---------------------------------------------------------------------------


def test_start_session_when_baseline_worktree_missing_does_put_it_back_at_the_last_kept_commit(
    repo: str,
):
    start_session(repo, "main", CONFIG)
    kept = commit_and_keep(repo, 1, "cache the regex")
    shutil.rmtree(baseline_worktree_dir(repo))

    start_session(repo, "main", CONFIG)

    assert head_of(baseline_worktree_dir(repo)) == kept


def test_start_session_when_baseline_worktree_missing_and_nothing_kept_does_put_it_back_at_pinned_sha(
    repo: str, repo_head: str
):
    start_session(repo, "main", CONFIG)
    commit_all(experiment_worktree_dir(repo), "work the agent has not kept", file="README.md")
    shutil.rmtree(baseline_worktree_dir(repo))

    start_session(repo, "main", CONFIG)

    assert head_of(baseline_worktree_dir(repo)) == repo_head


# ---------------------------------------------------------------------------
# when the start fails after the workspace is built
# ---------------------------------------------------------------------------


@needs_mode_bits
@pytest.mark.usefixtures("read_only_log_dir")
def test_start_session_when_header_append_fails_does_remove_the_branch_and_worktrees_it_created(
    repo: str,
):
    with pytest.raises(PermissionError):
        start_session(repo, "main", CONFIG)

    assert session_branches(repo) == []
    assert list_worktree_dirs(repo, include_main=False) == []


@needs_mode_bits
@pytest.mark.usefixtures("read_only_log_dir")
def test_start_session_when_unwinding_a_failed_start_fails_does_warn_naming_each_worktree(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    # Neither git nor a plain delete can empty the worktrees, so the unwind's own steps fail.
    _pin_worktree_contents(repo)

    with pytest.raises(PermissionError):
        start_session(repo, "main", CONFIG)

    stderr = capsys.readouterr().err
    experiment, baseline = experiment_worktree_dir(repo), baseline_worktree_dir(repo)
    assert experiment in stderr
    assert baseline in stderr


@needs_mode_bits
def test_start_session_when_the_log_cannot_be_read_back_after_the_failure_does_unwind_its_workspace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    log = Path(session_jsonl_path(repo))

    def fail_and_lock_the_log(_descriptor: int) -> None:
        log.chmod(0o000)
        raise OSError(5, "sync failed")

    monkeypatch.setattr(os, "fsync", create_autospec(os.fsync, side_effect=fail_and_lock_the_log))

    with pytest.raises(OSError, match="sync failed"):
        start_session(repo, "main", CONFIG)

    assert session_branches(repo) == []
    assert list_worktree_dirs(repo, include_main=False) == []


def test_start_session_when_header_reached_the_log_before_the_failure_does_keep_its_workspace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    def failing_fsync(_descriptor: int) -> None:
        raise OSError(5, "sync failed")

    monkeypatch.setattr(os, "fsync", create_autospec(os.fsync, side_effect=failing_fsync))

    with pytest.raises(OSError, match="sync failed"):
        start_session(repo, "main", CONFIG)

    header = session_header_of(repo)
    assert session_branches(repo) == [header.branch]
    assert Path(experiment_worktree_dir(repo)).is_dir()


# ---------------------------------------------------------------------------
# when the baseline ref cannot be used
# ---------------------------------------------------------------------------


def test_start_session_when_baseline_ref_is_a_directory_does_raise_naming_the_ref(
    repo: str,
):
    target_dir = create_in_place_target_dir(repo, "bench-dir", "echo hi\n")

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, target_dir, CONFIG)

    assert target_dir in str(excinfo.value)
    assert not Path(session_jsonl_path(repo)).exists()
