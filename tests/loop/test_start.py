"""Behavioral tests for starting or resuming a session (``start_session``).

Every test drives the real ``start_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the
worktree checkouts, resume recreation, and finalize archive-and-recreate only
reveal their behavior against real worktrees, and the assertions read commit SHAs
straight out of the worktrees git laid down.
"""

import contextlib
import itertools
import os
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from gymrat.config import HooksConfig, ResolvedConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.finalize import finalize_session
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
    worktrees_at,
)

SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")

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


def _close_after_removing_the_worktree(root: str) -> str:
    """Keep one commit, delete the experiment worktree's directory, then finalize.

    With the directory gone first, ``git worktree remove`` finds nothing to take
    and git keeps its entry for the path, which the next start must step over.

    Args:
        root: The repository whose open session is closed.

    Returns:
        The id of the session it closed.
    """
    header = session_header_of(root)
    commit_and_keep(root, 1, "cache the regex")
    shutil.rmtree(experiment_worktree_dir(root))
    finalize_session(root)
    return header.session_id


def _pin_worktree_contents(repo_dir: str) -> None:
    """Install a post-checkout hook that leaves each new worktree a directory nothing can empty."""
    install_git_hook(
        repo_dir,
        "post-checkout",
        f"mkdir {PINNED_DIR} && : > {PINNED_DIR}/file && chmod 500 {PINNED_DIR}\n",
    )


def _refuse_session_branch_deletion(repo_dir: str) -> None:
    """Install a reference-transaction hook that aborts any delete of a ``gymrat/…`` branch.

    Git names the ref's new value as all zeros when it deletes the ref, so the hook
    lets the branch be created and moved and vetoes only its removal.

    Args:
        repo_dir: The repository the hook is installed in.
    """
    install_git_hook(
        repo_dir,
        "reference-transaction",
        '[ "$1" = prepared ] || exit 0\n'
        "while read -r _old new ref; do\n"
        '    if [ "${ref#refs/heads/gymrat/}" != "$ref" ] && [ -z "$(printf %s "$new" | tr -d 0)" ]; then\n'
        "        exit 1\n"
        "    fi\n"
        "done\n"
        "exit 0\n",
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
def test_start_session_when_no_session_yet_does_write_header_naming_baseline_branch_worktrees_and_config(
    repo: str, repo_head: str, config: ResolvedConfig, snapshot: SessionConfig
):
    result = start_session(repo, "main", config)

    header = session_header_of(repo)
    assert result == StartResult(session=header, state=fold_session([header]), resumed=False)
    assert log_records(repo) == [header]
    assert SESSION_ID_PATTERN.match(header.session_id)
    assert header.baseline == BaselineRef(ref="main", sha=repo_head)
    assert header.branch == f"gymrat/{header.session_id}"
    assert header.worktrees == worktrees_at(repo)
    assert Path(experiment_worktree_dir(repo)).exists()
    assert Path(baseline_worktree_dir(repo)).exists()
    assert header.config == snapshot


def test_start_session_when_new_does_mint_the_session_id_from_the_instant_the_header_stamps(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # 2024-03-05T06:07:08.9Z in epoch nanoseconds, then a day later on every further read.
    instant_ns = 1_709_618_828_900_000_000
    day_ns = 86_400_000_000_000
    reads = itertools.count(instant_ns, day_ns)
    monkeypatch.setattr("gymrat.clock.now_ns", lambda: next(reads))

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
    assert result.state.keep_count == 1
    assert len(log_records(repo)) == 3


def test_start_session_when_experiment_worktree_missing_does_put_it_back(repo: str):
    start_session(repo, "main", CONFIG)
    shutil.rmtree(experiment_worktree_dir(repo))

    start_session(repo, "main", CONFIG)

    assert Path(experiment_worktree_dir(repo)).exists()


# ---------------------------------------------------------------------------
# when the session on disk was finalized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "close",
    [
        pytest.param(_close_session_with_one_keep, id="worktrees-taken-down"),
        pytest.param(_close_after_removing_the_worktree, id="worktree-removed-before-finalize"),
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

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "main", CONFIG)

    # The rollback's own failure never speaks for the start's.
    assert re.search(r"cannot create the experiment worktree", str(excinfo.value), re.IGNORECASE)
    assert log_records(repo) == closed_log
    assert not Path(archived_session_path(repo, closed)).exists()


@needs_posix_kill
def test_start_session_when_putting_the_closed_log_back_fails_does_raise_the_start_failure(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(Path, "rename", refuse_rename_back)

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "main", CONFIG)

    assert re.search(r"cannot create the experiment worktree", str(excinfo.value), re.IGNORECASE)
    assert Path(archived_session_path(repo, closed)).exists()


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
    assert not Path(experiment_worktree_dir(repo)).exists()
    assert not Path(baseline_worktree_dir(repo)).exists()


def _fail_a_start_on_the_header(repo: str, log_dir: Path) -> None:
    """Run a start that fails appending its header, then let the log directory take files again."""
    with contextlib.suppress(PermissionError):
        start_session(repo, "main", CONFIG)
    log_dir.chmod(0o700)


@needs_mode_bits
def test_start_session_when_earlier_start_failed_on_the_header_does_open_a_fresh_session(
    repo: str, read_only_log_dir: Path
):
    _fail_a_start_on_the_header(repo, read_only_log_dir)

    result = start_session(repo, "main", CONFIG)

    assert result.resumed is False
    assert log_records(repo) == [result.session]
    assert session_branches(repo) == [result.session.branch]


@needs_mode_bits
@pytest.mark.usefixtures("read_only_log_dir")
def test_start_session_when_unwinding_a_failed_start_fails_does_warn_naming_the_worktree(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    # Neither git nor a plain delete can empty the worktrees, so the unwind's own steps fail.
    _pin_worktree_contents(repo)

    with pytest.raises(PermissionError):
        start_session(repo, "main", CONFIG)

    assert experiment_worktree_dir(repo) in capsys.readouterr().err


@needs_mode_bits
@pytest.mark.usefixtures("read_only_log_dir")
def test_start_session_when_unwind_cannot_delete_the_branch_does_warn_with_the_branch_delete_command(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    _refuse_session_branch_deletion(repo)

    with pytest.raises(PermissionError):
        start_session(repo, "main", CONFIG)

    (branch,) = session_branches(repo)
    assert f"git branch -D {branch}" in capsys.readouterr().err


def test_start_session_when_header_reached_the_log_before_the_failure_does_keep_its_workspace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    def failing_fsync(_descriptor: int) -> None:
        raise OSError(5, "sync failed")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="sync failed"):
        start_session(repo, "main", CONFIG)

    header = session_header_of(repo)
    assert session_branches(repo) == [header.branch]
    assert Path(experiment_worktree_dir(repo)).is_dir()


@needs_posix_kill
def test_start_session_when_resume_fails_does_leave_the_standing_worktree_and_its_work(
    repo: str,
):
    start_session(repo, "main", CONFIG)
    draft = Path(experiment_worktree_dir(repo)) / "draft.txt"
    draft.write_text("work the agent has not kept\n", encoding="utf-8")
    shutil.rmtree(baseline_worktree_dir(repo))
    kill_git_during_worktree_add(repo)

    with pytest.raises(GymratError):
        start_session(repo, "main", CONFIG)

    assert draft.read_text(encoding="utf-8") == "work the agent has not kept\n"


# ---------------------------------------------------------------------------
# when the baseline ref cannot be used
# ---------------------------------------------------------------------------


def test_start_session_when_baseline_ref_does_not_resolve_does_leave_no_session(
    repo: str,
):
    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "no-such-ref", CONFIG)

    assert "no-such-ref" in str(excinfo.value)
    assert not Path(session_jsonl_path(repo)).exists()


def test_start_session_when_baseline_ref_is_a_directory_does_raise_naming_the_ref(
    repo: str,
):
    target_dir = create_in_place_target_dir(repo, "bench-dir", "echo hi\n")

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, target_dir, CONFIG)

    assert str(excinfo.value) == (
        f"Cannot start a session at '{target_dir}': it names a directory, not a git ref"
    )
    assert excinfo.value.hint == (
        "Pass a branch, tag, or commit the session's baseline is pinned to."
    )
    assert not Path(session_jsonl_path(repo)).exists()
