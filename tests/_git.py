"""Shared git helpers for test fixtures and test modules."""

import contextlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

#: A bench script that exits non-zero without reporting a metric.
FAILING_BENCH = "#!/bin/sh\nexit 1\n"

#: The flags that run the bench ``write_committed_bench`` commits, once.
BENCH_ONCE_FLAGS = ("--bench", "sh bench.sh", "--samples", "1")


def emit_bench(value: int) -> str:
    """A bench script that reports the metric ``x`` as ``value`` and exits cleanly."""
    return f"#!/bin/sh\necho 'METRIC x={value}'\n"


#: A bench script that reports one metric line and exits cleanly.
EMIT_ONE_BENCH = emit_bench(1)


def run_git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd``, returning its stripped stdout and failing loudly on error."""
    result = subprocess.run(  # noqa: S603 -- fixed git binary plus test-chosen args
        ["git", *args],  # noqa: S607 -- git resolved from PATH like a user's shell
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def head_of(worktree: str) -> str:
    """The commit ``worktree`` currently has checked out."""
    return run_git(["rev-parse", "HEAD"], worktree)


def checked_out_ref(worktree: str) -> str:
    """The ref ``worktree`` has checked out: a branch name, or ``HEAD`` when detached."""
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"], worktree)


def git_exclude_path(root: str) -> Path:
    """The ``.git/info/exclude`` file of the repository at ``root``."""
    return Path(root) / ".git" / "info" / "exclude"


def write_committed_bench(
    repo: str, script: str, *, message: str = "add bench", branches: tuple[str, ...] = ()
) -> None:
    """Drop ``script`` as ``bench.sh`` and commit it so every ref can run it.

    Args:
        repo: The repository to commit into.
        script: The bench script's contents.
        message: The commit message.
        branches: Branches to create at the new commit; ``repo`` stays on its
            current branch.
    """
    (Path(repo) / "bench.sh").write_text(script, encoding="utf-8")
    run_git(["add", "bench.sh"], repo)
    run_git(["commit", "-m", message], repo)
    for branch in branches:
        run_git(["branch", branch], repo)


def status_of(worktree: str) -> str:
    """The porcelain status of ``worktree`` — empty when nothing is uncommitted."""
    return run_git(["status", "--porcelain"], worktree)


def commit_all(
    worktree: str, message: str, *, file: str | None = None, content: str | None = None
) -> str:
    """Stage everything in ``worktree`` and commit it.

    Args:
        worktree: The checkout the commit is made in.
        message: The commit message.
        file: A file, relative to ``worktree``, written before staging so the
            commit is never empty; ``None`` commits only what is already there.
        content: The text written to ``file``; defaults to ``message`` plus a
            newline.

    Returns:
        The SHA ``worktree`` has checked out after the commit.
    """
    if file is not None:
        text = f"{message}\n" if content is None else content
        (Path(worktree) / file).write_text(text, encoding="utf-8")
    run_git(["add", "-A"], worktree)
    run_git(["commit", "-m", message], worktree)
    return head_of(worktree)


def session_branches(root: str) -> list[str]:
    """The session branches ``root`` still holds, one per ``gymrat/…`` ref."""
    output = run_git(["for-each-ref", "--format=%(refname:short)", "refs/heads/gymrat"], root)
    return output.splitlines()


def install_git_hook(repo_dir: str, name: str, body: str) -> None:
    """Install an executable ``sh`` hook called ``name`` in ``repo_dir``.

    Args:
        repo_dir: The repository whose ``.git/hooks`` directory receives the hook.
        name: The hook's name, such as ``post-checkout``.
        body: The script that follows the ``#!/bin/sh`` line.
    """
    hook_path = Path(repo_dir) / ".git" / "hooks" / name
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    hook_path.chmod(0o755)


def kill_git_during_worktree_add(repo_dir: str) -> None:
    """Install a post-checkout hook that kills git once a worktree is on disk."""
    install_git_hook(repo_dir, "post-checkout", 'exec >/dev/null 2>&1\nkill -9 "$PPID"\nsleep 1\n')


def init_scratch_repo(prefix: str) -> str:
    """Create one temporary git repo on ``main`` with a single committed file.

    The directory comes from ``tempfile.mkdtemp`` resolved through
    ``Path.resolve``, so macOS ``/var`` → ``/private/var`` matches what git
    reports in ``worktree list``.

    Args:
        prefix: How the repository's directory name starts.

    Returns:
        The resolved path of the new repository.
    """
    directory = str(Path(tempfile.mkdtemp(prefix=prefix)).resolve())
    try:
        run_git(["init", "-b", "main"], directory)
        for key, value in (
            ("user.name", "Test User"),
            ("user.email", "test@example.com"),
            ("commit.gpgsign", "false"),
            ("core.autocrlf", "false"),
        ):
            run_git(["config", key, value], directory)
        (Path(directory) / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        run_git(["add", "README.md"], directory)
        run_git(["commit", "-m", "Initial commit"], directory)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return directory


def list_worktree_dirs(repo_dir: str, *, include_main: bool = True) -> list[str]:
    """Directories git currently lists as worktrees of ``repo_dir``.

    ``git worktree remove`` clears a worktree's registry entry itself, so this
    only reveals pruning behavior when a directory vanished behind git's back.

    Args:
        repo_dir: The repository whose worktree registry is listed.
        include_main: Whether the main worktree's own directory stays in the
            result. Git prints resolved paths, so the main directory is matched
            through ``Path.resolve``.

    Returns:
        The listed directories, in git's order.

    Raises:
        subprocess.CalledProcessError: When git cannot list the registry.
    """
    output = run_git(["worktree", "list", "--porcelain"], repo_dir)
    dirs = [
        str(Path(line[len("worktree ") :]))
        for line in output.split("\n")
        if line.startswith("worktree ")
    ]
    if not include_main:
        main_dir = str(Path(repo_dir).resolve())
        return [directory for directory in dirs if directory != main_dir]
    return dirs


def wait_for_worktrees(repo_dir: str, count: int, timeout_s: float = 30.0) -> list[str]:
    """Poll until ``repo_dir`` lists at least ``count`` linked worktrees.

    Only for polling while another process is still adding or removing
    worktrees: ``git worktree list`` reads each registry entry non-atomically
    and exits 128 when it meets one half-written by a concurrent ``git worktree
    add`` or half-cleared by a ``remove`` — a file of the entry it expects is
    not there. Such a read counts as "not there yet". Once no writer runs,
    call :func:`list_worktree_dirs` directly so a real failure stays loud.

    Args:
        repo_dir: The main worktree whose registry is polled.
        count: The minimum number of linked worktrees to wait for.
        timeout_s: How long to poll before giving up.

    Returns:
        The linked worktree directories of the first read that reached ``count``.

    Raises:
        AssertionError: When ``count`` is not reached within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    listed: list[str] = []
    while True:
        with contextlib.suppress(subprocess.CalledProcessError):
            listed = list_worktree_dirs(repo_dir, include_main=False)
        if len(listed) >= count:
            return listed
        if time.monotonic() > deadline:
            message = f"expected >= {count} worktrees within {timeout_s}s, saw {listed}"
            raise AssertionError(message)
        time.sleep(0.05)


def add_worktree(repo_dir: str, directory: str) -> str:
    """Register a linked worktree of a repo, detached at its ``HEAD``.

    Args:
        repo_dir: The repository the worktree is registered with.
        directory: Where the worktree goes, absolute or relative to ``repo_dir``;
            missing parent directories are created.

    Returns:
        The worktree's directory.
    """
    worktree = Path(repo_dir, directory)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", "--detach", str(worktree), "HEAD"], repo_dir)
    return str(worktree)


def register_absent_worktree(repo_dir: str) -> str:
    """Register a worktree of a repo the way a user would, then delete its dir.

    Args:
        repo_dir: The repository the worktree is registered with.

    Returns:
        The registered worktree's directory, which no longer exists.
    """
    directory = add_worktree(repo_dir, str(Path(repo_dir).resolve() / "absent-user-worktree"))
    shutil.rmtree(directory)
    return directory


def create_in_place_target_dir(repo_dir: str, name: str, bench_script: str) -> str:
    """Write a bench script into a plain subdirectory of a repo.

    Args:
        repo_dir: The repository the subdirectory is created in.
        name: The subdirectory's name.
        bench_script: The text of the ``bench.sh`` written into it.

    Returns:
        The subdirectory's path.
    """
    target = Path(repo_dir) / name
    target.mkdir()
    (target / "bench.sh").write_text(bench_script, encoding="utf-8")
    return str(target)


def refuse_session_branch_deletion(repo_dir: str) -> None:
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
