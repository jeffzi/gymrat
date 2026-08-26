"""Behavioral tests for the loop hook runner.

Every test drives a real subprocess: a numbered Python script under the current
interpreter, run through the asyncio ``exec`` layer. The scripts and their
parked payload files live in a throwaway ``tmp_path`` root, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The one exception is the exec-cap test, which mocks the ``exec`` boundary: no
real subprocess can be made to overflow exec's own accumulation cap quickly.
"""

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from gymrat_py.exec import FAILURE_EXIT_CODE, ExecOptions, ExecResult
from gymrat_py.loop.hooks import run_hook
from gymrat_py.session import IterationRecord, Worktrees, record_to_wire
from gymrat_py.session.schema import HookStage
from tests.loop._hooks import (
    HookScripts,
    expected_hook_record,
    labeled_lines,
)
from tests.session._records import SESSION_ID, iteration_record, session_record

#: The cap the runner holds each of a hook's channels to before it reaches
#: gymrat's own output.
RELAY_LIMIT_BYTES = 8192


@pytest.fixture
def hooks(tmp_path: Path) -> HookScripts:
    experiment_dir = tmp_path / "side-experiment"
    experiment_dir.mkdir()
    return HookScripts(str(tmp_path), str(experiment_dir))


# ---------------------------------------------------------------------------
# run_hook — payload and worktree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "seq", "last_iteration", "iteration_count"),
    [
        pytest.param("before", 3, None, 2, id="before"),
        pytest.param("after", 3, iteration_record(seq=3), 3, id="after"),
    ],
)
async def test_run_hook_when_command_runs_does_hand_stage_payload_on_stdin(
    hooks: HookScripts,
    stage: HookStage,
    seq: int,
    last_iteration: IterationRecord | None,
    iteration_count: int,
) -> None:
    command = hooks.hook_command("import sys; sys.stdout.write(sys.stdin.read())")

    run = await run_hook(
        hooks.invocation_of(
            command,
            stage=stage,
            seq=seq,
            last_iteration=last_iteration,
            iteration_count=iteration_count,
        )
    )

    payload = json.loads("\n".join(labeled_lines(run.report, stage)))
    assert payload == {
        "stage": stage,
        "experimentDir": hooks.experiment_dir,
        "seq": seq,
        "lastIteration": record_to_wire(last_iteration) if last_iteration is not None else None,
        "session": {
            "sessionId": SESSION_ID,
            "baseline": {"ref": "main", "sha": "a" * 40},
            "branch": f"gymrat/{SESSION_ID}",
            "iterationCount": iteration_count,
        },
    }


async def test_run_hook_when_command_runs_does_run_in_experiment_worktree(
    hooks: HookScripts,
) -> None:
    command = hooks.hook_command('open("landed.txt", "w", encoding="utf-8").write("here")')

    await run_hook(hooks.invocation_of(command))

    assert (Path(hooks.experiment_dir) / "landed.txt").read_text(encoding="utf-8") == "here"


# ---------------------------------------------------------------------------
# run_hook — report labeling
# ---------------------------------------------------------------------------


async def test_run_hook_when_command_prints_does_label_each_line_with_stage(
    hooks: HookScripts,
) -> None:
    command = hooks.printing("archived the samples", "pushed the branch")

    run = await run_hook(hooks.invocation_of(command, stage="after"))

    assert run.report == "[after] archived the samples\n[after] pushed the branch"


async def test_run_hook_when_hook_prints_nothing_does_report_empty(hooks: HookScripts) -> None:
    command = hooks.hook_command("")

    run = await run_hook(hooks.invocation_of(command))

    assert run.report == ""


async def test_run_hook_when_successful_does_keep_stderr_out_of_report(hooks: HookScripts) -> None:
    command = hooks.hook_command(
        "import sys\n"
        'sys.stdout.buffer.write(b"warmed the cache\\n")\n'
        'sys.stderr.buffer.write(b"cache was already warm\\n")\n'
    )

    run = await run_hook(hooks.invocation_of(command))

    assert run.report == "[before] warmed the cache"


# ---------------------------------------------------------------------------
# run_hook — recorded byte counts
# ---------------------------------------------------------------------------


async def test_run_hook_when_command_runs_does_record_bytes_the_hook_printed(
    hooks: HookScripts,
) -> None:
    command = hooks.printing("hello")

    run = await run_hook(hooks.invocation_of(command))

    assert dataclasses.replace(run.record, duration_ms=0) == expected_hook_record(
        stage="before", seq=2, exit_code=0, stdout_bytes=6
    )


async def test_run_hook_when_hook_writes_stderr_does_record_stderr_bytes(
    hooks: HookScripts,
) -> None:
    command = hooks.hook_command(
        'import sys\nsys.stdout.buffer.write(b"hello\\n")\nsys.stderr.buffer.write(b"warning\\n")\n'
    )

    run = await run_hook(hooks.invocation_of(command))

    assert run.record.stdout_bytes == 6
    assert run.record.stderr_bytes == 8


# ---------------------------------------------------------------------------
# run_hook — the 8 KiB relay cut
# ---------------------------------------------------------------------------


async def test_run_hook_when_stdout_over_budget_does_cut_at_last_whole_line(
    hooks: HookScripts,
) -> None:
    line = "a" * 100
    text = f"{line}\n" * 200
    command = hooks.printing_content_of("many-lines.txt", text)

    run = await run_hook(hooks.invocation_of(command))

    assert labeled_lines(run.report, "before") == [line] * 81
    assert run.record.stdout_bytes == len(text.encode("utf-8"))


async def test_run_hook_when_stdout_single_long_line_does_not_split_multi_byte_char(
    hooks: HookScripts,
) -> None:
    content = "é" * 5000
    command = hooks.printing_content_of("one-long-line.txt", content)

    run = await run_hook(hooks.invocation_of(command))

    assert labeled_lines(run.report, "before") == ["é" * (RELAY_LIMIT_BYTES // 2)]
    assert run.record.stdout_bytes == len(content.encode("utf-8"))


# ---------------------------------------------------------------------------
# run_hook — a failing hook does not brick the loop
# ---------------------------------------------------------------------------


async def test_run_hook_when_failing_over_budget_does_cap_each_channel(hooks: HookScripts) -> None:
    out_line = "a" * 100
    err_line = "b" * 100
    stdout = f"{out_line}\n" * 200
    command = hooks.failing_content_of("both-channels", stdout, f"{err_line}\n" * 200)

    run = await run_hook(hooks.invocation_of(command))

    assert labeled_lines(run.report, "before") == (
        [out_line] * 81 + ["hook exited 3"] + [err_line] * 81
    )
    assert run.record.stdout_bytes == len(stdout.encode("utf-8"))


async def test_run_hook_when_failing_stderr_long_line_does_not_split_multi_byte_char(
    hooks: HookScripts,
) -> None:
    command = hooks.failing_content_of("long-stderr-line", "", "é" * 5000)

    run = await run_hook(hooks.invocation_of(command))

    assert labeled_lines(run.report, "before") == [
        "hook exited 3",
        "é" * (RELAY_LIMIT_BYTES // 2),
    ]


async def test_run_hook_when_hook_exits_nonzero_does_report_and_record(hooks: HookScripts) -> None:
    command = hooks.hook_command(
        "import sys\n"
        'sys.stdout.buffer.write(b"checked the cache\\n")\n'
        'sys.stderr.buffer.write(b"no warm copy\\n")\n'
        "sys.exit(3)\n"
    )

    run = await run_hook(hooks.invocation_of(command))

    assert labeled_lines(run.report, "before") == [
        "checked the cache",
        "hook exited 3",
        "no warm copy",
    ]
    assert dataclasses.replace(run.record, duration_ms=0) == expected_hook_record(
        stage="before", seq=2, exit_code=3, stdout_bytes=18, stderr_bytes=13
    )


async def test_run_hook_when_hook_outruns_timeout_does_kill_and_report(hooks: HookScripts) -> None:
    command = hooks.hook_command("import time\ntime.sleep(5)\n")

    run = await run_hook(hooks.invocation_of(command, timeout_ms=200))

    assert labeled_lines(run.report, "before") == ["hook timed out after 200ms"]
    assert run.record.timed_out is True
    assert run.record.exit_code == FAILURE_EXIT_CODE
    assert run.record.duration_ms < 4000


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells return 127 for a missing command")
async def test_run_hook_when_command_not_found_does_record_shell_exit_code(
    hooks: HookScripts,
) -> None:
    run = await run_hook(hooks.invocation_of("nonexistent-command-abc123xyz"))

    assert run.record.exit_code == 127


async def test_run_hook_when_worktree_vanished_does_report_instead_of_raising(
    hooks: HookScripts,
) -> None:
    command = hooks.printing("never runs")
    vanished_dir = str(Path(hooks.temp_dir) / "vanished")
    session = session_record(
        session_id=SESSION_ID,
        worktrees=Worktrees(
            experiment=vanished_dir,
            baseline=str(Path(hooks.temp_dir) / "side-baseline"),
        ),
    )

    run = await run_hook(hooks.invocation_of(command, session=session))

    lines = labeled_lines(run.report, "before")
    assert lines[0] == f"hook exited {FAILURE_EXIT_CODE}"
    assert "No such file or directory" in "\n".join(lines[1:])
    assert run.record.stdout_bytes == 0
    assert run.record.timed_out is False


async def test_run_hook_when_abort_signal_set_does_kill_and_record(hooks: HookScripts) -> None:
    command = hooks.hook_command("import time\ntime.sleep(10)\n")
    abort = asyncio.Event()

    async def trigger() -> None:
        await asyncio.sleep(0.05)
        abort.set()

    trigger_task = asyncio.create_task(trigger())
    run = await run_hook(hooks.invocation_of(command, abort=abort))
    await trigger_task

    assert run.record.exit_code == FAILURE_EXIT_CODE
    assert run.record.timed_out is False
    assert run.record.duration_ms < 4000


# ---------------------------------------------------------------------------
# run_hook — pre-cap byte counts survive exec's own output cap
# ---------------------------------------------------------------------------


async def test_run_hook_when_exec_output_capped_does_record_pre_cap_byte_counts(
    hooks: HookScripts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def capped_exec(command: str, options: ExecOptions) -> ExecResult:
        return ExecResult(
            stdout="capped stdout",
            stderr="capped stderr",
            exit_code=0,
            stdout_bytes=200_000,
            stderr_bytes=150_000,
        )

    monkeypatch.setattr("gymrat_py.loop.hooks.exec", capped_exec)

    run = await run_hook(hooks.invocation_of("unused-because-exec-is-mocked"))

    assert run.record.stdout_bytes == 200_000
    assert run.record.stderr_bytes == 150_000
