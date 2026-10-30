"""Behavioral tests for the loop hook runner.

Every test drives a real subprocess: a numbered Python script under the current
interpreter, run through the asyncio ``exec`` layer. The scripts and their
parked payload files live in a throwaway ``tmp_path`` root, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The exception is the exec-cap test, which mocks the ``exec`` boundary: no real
subprocess can be made to overflow exec's own accumulation cap quickly.
"""

import asyncio
import json
from pathlib import Path

import pytest

from gymrat import exec as gymrat_exec
from gymrat.exec import FAILURE_EXIT_CODE, ExecOptions, ExecResult, ExecTimeoutError
from gymrat.loop.iterate.run import run_hook
from gymrat.session.records import IterationRecord, record_to_wire
from gymrat.session.schema import HookStage
from tests._clock import install_monotonic_clock
from tests._exec_fixtures import (
    CAPPED_STDERR_BYTES,
    CAPPED_STDOUT_BYTES,
    capped_result,
    install_exec,
)
from tests.loop.iterate._hooks import HookScripts, expected_hook_record, labeled_lines
from tests.session.records._fixtures import SESSION_ID, iteration_record


@pytest.fixture
def hooks(tmp_path: Path) -> HookScripts:
    """A ``HookScripts`` builder scoped to a fresh scratch directory."""
    return HookScripts.for_root(str(tmp_path))


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
        "experiment_dir": hooks.experiment_dir,
        "seq": seq,
        "last_iteration": record_to_wire(last_iteration) if last_iteration is not None else None,
        "session": {
            "session_id": SESSION_ID,
            "baseline": {"ref": "main", "sha": "a" * 40},
            "branch": f"gymrat/{SESSION_ID}",
            "iteration_count": iteration_count,
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


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        pytest.param(
            ("archived the samples", "pushed the branch"),
            "[after] archived the samples\n[after] pushed the branch",
            id="each-line-labeled",
        ),
        pytest.param((), "", id="nothing-printed-reports-empty"),
    ],
)
async def test_run_hook_when_command_prints_does_label_each_line_with_stage(
    hooks: HookScripts, lines: tuple[str, ...], expected: str
) -> None:
    command = hooks.printing(*lines)

    run = await run_hook(hooks.invocation_of(command, stage="after"))

    assert run.report == expected


async def test_run_hook_when_successful_hook_writes_both_channels_does_report_only_stdout(
    hooks: HookScripts,
) -> None:
    command = hooks.hook_command(
        "import sys\n"
        'sys.stdout.buffer.write(b"warmed the cache\\n")\n'
        'sys.stderr.buffer.write(b"cache was already warm\\n")\n'
    )

    run = await run_hook(hooks.invocation_of(command))

    assert run.report == "[before] warmed the cache"


# ---------------------------------------------------------------------------
# run_hook — timing
# ---------------------------------------------------------------------------


async def test_run_hook_when_command_finishes_does_record_its_duration(
    hooks: HookScripts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The clock stands still except while the hook runs, which takes 250 ms.
    clock = install_monotonic_clock(monkeypatch)

    async def exec_taking_250_ms(
        command: str, options: ExecOptions
    ) -> ExecResult | ExecTimeoutError:
        result = await gymrat_exec.exec(command, options)
        clock.tick(250.0)
        return result

    monkeypatch.setattr("gymrat.loop.iterate.run.exec", exec_taking_250_ms)

    run = await run_hook(hooks.invocation_of(hooks.hook_command("")))

    assert run.record.duration_ms == 250.0


# ---------------------------------------------------------------------------
# run_hook — a failing hook does not brick the loop
# ---------------------------------------------------------------------------


async def test_run_hook_when_failing_output_exceeds_relay_limit_does_cap_each_channel(
    hooks: HookScripts,
) -> None:
    out_line = "a" * 100
    err_line = "b" * 100
    stdout = f"{out_line}\n" * 200
    command = hooks.failing_content_of("both-channels", stdout, f"{err_line}\n" * 200)

    run = await run_hook(hooks.invocation_of(command))

    lines = labeled_lines(run.report, "before")
    exit_at = next(index for index, line in enumerate(lines) if line != out_line)
    relayed_out, relayed_err = lines[:exit_at], lines[exit_at + 1 :]
    assert set(relayed_out) == {out_line}
    assert set(relayed_err) == {err_line}
    assert len(relayed_out) < 200
    assert len(relayed_err) < 200


async def test_run_hook_when_hook_exits_nonzero_does_surface_the_failure(
    hooks: HookScripts,
) -> None:
    command = hooks.failing_content_of("exits-nonzero", "checked the cache\n", "no warm copy\n")

    run = await run_hook(hooks.invocation_of(command))

    stdout_line, exit_line, stderr_line = labeled_lines(run.report, "before")
    assert (stdout_line, "3" in exit_line, stderr_line) == (
        "checked the cache",
        True,
        "no warm copy",
    )
    assert run.record.model_copy(update={"duration_ms": 0, "at": 0}) == expected_hook_record(
        stage="before", seq=2, exit_code=3, stdout_bytes=18, stderr_bytes=13
    )


async def test_run_hook_when_hook_outruns_timeout_does_kill_it_as_timed_out(
    hooks: HookScripts, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = hooks.hook_command("import time\ntime.sleep(5)\n")
    monkeypatch.setattr("gymrat.loop.iterate.run.HOOK_TIMEOUT_MS", 200)

    run = await run_hook(hooks.invocation_of(command))

    (timeout_line,) = labeled_lines(run.report, "before")
    assert "200ms" in timeout_line
    assert run.record.timed_out is True
    assert run.record.exit_code == FAILURE_EXIT_CODE


async def test_run_hook_when_abort_signal_set_does_kill_it(hooks: HookScripts) -> None:
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


# ---------------------------------------------------------------------------
# run_hook — pre-cap byte counts survive exec's own output cap
# ---------------------------------------------------------------------------


async def test_run_hook_when_exec_output_capped_does_record_pre_cap_byte_counts(
    hooks: HookScripts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_exec(monkeypatch, "gymrat.loop.iterate.run.exec", capped_result())

    run = await run_hook(hooks.invocation_of("unused-because-exec-is-mocked"))

    assert run.record.stdout_bytes == CAPPED_STDOUT_BYTES
    assert run.record.stderr_bytes == CAPPED_STDERR_BYTES
