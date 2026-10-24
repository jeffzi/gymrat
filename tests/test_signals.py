import contextlib
import dataclasses
import io
import signal
import sys
import threading
import time
import warnings
from collections.abc import Buffer, Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import NoReturn, override

import pytest
from rich.console import Console
from rich.file_proxy import FileProxy

from gymrat import signals
from gymrat.signals import install_termination_cleanup
from tests._inner_pytest import SINGLE_PROCESS_ARGS, inherited_env, run_inner_pytest
from tests._signal_masking import needs_signal_masking
from tests._streams import RecordingStream

# Invokes the handler installed for a signal and returns the code it would exit
# with. Supplied by the ``raise_signal`` fixture.
RaiseSignal = Callable[[int], int]

# Termination signals available on this platform. SIGHUP is POSIX-only.
_TERMINATION_SIGNALS = [signal.SIGINT, signal.SIGTERM]
if hasattr(signal, "SIGHUP"):
    _TERMINATION_SIGNALS.append(signal.SIGHUP)

# An exit-write timeout apart from the one-second default, so a handler that
# kept the default instead of the installed timeout is told apart; short enough
# that waiting it out for real keeps a test fast.
_SHORT_EXIT_WRITE_TIMEOUT_S = 0.05

# The exit-write timeout the second-signal test installs. The second signal ends
# the process before any exit wait runs to its end; a handler that ignored it
# would sit through this whole wait on the stalled write instead.
_LONG_EXIT_WRITE_TIMEOUT_S = 5.0

# Longest a scenario run by ``_run_capped`` may take: well past the longest exit
# wait those scenarios install, well short of a hang.
_SCENARIO_CAP_S = 5.0

# Threads ``_run_capped`` gave up on, still inside a handler whose wait may yet
# end in a call to the exit seam.
_stranded_threads: list[threading.Thread] = []


def _signal_id(signal_number: int) -> str:
    return signal.Signals(signal_number).name


def _boom() -> None:
    message = "cleanup boom"
    raise RuntimeError(message)


def _write_erase_notice() -> None:
    signals.write_on_exit("live display erased\n")


def _escalation_boom() -> None:
    message = "escalation boom"
    raise RuntimeError(message)


def _warn_killpg_failed() -> None:
    warnings.warn("killpg failed", RuntimeWarning, stacklevel=1)


def _installed_handler(signal_number: int) -> Callable[[int, FrameType | None], object]:
    handler = signal.getsignal(signal_number)
    if not callable(handler):
        pytest.fail(f"no handler installed for signal {signal_number}")
    return handler


def _deliver(signal_number: int) -> None:
    # Runs the installed handler the way a signal arriving now would, without
    # the exit bookkeeping ``raise_signal`` adds: an ignored signal returns.
    _installed_handler(signal_number)(signal_number, None)


def _deliver_from_thread(signal_number: int) -> None:
    # Runs the handler on another thread, the way a signal received there with
    # the main thread's mask set would still reach it.
    deliverer = threading.Thread(target=_deliver, args=(signal_number,))
    deliverer.start()
    deliverer.join()


def _install_sigterm_on_cleanup(raise_signal: RaiseSignal) -> None:
    # A cleanup that sends the second signal, so the escalation runs.
    def interrupt() -> None:
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)


class _HookedRaw(io.RawIOBase):
    """A raw byte sink that runs a hook inside every ``write`` before storing."""

    def __init__(self, on_write: Callable[[], None]) -> None:
        super().__init__()
        self._on_write = on_write
        self.data = bytearray()

    @override
    def writable(self) -> bool:
        return True

    @override
    def write(self, b: Buffer, /) -> int:
        self._on_write()
        chunk = bytes(b)
        self.data.extend(chunk)
        return len(chunk)


@dataclasses.dataclass(frozen=True, slots=True)
class _Terminal:
    """A buffered stderr stand-in and the raw sink holding what reached the terminal."""

    stream: io.TextIOWrapper
    raw: _HookedRaw

    def write_progress(self) -> None:
        """Write a progress line to the buffered stream and flush it toward the raw sink."""
        self.stream.write("progress\n")
        self.stream.flush()


def _buffered_terminal(on_write: Callable[[], None]) -> _Terminal:
    # The same layering as the real sys.stderr: a TextIOWrapper over a
    # BufferedWriter, whose lock is what makes a same-thread write reentrant.
    raw = _HookedRaw(on_write)
    return _Terminal(io.TextIOWrapper(io.BufferedWriter(raw), encoding="utf-8"), raw)


def _run_capped[T](scenario: Callable[[], T]) -> T:
    """Run ``scenario`` on a daemon thread and fail the test if it outlasts the cap."""
    # A handler whose exit-output wait lost its bound deadlocks the thread that
    # runs it; on a daemon thread that fails this test instead of hanging the
    # suite.
    results: list[T] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(scenario())
        except BaseException as error:  # noqa: BLE001 - re-raised on the test's own thread below
            errors.append(error)

    worker = threading.Thread(target=run, name="capped-scenario", daemon=True)
    worker.start()
    worker.join(_SCENARIO_CAP_S)
    if worker.is_alive():
        _stranded_threads.append(worker)
        pytest.fail(f"scenario still running after {_SCENARIO_CAP_S}s: the exit wait is unbounded")
    if errors:
        raise errors[0]
    return results[0]


def _park_forever(code: int) -> NoReturn:
    while True:
        threading.Event().wait()


@pytest.fixture(autouse=True)
def _disarm_stranded_exits() -> Iterator[None]:
    """Keep a thread ``_run_capped`` gave up on from ever reaching the real process exit."""
    # Autouse, so it is set up before the monkeypatch that stubs the exit seam
    # and torn down after that stub is undone. A stranded thread whose wait
    # ends later would then call the real os._exit and take the whole test
    # worker down; the seam is swapped for good for one that parks it instead.
    yield
    if any(thread.is_alive() for thread in _stranded_threads):
        signals.exit_process = _park_forever


# ---------------------------------------------------------------------------
# Cleanup ordering and exit codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal_number", _TERMINATION_SIGNALS, ids=_signal_id)
def test_install_termination_cleanup_when_signal_received_does_exit_128_plus_signal_number_after_running_cleanup(
    raise_signal: RaiseSignal, signal_number: int
):
    calls = []
    install_termination_cleanup(lambda: calls.append("cleanup"))

    code = raise_signal(signal_number)

    assert calls == ["cleanup"]
    assert code == 128 + signal_number


class _ReplacedExitError(BaseException):
    """Unwinds the handler where the replaced exit seam would end the process."""


def _record_exits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the exit seam with one that records each exit code and raises ``_ReplacedExitError``.

    Args:
        monkeypatch: The fixture that installs the replacement.

    Returns:
        The exit codes, in the order the replacement received them.
    """
    exits: list[int] = []

    def record_exit(code: int) -> NoReturn:
        exits.append(code)
        raise _ReplacedExitError

    monkeypatch.setattr(signals, "exit_process", record_exit)
    return exits


@pytest.mark.parametrize(
    "second_signal", [None, signal.SIGTERM], ids=["first-signal", "escalation"]
)
@pytest.mark.usefixtures("forbid_direct_exit")
def test_install_termination_cleanup_when_exit_seam_replaced_after_install_does_exit_through_replacement(
    monkeypatch: pytest.MonkeyPatch, second_signal: int | None
):
    def interrupt() -> None:
        if second_signal is None:
            return
        _installed_handler(second_signal)(second_signal, None)

    install_termination_cleanup(interrupt)
    handler = _installed_handler(signal.SIGINT)
    exits = _record_exits(monkeypatch)

    with contextlib.suppress(_ReplacedExitError):
        handler(signal.SIGINT, None)

    assert exits == [128 + signal.SIGINT]


def test_install_termination_cleanup_when_multiple_registered_does_run_them_in_install_order(
    raise_signal: RaiseSignal,
):
    order = []
    install_termination_cleanup(lambda: order.append("first"))
    install_termination_cleanup(lambda: order.append("second"))
    install_termination_cleanup(lambda: order.append("third"))

    raise_signal(signal.SIGINT)

    assert order == ["first", "second", "third"]


@pytest.mark.parametrize(
    ("later_cleanup", "expected_calls"),
    [(False, []), (True, ["second"])],
    ids=["alone", "before-a-later-install"],
)
def test_install_termination_cleanup_when_uninstalled_does_exit_without_running_that_cleanup(
    raise_signal: RaiseSignal, later_cleanup: bool, expected_calls: list[str]
):
    calls = []
    install_termination_cleanup(lambda: calls.append("first"))()
    if later_cleanup:
        install_termination_cleanup(lambda: calls.append("second"))

    code = raise_signal(signal.SIGINT)

    assert calls == expected_calls
    assert code == 128 + signal.SIGINT


def test_install_termination_cleanup_when_second_signal_arrives_during_cleanup_does_escalate_instead_of_remaining_cleanups(
    raise_signal: RaiseSignal,
):
    calls = []
    signals.install_termination_escalation(lambda: calls.append("first escalation"))
    signals.install_termination_escalation(lambda: calls.append("second escalation"))

    def interrupt() -> None:
        calls.append("interrupted")
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    install_termination_cleanup(lambda: calls.append("skipped"))

    code = raise_signal(signal.SIGINT)

    assert calls == ["interrupted", "first escalation", "second escalation"]
    assert code == 128 + signal.SIGINT


def test_install_termination_cleanup_when_escalation_raises_does_exit_with_first_signal(
    raise_signal: RaiseSignal,
):
    calls = []
    signals.install_termination_escalation(_boom)
    signals.install_termination_escalation(lambda: calls.append("escalated"))

    def interrupt() -> None:
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    install_termination_cleanup(lambda: calls.append("skipped"))

    code = raise_signal(signal.SIGINT)

    assert calls == ["escalated"]
    assert code == 128 + signal.SIGINT


# ---------------------------------------------------------------------------
# Exit output written to stderr
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_stderr() -> RecordingStream:
    """A stderr stand-in that records each write the handler makes; tests install it on sys.stderr."""
    return RecordingStream()


@pytest.fixture
def stalled_stderr() -> _Terminal:
    """A terminal whose stream lock another thread holds mid-write for good; tests install its stream on sys.stderr."""
    # The raw write never returns, so the writer thread keeps the
    # BufferedWriter lock the way a write stalled by terminal flow control does.
    # It is never released: a handler that regressed to an unbounded wait then
    # stays stuck on the lock instead of waking once the test is torn down.
    entered = threading.Event()

    def stall() -> None:
        entered.set()
        threading.Event().wait()

    terminal = _buffered_terminal(stall)
    threading.Thread(target=terminal.write_progress, daemon=True).start()
    entered.wait(timeout=5)
    return terminal


@pytest.fixture
def interrupting_stderr() -> Iterator[_Terminal]:
    """A terminal whose first write sends SIGTERM to the main thread, then stalls."""
    # The handler writes its exit output off the main thread and waits for it
    # there, so a real signal aimed at the main thread lands inside that wait.
    # The write then stalls until teardown, so the wait cannot finish first.
    release = threading.Event()
    main_thread_id = threading.main_thread().ident
    assert main_thread_id is not None
    signal_sent = threading.Event()

    def interrupt_then_stall() -> None:
        if not signal_sent.is_set():
            signal_sent.set()
            signal.pthread_kill(main_thread_id, signal.SIGTERM)
            release.wait(timeout=10)

    yield _buffered_terminal(interrupt_then_stall)
    release.set()


def _nothing() -> None:
    return None


@pytest.mark.parametrize(
    ("cleanups", "writes"),
    [
        pytest.param(
            (_boom, _write_erase_notice),
            ["live display erased\ntermination cleanup failed: cleanup boom\n"],
            id="remaining-cleanups-run-then-failures-in-one-write",
        ),
        pytest.param((_warn_killpg_failed,), ["killpg failed\n"], id="a-cleanup-warns"),
        pytest.param((_nothing,), [], id="nothing-pending"),
    ],
)
def test_install_termination_cleanup_when_signal_received_does_write_pending_output_once(
    raise_signal: RaiseSignal,
    recording_stderr: RecordingStream,
    monkeypatch: pytest.MonkeyPatch,
    cleanups: tuple[Callable[[], None], ...],
    writes: list[str],
):
    monkeypatch.setattr(sys, "stderr", recording_stderr)
    for cleanup in cleanups:
        install_termination_cleanup(cleanup)

    raise_signal(signal.SIGINT)

    assert recording_stderr.writes == writes


def test_install_termination_cleanup_when_stderr_is_rich_proxy_does_write_to_underlying_stream(
    raise_signal: RaiseSignal, monkeypatch: pytest.MonkeyPatch
):
    underlying = io.StringIO()
    console_file = io.StringIO()
    console = Console(
        file=console_file, width=80, height=24, force_terminal=False, no_color=True, _environ={}
    )
    monkeypatch.setattr(sys, "stderr", FileProxy(console, underlying))
    install_termination_cleanup(_write_erase_notice)

    raise_signal(signal.SIGINT)

    assert underlying.getvalue() == "live display erased\n"
    assert console_file.getvalue() == ""


@pytest.fixture
def recorded_joins(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the exit-writer join with one that records its timeout and returns at once."""
    joins: list[float] = []

    def record_join(writer: threading.Thread, timeout_s: float) -> None:
        joins.append(timeout_s)

    monkeypatch.setattr(signals, "join_exit_writer", record_join)
    return joins


@pytest.fixture
def finished_exit_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Wrap the exit-writer join to record the timeout of each wait that ran to its end."""
    finished: list[float] = []
    real_join = signals.join_exit_writer

    def join_and_record(writer: threading.Thread, timeout_s: float) -> None:
        real_join(writer, timeout_s)
        finished.append(timeout_s)

    monkeypatch.setattr(signals, "join_exit_writer", join_and_record)
    return finished


@pytest.fixture
def endless_thread() -> Iterator[threading.Thread]:
    """A started daemon thread that does not finish before the test is torn down."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    yield thread
    release.set()


def test_join_exit_writer_when_writer_never_finishes_does_return_after_timeout(
    endless_thread: threading.Thread,
):
    started_at = time.perf_counter()

    _run_capped(lambda: signals.join_exit_writer(endless_thread, _SHORT_EXIT_WRITE_TIMEOUT_S))

    assert time.perf_counter() - started_at >= _SHORT_EXIT_WRITE_TIMEOUT_S


def test_install_termination_cleanup_when_stderr_write_stalls_does_exit_after_one_bounded_wait(
    raise_signal: RaiseSignal,
    stalled_stderr: _Terminal,
    recorded_joins: list[float],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(signals, "EXIT_WRITE_TIMEOUT_S", _SHORT_EXIT_WRITE_TIMEOUT_S)
    monkeypatch.setattr(sys, "stderr", stalled_stderr.stream)
    install_termination_cleanup(_write_erase_notice)

    code = _run_capped(lambda: raise_signal(signal.SIGINT))

    assert code == 128 + signal.SIGINT
    assert recorded_joins == [_SHORT_EXIT_WRITE_TIMEOUT_S]


def test_install_termination_cleanup_when_signal_interrupts_buffered_stderr_write_does_exit_after_one_bounded_wait(
    raise_signal: RaiseSignal, recorded_joins: list[float], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(signals, "EXIT_WRITE_TIMEOUT_S", _SHORT_EXIT_WRITE_TIMEOUT_S)
    codes: list[int] = []

    def interrupt_once() -> None:
        if not codes:
            codes.append(raise_signal(signal.SIGINT))

    terminal = _buffered_terminal(interrupt_once)
    monkeypatch.setattr(sys, "stderr", terminal.stream)
    install_termination_cleanup(_boom)

    _run_capped(terminal.write_progress)

    assert codes == [128 + signal.SIGINT]
    assert recorded_joins == [_SHORT_EXIT_WRITE_TIMEOUT_S]


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="pthread_kill is POSIX-only")
def test_install_termination_cleanup_when_second_signal_arrives_during_exit_output_write_does_exit_at_once_without_rerunning_cleanup(
    raise_signal: RaiseSignal,
    interrupting_stderr: _Terminal,
    finished_exit_waits: list[float],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "stderr", interrupting_stderr.stream)
    monkeypatch.setattr(signals, "EXIT_WRITE_TIMEOUT_S", _LONG_EXIT_WRITE_TIMEOUT_S)
    calls = []

    def erase_display() -> None:
        calls.append("cleanup")
        _write_erase_notice()

    install_termination_cleanup(erase_display)

    code = raise_signal(signal.SIGINT)

    assert calls == ["cleanup"]
    assert code == 128 + signal.SIGINT
    assert finished_exit_waits == []


@needs_signal_masking
def test_install_termination_cleanup_when_writing_exit_output_does_block_termination_signals_on_writer_thread(
    raise_signal: RaiseSignal, monkeypatch: pytest.MonkeyPatch
):
    blocked_during_write: list[set[int | signal.Signals]] = []
    terminal = _buffered_terminal(
        lambda: blocked_during_write.append(signal.pthread_sigmask(signal.SIG_BLOCK, []))
    )
    monkeypatch.setattr(sys, "stderr", terminal.stream)
    install_termination_cleanup(_write_erase_notice)

    raise_signal(signal.SIGINT)

    (blocked,) = blocked_during_write
    assert blocked >= signals.TERMINATION_SIGNALS


# ---------------------------------------------------------------------------
# Escalation on a second signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deliver",
    [
        pytest.param(_deliver, id="on-the-main-thread"),
        pytest.param(_deliver_from_thread, id="on-another-thread"),
    ],
)
def test_install_termination_escalation_when_later_signals_arrive_during_escalation_does_ignore_them(
    raise_signal: RaiseSignal, deliver: Callable[[int], None]
):
    calls = []

    def escalate_under_held_ctrl_c() -> None:
        calls.append("escalate")
        if len(calls) == 1:
            for signal_number in [signal.SIGINT, *_TERMINATION_SIGNALS, signal.SIGINT]:
                deliver(signal_number)
        calls.append("done")

    signals.install_termination_escalation(escalate_under_held_ctrl_c)
    _install_sigterm_on_cleanup(raise_signal)

    code = raise_signal(signal.SIGINT)

    assert calls == ["escalate", "done"]
    assert code == 128 + signal.SIGINT


@pytest.mark.parametrize(
    ("action", "expected_writes"),
    [
        pytest.param(lambda: None, [], id="succeeds"),
        pytest.param(
            _escalation_boom, ["termination escalation failed: escalation boom\n"], id="raises"
        ),
        pytest.param(_warn_killpg_failed, ["killpg failed\n"], id="warns"),
    ],
)
def test_install_termination_escalation_when_action_runs_does_write_only_its_failures_to_stderr(
    raise_signal: RaiseSignal,
    recording_stderr: RecordingStream,
    monkeypatch: pytest.MonkeyPatch,
    action: Callable[[], None],
    expected_writes: list[str],
):
    monkeypatch.setattr(sys, "stderr", recording_stderr)
    signals.install_termination_escalation(action)

    def erase_then_interrupt() -> None:
        _write_erase_notice()
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(erase_then_interrupt)

    raise_signal(signal.SIGINT)

    assert recording_stderr.writes == expected_writes


def test_install_termination_escalation_when_earlier_cleanup_failed_does_drop_its_failure(
    raise_signal: RaiseSignal, recording_stderr: RecordingStream, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sys, "stderr", recording_stderr)
    signals.install_termination_escalation(lambda: None)
    install_termination_cleanup(_boom)
    _install_sigterm_on_cleanup(raise_signal)

    raise_signal(signal.SIGINT)

    assert recording_stderr.writes == []


# ---------------------------------------------------------------------------
# Handler installation and idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal_number", _TERMINATION_SIGNALS, ids=_signal_id)
def test_install_termination_cleanup_when_cycled_does_keep_the_same_handler(signal_number: int):
    install_termination_cleanup(_nothing)
    handler = signal.getsignal(signal_number)

    for _ in range(12):
        install_termination_cleanup(_nothing)()

    assert signal.getsignal(signal_number) is handler
    assert callable(handler)
    assert handler not in {signal.SIG_DFL, signal.SIG_IGN}


def test_install_termination_cleanup_when_handler_replaced_after_first_install_does_keep_the_replacement():
    def replacement(signal_number: int, frame: FrameType | None) -> None:
        pass  # pragma: no cover — only installed, never delivered

    install_termination_cleanup(_nothing)
    signal.signal(signal.SIGINT, replacement)

    install_termination_cleanup(_nothing)

    assert signal.getsignal(signal.SIGINT) is replacement


# ---------------------------------------------------------------------------
# Deferring termination signals
# ---------------------------------------------------------------------------


def test_deferring_termination_signals_when_signal_state_replaced_mid_deferral_does_handle_next_signal_immediately(
    raise_signal: RaiseSignal, monkeypatch: pytest.MonkeyPatch
):
    with signals.deferring_termination_signals():
        monkeypatch.setattr(signals, "_state", signals._SignalState())
        install_termination_cleanup(_nothing)

        code = raise_signal(signal.SIGINT)

    assert code == 128 + signal.SIGINT


def test_deferring_termination_signals_when_deferral_outlives_signal_state_does_still_defer_on_new_state(
    monkeypatch: pytest.MonkeyPatch,
):
    stale = signals.deferring_termination_signals()
    stale.__enter__()
    monkeypatch.setattr(signals, "_state", signals._SignalState())
    install_termination_cleanup(_nothing)
    handler = _installed_handler(signal.SIGINT)
    exits = _record_exits(monkeypatch)
    stale.__exit__(None, None, None)

    with contextlib.suppress(_ReplacedExitError), signals.deferring_termination_signals():
        handler(signal.SIGINT, None)
        exits_while_open = list(exits)

    assert (exits_while_open, exits) == ([], [128 + signal.SIGINT])


def test_deferring_termination_signals_when_deferral_outlives_signal_state_does_not_replay_its_stored_signal(
    monkeypatch: pytest.MonkeyPatch,
):
    exits = _record_exits(monkeypatch)
    install_termination_cleanup(_nothing)
    handler = _installed_handler(signal.SIGINT)
    calls: list[str] = []
    stale = signals.deferring_termination_signals()
    stale.__enter__()
    try:
        handler(signal.SIGINT, None)
        monkeypatch.setattr(signals, "_state", signals._SignalState())
        install_termination_cleanup(lambda: calls.append("cleanup"))
    finally:
        # Entering the deferral changed this thread's signal mask, so it is
        # exited even when the setup above fails.
        with contextlib.suppress(_ReplacedExitError):
            stale.__exit__(None, None, None)

    assert (calls, exits) == ([], [])


@needs_signal_masking
def test_deferring_termination_signals_when_entered_does_block_them_until_exit():
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    with signals.deferring_termination_signals():
        during = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    assert during >= signals.TERMINATION_SIGNALS
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before


@needs_signal_masking
def test_deferring_termination_signals_when_overlapping_deferrals_exit_out_of_order_does_restore_the_mask_once_both_exit():
    # Two coroutines on one event loop overlap their deferrals this way: the one
    # that entered first exits first, while the other is still inside its own.
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    first = signals.deferring_termination_signals()
    second = signals.deferring_termination_signals()
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    while_second_open = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    second.__exit__(None, None, None)

    assert while_second_open >= signals.TERMINATION_SIGNALS
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before


@pytest.mark.parametrize(
    "exit_order",
    [("first", "second"), ("second", "first")],
    ids=["first-entered-exits-first", "last-entered-exits-first"],
)
def test_deferring_termination_signals_when_signal_arrives_while_overlapping_deferrals_open_does_handle_it_once_the_last_exits(
    monkeypatch: pytest.MonkeyPatch, exit_order: tuple[str, str]
):
    install_termination_cleanup(_nothing)
    handler = _installed_handler(signal.SIGINT)
    exits = _record_exits(monkeypatch)
    deferrals = {
        "first": signals.deferring_termination_signals(),
        "second": signals.deferring_termination_signals(),
    }
    deferrals["first"].__enter__()
    deferrals["second"].__enter__()
    handler(signal.SIGINT, None)
    earlier, last = (deferrals[name] for name in exit_order)
    earlier.__exit__(None, None, None)
    exits_while_one_open = list(exits)

    with contextlib.suppress(_ReplacedExitError):
        last.__exit__(None, None, None)

    assert (exits_while_one_open, exits) == ([], [128 + signal.SIGINT])


@pytest.fixture
def exploding_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``pthread_sigmask`` call in the signals module raise ``OSError``."""

    def explode(*args: object, **kwargs: object) -> None:
        message = "mask failed"
        raise OSError(message)

    monkeypatch.setattr(signals, "pthread_sigmask", explode)


@needs_signal_masking
@pytest.mark.usefixtures("exploding_mask")
def test_deferring_termination_signals_when_mask_raises_does_propagate_the_error():
    with pytest.raises(OSError, match="mask failed"):
        with signals.deferring_termination_signals():
            pass  # pragma: no cover — never reached


@needs_signal_masking
@pytest.mark.usefixtures("exploding_mask")
def test_deferring_termination_signals_when_mask_raised_does_handle_the_next_signal_immediately(
    raise_signal: RaiseSignal,
):
    cleaned: list[str] = []
    install_termination_cleanup(lambda: cleaned.append("cleanup"))
    with contextlib.suppress(OSError), signals.deferring_termination_signals():
        pass  # pragma: no cover — never reached

    code = raise_signal(signal.SIGINT)

    assert (code, cleaned) == (128 + signal.SIGINT, ["cleanup"])


# ---------------------------------------------------------------------------
# Signal state isolation between tests
# ---------------------------------------------------------------------------

# Runs under the project conftest in a child pytest, in file order: the first
# test leaves signal state behind, including a deferral it never exits, and the
# second exits that deferral inside one of its own and must still see empty
# signal state: no signal replayed when its quiet deferral exits, its own signal
# deferred until a second deferral exits, and a later signal escalating through
# its own escalation action alone.
_INNER_LEAKED_STATE_TESTS = """
import contextlib
import io
import signal
import sys

from gymrat import signals
from gymrat.signals import (
    install_termination_cleanup,
    install_termination_escalation,
    write_on_exit,
)

RAN = []
LEFT_OPEN = []


class _Exited(BaseException):
    pass


{polluter}


def test_starts_from_empty_signal_state(monkeypatch):
    exits = []

    def record_exit(code):
        exits.append(code)
        raise _Exited

    monkeypatch.setattr(signals, "exit_process", record_exit)
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    RAN.clear()
    install_termination_cleanup(lambda: RAN.append("fresh cleanup"))
    install_termination_escalation(lambda: RAN.append("fresh escalation"))
    stale = LEFT_OPEN.pop()

    with contextlib.suppress(_Exited), signals.deferring_termination_signals():
        stale.__exit__(None, None, None)
        RAN.append("stale exited")
    exits_after_quiet_deferral = list(exits)
    with contextlib.suppress(_Exited), signals.deferring_termination_signals():
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        RAN.append("deferred")
    with contextlib.suppress(_Exited):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    assert (exits_after_quiet_deferral, exits, RAN, stderr.getvalue()) == (
        [],
        [128 + signal.SIGINT, 128 + signal.SIGINT],
        ["stale exited", "deferred", "fresh cleanup", "fresh escalation"],
        "",
    )
"""

# Leaves cleanups, an escalation action, the handled and escalating flags,
# pending exit output and an open deferral behind.
_HANDLED_SIGNAL_POLLUTER = """
def test_leaves_handled_signal_state_behind(raise_signal):
    install_termination_escalation(lambda: RAN.append("leaked escalation"))

    def interrupt():
        RAN.append("leaked cleanup")
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    raise_signal(signal.SIGINT)
    install_termination_cleanup(lambda: RAN.append("leaked cleanup"))
    write_on_exit("leaked output\\n")
    deferral = signals.deferring_termination_signals()
    deferral.__enter__()
    LEFT_OPEN.append(deferral)
"""

# Leaves a cleanup and an open deferral holding a deferred signal behind.
_DEFERRED_SIGNAL_POLLUTER = """
def test_leaves_deferred_signal_behind():
    install_termination_cleanup(lambda: RAN.append("leaked cleanup"))
    deferral = signals.deferring_termination_signals()
    deferral.__enter__()
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    LEFT_OPEN.append(deferral)
"""


@pytest.mark.parametrize(
    "polluter",
    [
        pytest.param(_HANDLED_SIGNAL_POLLUTER, id="after-a-handled-signal"),
        pytest.param(_DEFERRED_SIGNAL_POLLUTER, id="after-a-deferred-signal"),
    ],
)
def test_signal_state_when_earlier_test_left_state_behind_does_start_next_test_empty(
    tmp_path: Path, polluter: str
):
    source = _INNER_LEAKED_STATE_TESTS.format(polluter=polluter)

    result = run_inner_pytest(tmp_path, source, inherited_env(), SINGLE_PROCESS_ARGS)

    assert (result.returncode, "2 passed" in result.stdout) == (0, True), result.stdout
