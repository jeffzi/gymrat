import contextlib
import dataclasses
import io
import signal
import sys
import threading
import time
import warnings
from collections.abc import Buffer, Callable, Iterator
from typing import NoReturn, override

import pytest
from rich.console import Console
from rich.file_proxy import FileProxy

from gymrat import signals
from gymrat.signals import install_termination_cleanup
from tests._streams import RecordingStream

# Invokes the handler installed for a signal and returns the code it would exit
# with. Supplied by the ``raise_signal`` fixture.
RaiseSignal = Callable[[int], int]

# Termination signals available on this platform. SIGHUP is POSIX-only.
_TERMINATION_SIGNALS = [signal.SIGINT, signal.SIGTERM]
if hasattr(signal, "SIGHUP"):
    _TERMINATION_SIGNALS.append(signal.SIGHUP)

# The contract is an exit about one second after the cleanups when the stderr
# write stalls: never sooner than that second, and clearly before a doubled wait.
_EXIT_WAIT_FLOOR_S = 1.0
_EXIT_WAIT_CEILING_S = 1.9

# A second signal during the stderr write exits at once: far below the one
# second the stalled write would otherwise hold the exit for.
_SECOND_SIGNAL_EXIT_CEILING_S = 0.5

# Longest a scenario run by ``_run_capped`` may take: well past the one-second
# exit wait, well short of a hang.
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


def _deliver(signal_number: int) -> None:
    # Runs the installed handler the way a signal arriving now would, without
    # the exit bookkeeping ``raise_signal`` adds: an ignored signal returns.
    handler = signal.getsignal(signal_number)
    assert callable(handler)
    handler(signal_number, None)


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


@dataclasses.dataclass(frozen=True, slots=True)
class _InterruptingTerminal(_Terminal):
    """A terminal whose first write signals the main thread, recording when it did."""

    signal_sent_at: list[float]


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
def test_install_termination_cleanup_when_signal_received_does_run_cleanup_and_exit_128_plus_signal_number(
    raise_signal: RaiseSignal, signal_number: int
):
    calls = []
    install_termination_cleanup(lambda: calls.append("cleanup"))

    code = raise_signal(signal_number)

    assert calls == ["cleanup"]
    assert code == 128 + signal_number


class _ReplacedExitError(BaseException):
    """Unwinds the handler where the replaced exit seam would end the process."""


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
        nested = signal.getsignal(second_signal)
        if not callable(nested):
            pytest.fail(f"no handler installed for signal {second_signal}")
        nested(second_signal, None)

    install_termination_cleanup(interrupt)
    handler = signal.getsignal(signal.SIGINT)
    if not callable(handler):
        pytest.fail("no handler installed for SIGINT")
    exits: list[int] = []

    def record_exit(code: int) -> NoReturn:
        exits.append(code)
        raise _ReplacedExitError

    monkeypatch.setattr(signals, "exit_process", record_exit)

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


def test_install_termination_cleanup_when_uninstalled_does_exit_without_running_cleanup(
    raise_signal: RaiseSignal,
):
    calls = []
    uninstall = install_termination_cleanup(lambda: calls.append("cleanup"))
    uninstall()

    code = raise_signal(signal.SIGINT)

    assert calls == []
    assert code == 128 + signal.SIGINT


def test_install_termination_cleanup_when_prior_run_uninstalled_does_run_only_later_cleanup(
    raise_signal: RaiseSignal,
):
    calls = []
    install_termination_cleanup(lambda: calls.append("first"))()
    install_termination_cleanup(lambda: calls.append("second"))

    raise_signal(signal.SIGINT)

    assert calls == ["second"]


def test_install_termination_cleanup_when_a_cleanup_raises_does_run_remaining_cleanups(
    raise_signal: RaiseSignal,
):
    survivors = []
    install_termination_cleanup(_boom)
    install_termination_cleanup(lambda: survivors.append("survivor"))

    code = raise_signal(signal.SIGINT)

    assert survivors == ["survivor"]
    assert code == 128 + signal.SIGINT


def test_install_termination_cleanup_when_second_signal_arrives_during_cleanup_does_escalate_instead_of_remaining_cleanups(
    raise_signal: RaiseSignal,
):
    calls = []
    signals.install_termination_escalation(lambda: calls.append("escalate"))

    def interrupt() -> None:
        calls.append("interrupted")
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    install_termination_cleanup(lambda: calls.append("skipped"))

    code = raise_signal(signal.SIGINT)

    assert calls == ["interrupted", "escalate"]
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
def interrupting_stderr() -> Iterator[_InterruptingTerminal]:
    """A terminal whose first write sends SIGTERM to the main thread, then stalls."""
    # The handler writes its exit output off the main thread and waits for it
    # there, so a real signal aimed at the main thread lands inside that wait.
    # The write then stalls until teardown, so the wait cannot finish first.
    release = threading.Event()
    main_thread_id = threading.main_thread().ident
    assert main_thread_id is not None
    signal_sent_at: list[float] = []

    def interrupt_then_stall() -> None:
        if not signal_sent_at:
            signal_sent_at.append(time.monotonic())
            signal.pthread_kill(main_thread_id, signal.SIGTERM)
            release.wait(timeout=10)

    terminal = _buffered_terminal(interrupt_then_stall)
    yield _InterruptingTerminal(terminal.stream, terminal.raw, signal_sent_at)
    release.set()


def _nothing() -> None:
    return None


@pytest.mark.parametrize(
    ("cleanups", "writes"),
    [
        pytest.param(
            (_boom, _write_erase_notice),
            ["live display erased\ntermination cleanup failed: cleanup boom\n"],
            id="output-then-failures-in-one-write",
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


def test_install_termination_cleanup_when_stderr_write_stalls_does_exit_after_bounded_wait(
    raise_signal: RaiseSignal, stalled_stderr: _Terminal, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sys, "stderr", stalled_stderr.stream)
    cleanups_done_at: list[float] = []

    def erase_display() -> None:
        _write_erase_notice()
        cleanups_done_at.append(time.monotonic())

    install_termination_cleanup(erase_display)

    code = _run_capped(lambda: raise_signal(signal.SIGINT))

    exit_wait = time.monotonic() - cleanups_done_at[0]
    assert code == 128 + signal.SIGINT
    assert _EXIT_WAIT_FLOOR_S <= exit_wait < _EXIT_WAIT_CEILING_S


def test_install_termination_cleanup_when_signal_interrupts_buffered_stderr_write_does_exit_after_bounded_wait(
    raise_signal: RaiseSignal, monkeypatch: pytest.MonkeyPatch
):
    cleanups_done_at: list[float] = []
    exits: list[tuple[int, float]] = []

    def interrupt_once() -> None:
        if not exits:
            code = raise_signal(signal.SIGINT)
            exits.append((code, time.monotonic()))

    def note_cleanups_done() -> None:
        cleanups_done_at.append(time.monotonic())

    terminal = _buffered_terminal(interrupt_once)
    monkeypatch.setattr(sys, "stderr", terminal.stream)
    install_termination_cleanup(_boom)
    install_termination_cleanup(note_cleanups_done)

    _run_capped(terminal.write_progress)

    ((code, exited_at),) = exits
    assert code == 128 + signal.SIGINT
    assert _EXIT_WAIT_FLOOR_S <= exited_at - cleanups_done_at[0] < _EXIT_WAIT_CEILING_S


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="pthread_kill is POSIX-only")
def test_install_termination_cleanup_when_second_signal_arrives_during_exit_output_write_does_exit_at_once_without_rerunning_cleanup(
    raise_signal: RaiseSignal,
    interrupting_stderr: _InterruptingTerminal,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "stderr", interrupting_stderr.stream)
    calls = []

    def erase_display() -> None:
        calls.append("cleanup")
        _write_erase_notice()

    install_termination_cleanup(erase_display)

    code = raise_signal(signal.SIGINT)

    exit_delay = time.monotonic() - interrupting_stderr.signal_sent_at[0]
    assert calls == ["cleanup"]
    assert code == 128 + signal.SIGINT
    assert exit_delay < _SECOND_SIGNAL_EXIT_CEILING_S


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
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


def test_reset_when_escalation_already_ran_does_escalate_once_on_next_signal_pair(
    raise_signal: RaiseSignal,
):
    signals.install_termination_escalation(lambda: None)
    _install_sigterm_on_cleanup(raise_signal)
    raise_signal(signal.SIGINT)
    signals.reset()
    calls = []

    def escalate_then_receive_another_signal() -> None:
        calls.append("escalate")
        if len(calls) == 1:
            _deliver(signal.SIGINT)
        calls.append("done")

    def clean_up_then_interrupt() -> None:
        calls.append("cleanup")
        raise_signal(signal.SIGTERM)

    signals.install_termination_escalation(escalate_then_receive_another_signal)
    install_termination_cleanup(clean_up_then_interrupt)

    code = raise_signal(signal.SIGINT)

    assert calls == ["cleanup", "escalate", "done"]
    assert code == 128 + signal.SIGINT


# ---------------------------------------------------------------------------
# Handler installation and idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal_number", _TERMINATION_SIGNALS, ids=_signal_id)
def test_install_termination_cleanup_when_installed_again_does_keep_the_same_handler(
    signal_number: int,
):
    install_termination_cleanup(_nothing)
    handler = signal.getsignal(signal_number)

    install_termination_cleanup(_nothing)

    assert signal.getsignal(signal_number) is handler
    assert callable(handler)
    assert handler not in {signal.SIG_DFL, signal.SIG_IGN}


@pytest.mark.parametrize("signal_number", _TERMINATION_SIGNALS, ids=_signal_id)
def test_install_termination_cleanup_when_cycled_does_keep_the_same_handler(signal_number: int):
    install_termination_cleanup(_nothing)
    handler = signal.getsignal(signal_number)

    for _ in range(12):
        install_termination_cleanup(_nothing)()

    assert signal.getsignal(signal_number) is handler


# ---------------------------------------------------------------------------
# Deferring termination signals
# ---------------------------------------------------------------------------


def test_reset_when_called_during_deferral_does_handle_next_signal_immediately(
    raise_signal: RaiseSignal,
):
    install_termination_cleanup(lambda: None)

    with signals.deferring_termination_signals():
        signals.reset()

        code = raise_signal(signal.SIGINT)

    assert code == 128 + signal.SIGINT


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
def test_deferring_termination_signals_when_entered_does_block_them_until_exit():
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    with signals.deferring_termination_signals():
        during = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    assert during >= signals.TERMINATION_SIGNALS
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
def test_deferring_termination_signals_when_mask_raises_does_handle_the_next_signal_immediately(
    monkeypatch: pytest.MonkeyPatch, raise_signal: RaiseSignal
):
    cleaned: list[str] = []
    install_termination_cleanup(lambda: cleaned.append("cleanup"))

    def exploding_mask(*args: object, **kwargs: object) -> None:
        message = "mask failed"
        raise OSError(message)

    monkeypatch.setattr(signals, "pthread_sigmask", exploding_mask)
    with pytest.raises(OSError, match="mask failed"):
        with signals.deferring_termination_signals():
            pass  # pragma: no cover — never reached

    code = raise_signal(signal.SIGINT)

    assert (code, cleaned) == (128 + signal.SIGINT, ["cleanup"])
