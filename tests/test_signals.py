import signal
from collections.abc import Callable, Iterator

import pytest

from gymrat import signals
from gymrat.signals import install_termination_cleanup

# Invokes the handler installed for a signal and returns the code it would exit
# with. Supplied by the ``raise_signal`` fixture.
RaiseSignal = Callable[[int], int]

# Termination signals available on this platform. SIGHUP is POSIX-only; the
# win32 case (no SIGHUP) is exercised separately by simulating its absence.
_TERMINATION_SIGNALS = [signal.SIGINT, signal.SIGTERM]
if hasattr(signal, "SIGHUP"):
    _TERMINATION_SIGNALS.append(signal.SIGHUP)


def _signal_id(signal_number: int) -> str:
    return signal.Signals(signal_number).name


class _ProcessExitedError(Exception):
    """Raised by the stubbed exit seam so a handler unwinds where it would exit."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"_exit_process({code})")


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Keep module-global registry state isolated between tests.

    Saves the signal dispositions for every termination signal before the test,
    and restores them after — so a Ctrl-C during the suite reaches pytest's own
    handler instead of the gymrat handler installed by the test.
    """
    saved = {sig: signal.getsignal(sig) for sig in signals.TERMINATION_SIGNALS}
    yield
    signals.reset()
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture
def raise_signal(monkeypatch: pytest.MonkeyPatch) -> RaiseSignal:
    """Stub the exit seam and return a helper that invokes an installed handler.

    Emitting a real signal would take the test runner down, so the helper fetches
    the handler the module registered via ``signal.getsignal`` and calls it
    directly with ``(signal_number, frame)``. With the exit seam stubbed to raise, the
    handler unwinds exactly where the real one would exit, and the helper reports
    the code it would have exited with.
    """

    def fake_exit(code: int) -> None:
        raise _ProcessExitedError(code)

    monkeypatch.setattr(signals, "_exit_process", fake_exit)

    def _raise(signal_number: int) -> int:
        handler = signal.getsignal(signal_number)
        if not callable(handler):
            pytest.fail(f"no handler installed for signal {signal_number}")
        try:
            handler(signal_number, None)
        except _ProcessExitedError as exited:
            return exited.code
        pytest.fail("handler returned instead of exiting")

    return _raise


@pytest.mark.parametrize("signal_number", _TERMINATION_SIGNALS, ids=_signal_id)
def test_install_termination_cleanup_when_signal_received_does_run_cleanup_and_exit_128_plus_signal_number(
    raise_signal: RaiseSignal, signal_number: int
):
    calls = []
    install_termination_cleanup(lambda: calls.append("cleanup"))

    code = raise_signal(signal_number)

    assert calls == ["cleanup"]
    assert code == 128 + signal_number


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


def test_install_termination_cleanup_when_a_cleanup_raises_does_warn_and_run_remaining(
    raise_signal: RaiseSignal,
):
    survivors = []

    def boom() -> None:
        message = "cleanup boom"
        raise RuntimeError(message)

    install_termination_cleanup(boom)
    install_termination_cleanup(lambda: survivors.append("survivor"))

    with pytest.warns(UserWarning, match="cleanup boom"):
        code = raise_signal(signal.SIGINT)

    assert survivors == ["survivor"]
    assert code == 128 + signal.SIGINT


def test_install_termination_cleanup_when_second_signal_arrives_during_cleanup_does_not_reenter(
    raise_signal: RaiseSignal,
):
    calls = []
    reentered = False

    def cleanup() -> None:
        nonlocal reentered
        calls.append("cleanup")
        if not reentered:
            reentered = True
            raise_signal(signal.SIGINT)

    install_termination_cleanup(cleanup)

    code = raise_signal(signal.SIGINT)

    assert calls == ["cleanup"]
    assert code == 128 + signal.SIGINT


@pytest.mark.parametrize("signal_number", _TERMINATION_SIGNALS, ids=_signal_id)
def test_install_termination_cleanup_when_cycled_repeatedly_does_keep_exactly_one_handler(
    signal_number: int,
):
    install_termination_cleanup(lambda: None)()
    handler = signal.getsignal(signal_number)

    for _ in range(12):
        install_termination_cleanup(lambda: None)()

    assert signal.getsignal(signal_number) is handler
    assert callable(handler)


def test_install_termination_cleanup_when_sighup_undefined_does_register_only_available_signals(
    monkeypatch: pytest.MonkeyPatch, raise_signal: RaiseSignal
):
    monkeypatch.delattr(signal, "SIGHUP", raising=False)
    monkeypatch.setattr(signals, "_installed_signals", set())
    calls = []

    uninstall = install_termination_cleanup(lambda: calls.append("cleanup"))

    code = raise_signal(signal.SIGINT)
    assert calls == ["cleanup"]
    assert code == 128 + signal.SIGINT
    uninstall()


# ---------------------------------------------------------------------------
# reset — deferral state cleanup
# ---------------------------------------------------------------------------


def test_reset_when_deferral_active_does_clear_deferring_state():
    signals._deferring = True
    signals._deferred_signal = signal.SIGINT

    signals.reset()

    assert signals._deferring is False
    assert signals._deferred_signal is None


# ---------------------------------------------------------------------------
# deferring_termination_signals — mask failure safety
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
def test_deferring_termination_signals_when_mask_raises_does_not_strand_deferral(
    monkeypatch: pytest.MonkeyPatch,
):
    def exploding_mask(*args: object, **kwargs: object) -> None:
        message = "mask failed"
        raise OSError(message)

    monkeypatch.setattr(signals, "pthread_sigmask", exploding_mask)

    with pytest.raises(OSError, match="mask failed"):
        with signals.deferring_termination_signals():
            pass  # pragma: no cover — never reached

    assert signals._deferring is False
