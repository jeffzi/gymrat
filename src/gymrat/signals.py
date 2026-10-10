"""Termination-cleanup registry for graceful shutdown on POSIX signals.

Callers register a zero-argument cleanup with :func:`install_termination_cleanup`
and receive an uninstall callable. When the process receives ``SIGINT``,
``SIGTERM``, or ``SIGHUP``, every active cleanup runs in install order and the
process then exits with ``128 + signal_number`` — the shell convention for
"terminated by signal N".

A second termination signal while the first is still being handled escalates:
every action registered with :func:`install_termination_escalation` runs at
once, the remaining cleanups and the pending exit output are skipped, and the
process exits with ``128 +`` the number of the *first* signal. This holds from
the first cleanup through the final stderr write, so a user pressing Ctrl-C
twice never waits out a cleanup's grace period and never re-runs a cleanup.
The escalation runs once: a third or later signal is ignored, so a held Ctrl-C
neither re-runs the actions nor delays the exit. What the actions raise or warn
is written to stderr the same bounded way as the cleanups' output.

The handler never writes to stderr on the main thread. A signal can land while
the main thread is itself inside a stderr write, where a second write on the
same thread raises a reentrant-call ``RuntimeError``, and a terminal under flow
control can stall a write indefinitely. Warnings raised during the cleanups and
any output a cleanup hands over through :func:`write_on_exit` are collected
instead, then written once from a daemon thread that the handler waits on for a
bounded time before exiting regardless.

The handler is installed once per signal for the lifetime of the process and is
deliberately never restored. Python allows exactly one handler per signal, so a
single module-level handler owns each termination signal and consults the live
registry every time it fires; installing and uninstalling cleanups only mutates
that registry, never the signal disposition.
"""

import os
import signal
import sys
import threading
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from types import FrameType
from typing import NoReturn, TextIO

# Termination signals gymrat installs cleanup for. SIGHUP is POSIX-only and
# absent on win32, so each name is resolved defensively and any the platform
# does not define is dropped. This is the canonical set: :mod:`gymrat.exec`
# imports it to unblock exactly these signals in a spawned child, reversing the
# mask the parent holds across the spawn.
TERMINATION_SIGNALS: frozenset[int] = frozenset(
    resolved
    for name in ("SIGINT", "SIGTERM", "SIGHUP")
    if (resolved := getattr(signal, name, None)) is not None
)

# POSIX-only seam for blocking signals. ``None`` on platforms without
# ``pthread_sigmask`` (win32), where callers fall back to running unmasked.
# Kept as a module-level reference so the fallback branch stays testable.
# :mod:`gymrat.exec` imports this to unblock the same signals in a spawned
# child, rather than re-resolving ``pthread_sigmask`` itself.
pthread_sigmask: Callable[[int, Iterable[int]], set[int]] | None = getattr(
    signal, "pthread_sigmask", None
)

# Live cleanups keyed by an opaque install token. A dict preserves insertion
# order, which the handler relies on to run cleanups in install order; a set
# would not.
_registry: dict[object, Callable[[], None]] = {}

# Signals whose handler is already wired up. The disposition is set once and
# reused, so re-installing must not re-register it.
_installed_signals: set[int] = set()

# Actions a second termination signal runs before exiting, in registration
# order. A dict keyed by the action keeps that order and makes registering the
# same action again a no-op.
_escalations: dict[Callable[[], None], None] = {}

# The signal being handled, from its first cleanup until the process exits. Any
# signal arriving while it is set escalates instead of re-entering the cleanups.
_handled_signal: int | None = None

# Set once a second signal starts the escalation. The handler ignores every
# signal while it is set, so the escalation actions run exactly once. A flag
# rather than a signal mask: CPython runs the handler on the main thread even
# when a non-main thread receives the signal, whatever the main thread's mask.
_escalating: bool = False

# Python-level deferral state. In a multi-threaded program, ``pthread_sigmask``
# blocks OS delivery to the main thread, but a non-main thread's C handler can
# still set the pending-signal flag. ``time.sleep`` and asyncio event loops call
# ``PyErr_CheckSignals()`` which processes that flag regardless of the mask.
# While any deferral is open the Python-level handler stores the signal instead
# of processing it; the stored signal is replayed once the last one closes.
#
# A count rather than a flag: deferrals overlap without nesting when two
# coroutines on one event loop each hold one across an ``await``, so the first
# to exit is not necessarily the last one open. The lock guards the count
# against deferrals on other threads; the handler only reads it, so a signal
# landing while this thread holds the lock cannot deadlock.
_open_deferrals: int = 0
_open_deferrals_lock = threading.Lock()
_deferred_signal: int | None = None


class _ThreadMaskHold(threading.local):
    """This thread's open deferrals, and the signal mask the first one replaced.

    ``pthread_sigmask`` is per thread, so the mask is blocked by the first
    deferral a thread opens and restored only when its last one closes, in
    whatever order they close.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.saved_mask: set[int] = set()


_thread_mask_hold = _ThreadMaskHold()

# Terminal output cleanups hand over through ``write_on_exit``, written ahead of
# the collected warnings once every cleanup has run.
_exit_output: list[str] = []

EXIT_WRITE_TIMEOUT_S: float = 1.0
"""Seconds the handler waits on the exit-output writer thread before exiting anyway.

A terminal stalled by flow control delays exit by at most this long. It must
stay bounded: an unbounded wait lets a stalled stderr keep the process alive
past a termination signal.

Test seam: the handler reads it at call time, so a replacement installed after
the handlers are wired up still takes effect.
"""

# Shell convention: a process terminated by signal N exits with 128 + N.
_SIGNAL_EXIT_BASE = 128


def reset() -> None:
    """Clear the registries, installed-signal set, pending exit output, and flags.

    Test-only seam: production code never calls this, since handlers are wired
    up once for the process lifetime and deliberately never torn down. Tests use
    it to isolate module-global state between cases instead of reaching into the
    private attributes directly.
    """
    global _handled_signal, _escalating, _open_deferrals, _deferred_signal  # noqa: PLW0603 - module-level state the reset owns
    _registry.clear()
    _escalations.clear()
    _handled_signal = None
    _escalating = False
    _open_deferrals = 0
    _deferred_signal = None
    _installed_signals.clear()
    _exit_output.clear()


def exit_process(code: int) -> NoReturn:
    """Terminate the process immediately with ``code``.

    Uses ``os._exit`` rather than ``sys.exit``: the cleanups have already run,
    and raising ``SystemExit`` from a signal handler could be swallowed by an
    application ``except`` block, leaving the process alive after a termination
    signal.

    Test seam: tests replace this function to observe the exit code instead of
    exiting. The termination handler looks it up at call time, so a replacement
    installed after the handlers are wired up still takes effect.

    Args:
        code: The process exit status.
    """
    os._exit(code)


def join_exit_writer(writer: threading.Thread, timeout_s: float) -> None:
    """Wait for the exit-output writer thread, giving up after ``timeout_s``.

    Returns after at most ``timeout_s``, whether or not the writer finished: a
    write stalled by terminal flow control must not hold the process past a
    termination signal.

    Test seam: tests replace this function to observe the wait instead of
    sitting through it. The termination handler looks it up at call time, so a
    replacement installed after the handlers are wired up still takes effect.

    Args:
        writer: The started thread writing the exit output.
        timeout_s: Longest the wait may last, in seconds.
    """
    writer.join(timeout_s)


def write_on_exit(text: str) -> None:
    """Hand the termination handler terminal output to write before it exits.

    Call this from a termination cleanup instead of writing to stderr directly:
    the handler writes every contributed text, in call order and ahead of any
    collected warnings, in a single bounded write once all cleanups have run.

    Args:
        text: Output written verbatim, so it carries its own trailing newline or
            escape sequences.
    """
    _exit_output.append(text)


def _run_collecting_warnings(callables: list[Callable[[], None]], kind: str) -> list[str]:
    """Run each callable in order, collecting what they warn or raise.

    Warnings are recorded rather than displayed, since displaying one writes to
    stderr on the main thread.

    Args:
        callables: The zero-argument callables to run.
        kind: What the callables are, naming them in a failure's text.

    Returns:
        The text of every warning the callables emitted, and of every failure,
        in the order they occurred.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for run in callables:
            try:
                run()
            except Exception as exc:  # noqa: BLE001 - a failing callable must not stop the rest
                warnings.warn(f"termination {kind} failed: {exc}", RuntimeWarning, stacklevel=2)
    return [str(warning.message) for warning in caught]


def _unwrapped_stderr() -> TextIO:
    # A rich live display swaps sys.stderr for a FileProxy that routes writes
    # through its console; the exit output must reach the terminal directly.
    stream = sys.stderr
    while (proxied := getattr(stream, "rich_proxied_file", None)) is not None:
        stream = proxied
    return stream


def _write_exit_output(warning_texts: list[str]) -> None:
    text = "".join(_exit_output) + "".join(f"{warning}\n" for warning in warning_texts)
    _exit_output.clear()
    if not text:
        return
    stream = _unwrapped_stderr()

    def write() -> None:
        # The kernel hands a process-directed signal to any thread not blocking
        # it. Landing here, CPython would only flag it for the main thread and
        # leave that thread waiting out the whole timeout below; blocked here,
        # it reaches the main thread and interrupts the wait at once.
        if pthread_sigmask is not None:
            pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
        stream.write(text)
        stream.flush()

    writer = threading.Thread(target=write, name="gymrat-exit-output", daemon=True)
    writer.start()
    join_exit_writer(writer, EXIT_WRITE_TIMEOUT_S)


def _escalate(first_signal: int) -> NoReturn:
    global _escalating  # noqa: PLW0603 - module-level state the escalation owns
    _escalating = True
    _exit_output.clear()
    _write_exit_output(_run_collecting_warnings(list(_escalations), "escalation"))
    exit_process(_SIGNAL_EXIT_BASE + first_signal)


def _handler(signal_number: int, _frame: FrameType | None) -> None:
    global _handled_signal, _deferred_signal  # noqa: PLW0603 - module-level state the handler owns

    if _escalating:
        return

    if _open_deferrals:
        _deferred_signal = signal_number
        return

    if _handled_signal is not None:
        _escalate(_handled_signal)

    _handled_signal = signal_number
    _write_exit_output(_run_collecting_warnings(list(_registry.values()), "cleanup"))
    exit_process(_SIGNAL_EXIT_BASE + signal_number)


def _ensure_handlers_installed() -> None:
    for signal_number in TERMINATION_SIGNALS - _installed_signals:
        signal.signal(signal_number, _handler)
        _installed_signals.add(signal_number)


def _hold_thread_mask() -> bool:
    """Block termination signals on this thread unless a deferral here already does.

    Returns:
        Whether this call counted towards the thread's hold, and so must be
        matched by :func:`_release_thread_mask`.
    """
    if pthread_sigmask is None:
        return False
    hold = _thread_mask_hold
    if hold.depth == 0:
        hold.saved_mask = pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    hold.depth += 1
    return True


def _release_thread_mask() -> None:
    hold = _thread_mask_hold
    hold.depth -= 1
    if hold.depth == 0 and pthread_sigmask is not None:
        pthread_sigmask(signal.SIG_SETMASK, hold.saved_mask)


@contextmanager
def deferring_termination_signals() -> Generator[None]:
    """Defer termination signals for the duration of the wrapped call.

    A termination signal delivered while the wrapped code is running must not
    fire the process's termination cleanup mid-call. The deferral works at two
    levels: ``pthread_sigmask`` blocks OS-level delivery to the calling thread
    (where available), and a Python-level count of open deferrals makes the
    handler store the signal instead of processing it. The second level is
    needed because in multi-threaded programs a non-main thread can receive the
    OS signal, set CPython's pending-signal flag, and ``PyErr_CheckSignals()``
    on the main thread then runs the handler despite the mask.

    Deferrals may overlap in any order, as two coroutines on one event loop do
    when each holds one across an ``await``: the thread's mask is restored when
    its last open deferral exits, and any signal stored meanwhile is replayed
    when the last open deferral in the process exits, so a registered handler
    runs only after every wrapped call completes.
    """
    global _open_deferrals, _deferred_signal  # noqa: PLW0603 - module-level state the deferral owns

    with _open_deferrals_lock:
        _open_deferrals += 1
    holding_mask = False
    try:
        holding_mask = _hold_thread_mask()
        yield
    finally:
        with _open_deferrals_lock:
            # Floored at zero: reset() may have cleared the count while this
            # deferral was still open.
            _open_deferrals = max(0, _open_deferrals - 1)
            last_open = _open_deferrals == 0
        if holding_mask:
            _release_thread_mask()
        if last_open:
            deferred = _deferred_signal
            _deferred_signal = None
            if deferred is not None:
                _handler(deferred, None)


def install_termination_cleanup(cleanup: Callable[[], None]) -> Callable[[], None]:
    """Register a cleanup to run when the process is terminated by a signal.

    On the first call the module wires a handler onto every termination signal
    the platform defines (``SIGINT``, ``SIGTERM``, and ``SIGHUP`` where
    available). On ``SIGINT``, ``SIGTERM``, or ``SIGHUP`` every active cleanup
    runs in install order and the process exits with ``128 + signal_number``.

    Args:
        cleanup: A zero-argument callable invoked during shutdown. An exception
            it raises, and any warning it emits, is written to stderr before
            the process exits and does not stop the remaining cleanups.

    Returns:
        An uninstall callable that removes this cleanup from the registry.
        Calling it more than once is harmless.
    """
    _ensure_handlers_installed()

    token = object()
    _registry[token] = cleanup

    def uninstall() -> None:
        _registry.pop(token, None)

    return uninstall


def install_termination_escalation(action: Callable[[], None]) -> None:
    """Register an action to run at once when a second termination signal arrives.

    A second signal while the first is still being handled, whether during a
    cleanup or during the final stderr write, runs every registered action in
    registration order, skips whatever the first signal had left to do, and
    exits with ``128 +`` the first signal's number. The escalation runs once: a
    third or later signal that arrives while it runs is ignored. An action is
    the fast, forceful form of a cleanup's slow one, such as killing child
    processes outright instead of granting them a grace period. Registering the
    same action again is a no-op; the registration lasts for the process
    lifetime.

    Args:
        action: A zero-argument callable. An exception it raises, and any
            warning it emits, is written to stderr before the process exits and
            does not stop the remaining actions.
    """
    _escalations[action] = None
