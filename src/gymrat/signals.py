"""Termination-cleanup registry for graceful shutdown on POSIX signals.

Callers register a zero-argument cleanup with :func:`install_termination_cleanup`
and receive an uninstall callable. When the process receives ``SIGINT``,
``SIGTERM``, or ``SIGHUP``, every active cleanup runs in install order and the
process then exits with ``128 + signal_number`` — the shell convention for
"terminated by signal N".

The handler is installed once per signal for the lifetime of the process and is
deliberately never restored. Python allows exactly one handler per signal, so a
single module-level handler owns each termination signal and consults the live
registry every time it fires; installing and uninstalling cleanups only mutates
that registry, never the signal disposition.
"""

import os
import signal
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import NoReturn

# Termination signals gymrat installs cleanup for, in the order the handler is
# wired up. SIGHUP is POSIX-only and absent on win32, so each name is resolved
# defensively.
_TERMINATION_SIGNAL_NAMES = ("SIGINT", "SIGTERM", "SIGHUP")

# The same signals resolved to their numbers, dropping any the platform does not
# define. This is the canonical set: :mod:`gymrat.git` imports it to block
# exactly these signals across a git subprocess call, so a signal cannot fire
# this module's cleanup while a ``git worktree add`` is only half-materialized.
TERMINATION_SIGNALS: frozenset[int] = frozenset(
    resolved
    for name in _TERMINATION_SIGNAL_NAMES
    if (resolved := getattr(signal, name, None)) is not None
)

# Live cleanups keyed by an opaque install token. A dict preserves insertion
# order, which the handler relies on to run cleanups in install order; a set
# would not.
_registry: dict[object, Callable[[], None]] = {}

# Signals whose handler is already wired up. The disposition is set once and
# reused, so re-installing must not re-register it.
_installed_signals: set[int] = set()

# True while the handler is draining the registry. A second signal arriving
# mid-drain must exit immediately rather than re-enter the cleanups.
_handling = False

# Python-level deferral state. In a multi-threaded program, ``pthread_sigmask``
# blocks OS delivery to the main thread, but a non-main thread's C handler can
# still set the pending-signal flag. ``time.sleep`` and asyncio event loops call
# ``PyErr_CheckSignals()`` which processes that flag regardless of the mask.
# The deferral flag makes the Python-level handler store the signal instead of
# processing it; the stored signal is replayed when deferral ends.
_deferring: bool = False
_deferred_signal: int | None = None


def reset() -> None:
    """Clear the registry, installed-signal set, and in-handler flag.

    Test-only seam: production code never calls this, since handlers are wired
    up once for the process lifetime and deliberately never torn down. Tests use
    it to isolate module-global state between cases instead of reaching into the
    private attributes directly.
    """
    global _handling, _deferring, _deferred_signal  # noqa: PLW0603 - module-level state the reset owns
    _registry.clear()
    _handling = False
    _deferring = False
    _deferred_signal = None
    _installed_signals.clear()


def _exit_process(code: int) -> NoReturn:
    """Terminate the process immediately with ``code``.

    Uses ``os._exit`` rather than ``sys.exit``: the cleanups have already run,
    and raising ``SystemExit`` from a signal handler could be swallowed by an
    application ``except`` block, leaving the process alive after a termination
    signal. Tests monkeypatch this seam to observe the code instead of exiting.
    """
    os._exit(code)


def _run_cleanups() -> None:
    """Run every registered cleanup in install order, warning on failures."""
    for cleanup in list(_registry.values()):
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001 - a failing cleanup must not stop the rest
            warnings.warn(f"termination cleanup failed: {exc}", RuntimeWarning, stacklevel=2)


def _handler(signal_number: int, _frame: FrameType | None) -> None:
    global _handling, _deferred_signal  # noqa: PLW0603 - module-level state the handler owns

    if _deferring:
        _deferred_signal = signal_number
        return

    if not _handling:
        _handling = True
        try:
            _run_cleanups()
        finally:
            _handling = False

    _exit_process(128 + signal_number)


def _ensure_handlers_installed() -> None:
    for signal_number in TERMINATION_SIGNALS - _installed_signals:
        signal.signal(signal_number, _handler)
        _installed_signals.add(signal_number)


# POSIX-only seam for blocking signals. ``None`` on platforms without
# ``pthread_sigmask`` (win32), where callers fall back to running unmasked.
# Kept as a module-level reference so the fallback branch stays testable.
# :mod:`gymrat.git` imports this to block the same signals across a git
# subprocess call, rather than re-resolving ``pthread_sigmask`` itself.
pthread_sigmask: Callable[[int, Iterable[int]], list[int]] | None = getattr(
    signal, "pthread_sigmask", None
)


@contextmanager
def deferring_termination_signals() -> Iterator[None]:
    """Defer termination signals for the duration of the wrapped call.

    A termination signal delivered while the wrapped code is running must not
    fire the process's termination cleanup mid-call. The deferral works at two
    levels: ``pthread_sigmask`` blocks OS-level delivery to the main thread
    (where available), and a Python-level flag makes the handler store the
    signal instead of processing it. The second level is needed because in
    multi-threaded programs a non-main thread can receive the OS signal, set
    CPython's pending-signal flag, and ``PyErr_CheckSignals()`` on the main
    thread then runs the handler despite the mask.

    Any signal stored during deferral is replayed when the context exits, so a
    registered handler runs only after the wrapped code completes.
    """
    global _deferring, _deferred_signal  # noqa: PLW0603

    previous: list[int] | None = None
    try:
        _deferring = True
        if pthread_sigmask is not None and TERMINATION_SIGNALS:
            previous = pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
        yield
    finally:
        _deferring = False
        if previous is not None and pthread_sigmask is not None:
            pthread_sigmask(signal.SIG_SETMASK, previous)
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
            it raises is reported as a warning and does not stop the remaining
            cleanups.

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
