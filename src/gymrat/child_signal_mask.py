"""Lift, in a freshly forked child, the termination-signal mask its parent holds."""

import signal

from gymrat.signals import TERMINATION_SIGNALS, pthread_sigmask


def unblock_termination_signals() -> None:
    """Unblock the termination signals in this process, a no-op where there is no signal mask.

    Passed as a child's ``preexec_fn`` so a child spawned while the parent defers
    termination signals can still be ended by one.
    """
    if pthread_sigmask is not None:
        pthread_sigmask(signal.SIG_UNBLOCK, TERMINATION_SIGNALS)
