"""Skip marker for tests that block termination signals with ``pthread_sigmask``.

Signal masking is POSIX-only: Windows has no ``signal.pthread_sigmask``, so a
test that inspects or relies on a blocked mask cannot run there.
"""

import signal

import pytest

needs_signal_masking = pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
