"""Skip marker for tests that rely on POSIX file mode bits being enforced.

A test that strips a read, write, or search bit expects the operating system to
refuse the access. Windows has no such bits for ``chmod`` to clear, and root
bypasses them, so on either the refusal never happens and the test cannot run.
"""

import os
import sys

import pytest

needs_mode_bits = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX mode bits: Windows lacks them and root bypasses them",
)
