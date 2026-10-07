"""Unit tests for the Windows branch of ``shell_quote_name`` in ``gymrat.loop.iterate.confirm``."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from gymrat.loop.iterate.confirm import shell_quote_name

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# shell_quote_name on Windows
# ---------------------------------------------------------------------------


def test_shell_quote_name_when_win32_does_wrap_the_name_in_double_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    value = "decode large payload"

    result = shell_quote_name(value)

    assert result == '"decode large payload"'
