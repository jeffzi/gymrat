"""Unit tests for the Windows branch of ``shell_quote_name`` in ``gymrat.loop.iterate.judge``."""

from __future__ import annotations

import sys

import pytest

from gymrat.loop.iterate.judge import shell_quote_name

# ---------------------------------------------------------------------------
# shell_quote_name on Windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("decode large payload", '"decode large payload"', id="spaces-quoted"),
        pytest.param("total_ms", "total_ms", id="bare-word-unquoted"),
    ],
)
def test_shell_quote_name_when_win32_does_quote_only_names_cmd_would_split(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    result = shell_quote_name(value)

    assert result == expected
