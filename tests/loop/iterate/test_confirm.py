"""Unit tests for the POSIX/win32 shell-quoting in ``_shell_quote``."""

from __future__ import annotations

import shlex
import subprocess
import sys

import pytest

from gymrat.loop.iterate import confirm as confirm_module
from gymrat.loop.iterate.confirm import _shell_quote

# ---------------------------------------------------------------------------
# _shell_quote
# ---------------------------------------------------------------------------


def test_shell_safe_word_regex_when_module_loaded_does_not_exist():
    assert not hasattr(confirm_module, "_SHELL_SAFE_WORD")


# ---------------------------------------------------------------------------
# POSIX quoting delegates to shlex.quote
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX quoting only")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("simple", id="safe-word"),
        pytest.param("hello world", id="space"),
        pytest.param("it's", id="single-quote"),
        pytest.param("a;b", id="semicolon"),
        pytest.param("$HOME", id="dollar"),
        pytest.param("a&b", id="ampersand"),
        pytest.param("", id="empty-string"),
        pytest.param("sort(n=1000)/time", id="parentheses-and-equals"),
    ],
)
def test_shell_quote_when_posix_does_match_shlex_quote(value: str) -> None:
    result = _shell_quote(value)

    assert result == shlex.quote(value)


# ---------------------------------------------------------------------------
# Windows quoting delegates to subprocess.list2cmdline
# ---------------------------------------------------------------------------


def test_shell_quote_when_win32_does_use_list2cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    value = "decode large payload"

    result = _shell_quote(value)

    assert result == subprocess.list2cmdline([value])
    assert '"' in result
