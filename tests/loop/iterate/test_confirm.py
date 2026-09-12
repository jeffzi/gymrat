"""Unit tests for the bench-command helpers in ``gymrat.loop.iterate.confirm``."""

from __future__ import annotations

import shlex
import subprocess
import sys

import pytest

from gymrat.loop.iterate import confirm as confirm_module
from gymrat.loop.iterate.confirm import scoped_bench, shell_quote_name
from tests.loop.iterate._fixtures import resolved_config

#: The confirm-rerun template a consumer configures when their bench can be narrowed.
FILTER = "npm run bench -- --filter {names}"

# ---------------------------------------------------------------------------
# scoped_bench
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX quoting only")
@pytest.mark.parametrize(
    ("names", "expected"),
    [
        pytest.param(["total_ms"], "npm run bench -- --filter total_ms", id="one-name"),
        pytest.param(
            ["total_ms", "alloc_bytes"],
            "npm run bench -- --filter total_ms alloc_bytes",
            id="two-names",
        ),
        pytest.param(
            ["decode large payload"],
            "npm run bench -- --filter 'decode large payload'",
            id="name-needing-quoting",
        ),
    ],
)
def test_scoped_bench_when_names_given_does_substitute_them_quoted_into_the_filter(
    names: list[str], expected: str
) -> None:
    config = resolved_config(filter=FILTER)

    result = scoped_bench(config, names)

    assert result == expected


@pytest.mark.parametrize(
    ("filter_template", "names"),
    [
        pytest.param(FILTER, [], id="no-names"),
        pytest.param(None, ["total_ms"], id="no-filter-configured"),
    ],
)
def test_scoped_bench_when_nothing_to_scope_does_return_the_whole_bench(
    filter_template: str | None, names: list[str]
) -> None:
    config = resolved_config(filter=filter_template)

    result = scoped_bench(config, names)

    assert result == config.bench


# ---------------------------------------------------------------------------
# shell_quote_name
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
def test_shell_quote_name_when_posix_does_match_shlex_quote(value: str) -> None:
    result = shell_quote_name(value)

    assert result == shlex.quote(value)


# ---------------------------------------------------------------------------
# Windows quoting delegates to subprocess.list2cmdline
# ---------------------------------------------------------------------------


def test_shell_quote_name_when_win32_does_use_list2cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    value = "decode large payload"

    result = shell_quote_name(value)

    assert result == subprocess.list2cmdline([value])
    assert '"' in result
