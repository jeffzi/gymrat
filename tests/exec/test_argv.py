"""Behavioral tests for the ``exec_argv`` subprocess layer.

``exec_argv`` runs ``argv[0]`` with the remaining items as its arguments and
no shell interpretation: every argument reaches the child as one ``sys.argv``
entry, and no shell metacharacter is ever expanded. Its teardown is shared
with the shell form and tested in ``test_teardown``.

Real-subprocess tests are POSIX-only for the same reasons as the shell form:
process groups, session leaders, and ``os.killpg`` do not exist on win32.
"""

import json
import sys
from collections.abc import Callable

import pytest

from gymrat.exec import ExecOptions, ExecResult, exec_argv

# ---------------------------------------------------------------------------
# no shell interpretation — metacharacters pass through verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param("hello world", id="spaces"),
        pytest.param("$HOME", id="dollar-variable"),
        pytest.param("foo|bar", id="pipe"),
        pytest.param('say "hi"', id="double-quotes"),
        pytest.param("it's", id="single-quote"),
        pytest.param("a;b", id="semicolon"),
        pytest.param("a && b", id="double-ampersand"),
    ],
)
async def test_exec_argv_when_arg_contains_metacharacter_does_pass_verbatim(
    make_opts: Callable[..., ExecOptions],
    arg: str,
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))", arg],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    received = json.loads(result.stdout.strip())
    assert received == [arg]
