"""Tests for how a Claude session disconnects its client.

Every way a session ends reaches the client's ``disconnect`` once. The racing
fake's ``disconnect`` stays suspended across several loop turns, so a cancelled
waiter that does not shield the call would cut the disconnect short.
"""

import asyncio
import contextlib
import warnings
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import override

import pytest
from typing_extensions import TypedDict

from gymrat.supervisor.driver import DriverSession, SessionOutcome
from tests.supervisor._fixtures import (
    FakeClient,
    abort_on_first_usage_update,
    end_and_settle,
    result_message,
    settled_outcome,
    start_claude_session,
    start_past_turns,
)

#: The error every failing disconnect raises, which its warning must carry.
_DISCONNECT_ERROR = "transport already gone"

# ---------------------------------------------------------------------------
# test-local client variants
# ---------------------------------------------------------------------------


class _RacingOptions(TypedDict, total=False, closed=True):
    """The keyword options a test row passes to ``_RacingClient``."""

    finite: bool


class _RacingClient(FakeClient):
    """A client whose ``disconnect`` stays suspended across several loop turns."""

    @override
    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()
        await _drain(10)


class _DisconnectFailingClient(FakeClient):
    """A client whose every ``disconnect`` releases the stream, then raises."""

    @override
    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()
        raise RuntimeError(_DISCONNECT_ERROR)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _one_turn() -> list[object]:
    """The messages of a client that finishes a single turn."""
    return [result_message(total_cost_usd=0.01)]


async def _drain(n: int = 30) -> None:
    """Yield to the event loop.

    The default outlasts the turns ``_RacingClient.disconnect`` stays suspended.
    """
    for _ in range(n):
        await asyncio.sleep(0)


async def _outcome(session: DriverSession) -> SessionOutcome:
    """Await the session's outcome, then let any trailing teardown finish."""
    outcome = await settled_outcome(session)
    await _drain()
    return outcome


async def _start_past_first_turn(client: FakeClient) -> DriverSession:
    """Start a session over ``client`` and wait until it has closed its first turn."""
    session, _ = await start_past_turns(client, 1)
    return session


async def _end_by_end_call(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` and end it with ``end()`` after its first turn."""
    session = await _start_past_first_turn(client)
    outcome = await end_and_settle(session)
    await _drain()
    return outcome


async def _end_by_abort(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` and abort it on its first usage update."""
    abort = asyncio.Event()

    return await _outcome(
        start_claude_session(client, abort_on_first_usage_update(abort), abort=abort)
    )


async def _end_by_stream(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` until its own stream ends it."""
    return await _outcome(start_claude_session(client))


@contextlib.contextmanager
def _recorded_warnings() -> Generator[list[warnings.WarningMessage]]:
    """Record every warning raised in the block, none filtered out."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _disconnect_warnings(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning) and _DISCONNECT_ERROR in str(w.message)
    ]


# ---------------------------------------------------------------------------
# one disconnect per session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client_options", "end_session"),
    [
        pytest.param({}, _end_by_end_call, id="end-call"),
        pytest.param({}, _end_by_abort, id="abort"),
        pytest.param({"finite": True}, _end_by_stream, id="stream-exhaustion"),
    ],
)
async def test_start_when_session_ended_does_disconnect_client_exactly_once(
    client_options: _RacingOptions,
    end_session: Callable[[FakeClient], Awaitable[SessionOutcome]],
):
    client = _RacingClient(_one_turn(), **client_options)

    await end_session(client)

    assert client.disconnect_count == 1


async def test_start_when_disconnect_raises_does_keep_the_settled_outcome_with_one_warning():
    client = _DisconnectFailingClient(_one_turn())

    with _recorded_warnings() as caught:
        outcome = await _end_by_end_call(client)

    assert (outcome.reason, outcome.cost_usd) == ("completed", 0.01)
    assert len(_disconnect_warnings(caught)) == 1
