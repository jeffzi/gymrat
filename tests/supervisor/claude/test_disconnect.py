"""Tests for how a Claude session disconnects its client.

Every way a session ends reaches the client's ``disconnect`` once. The fake
client here copies the SDK client's own ``disconnect``: it checks its stream,
awaits the close, then uses and clears the stream. Two overlapping callers
therefore fail the way the SDK does, the second resuming with the stream
already cleared.
"""

import asyncio
import contextlib
import warnings
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import TypedDict, override

import pytest
from claude_agent_sdk import TextBlock

from gymrat.supervisor.driver import DriverSession, SessionOutcome
from tests.supervisor._fixtures import (
    FakeClient,
    abort_on_first_usage_update,
    assistant,
    end_and_settle,
    result_message,
    run_interrupting_on_first_usage_update,
    settled_outcome,
    start_claude_session,
    start_past_turns,
)

_DISCONNECT_WARNING = "claude client disconnect failed"

# ---------------------------------------------------------------------------
# test-local client variants
# ---------------------------------------------------------------------------


class _Stream:
    """The message stream a racing client closes when it disconnects."""

    def __init__(self, released: asyncio.Event) -> None:
        self._released = released

    async def close(self) -> None:
        """End the message stream, then stay suspended across several loop turns.

        The suspension is what lets a second ``disconnect`` caller, woken by
        the stream ending, overlap the first.
        """
        self._released.set()
        for _ in range(10):
            await asyncio.sleep(0)

    def close_receive_stream(self) -> None:
        return None


class _RacingOptions(TypedDict, total=False):
    """The keyword options a test row passes to ``_RacingClient``."""

    finite: bool
    fail_follow_up: bool
    after_turn: Sequence[object]


class _RacingClient(FakeClient):
    """A client whose ``disconnect`` checks, awaits, then clears, as the SDK's does."""

    def __init__(
        self,
        messages: Sequence[object],
        *,
        finite: bool = False,
        fail_follow_up: bool = False,
        after_turn: Sequence[object] = (),
    ) -> None:
        super().__init__([*messages, *after_turn], fail_follow_up=fail_follow_up, finite=finite)
        self._stream: _Stream | None = _Stream(self._released)

    @override
    async def disconnect(self) -> None:
        self.disconnect_count += 1
        if self._stream:
            await self._stream.close()
            self._stream.close_receive_stream()
            self._stream = None


class _DisconnectFailingClient(FakeClient):
    """A client whose every ``disconnect`` releases the stream, then raises."""

    @override
    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()
        message = "transport already gone"
        raise RuntimeError(message)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _one_turn() -> list[object]:
    """The messages of a client that finishes a single turn."""
    return [result_message(total_cost_usd=0.01)]


async def _drain(n: int = 30) -> None:
    """Yield to the event loop; the default outlasts the turns ``_Stream.close`` stays suspended."""
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


async def _end_by_interrupt(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` and interrupt it on its first usage update."""
    outcome = await run_interrupting_on_first_usage_update(client)
    await _drain()
    return outcome


async def _end_by_send(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` and send it a follow-up after its first turn."""
    session = await _start_past_first_turn(client)
    await session.send("one more thing")
    return await _outcome(session)


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
    return [str(w.message) for w in caught if _DISCONNECT_WARNING in str(w.message)]


# ---------------------------------------------------------------------------
# one disconnect per session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client_options", "end_session"),
    [
        pytest.param({}, _end_by_end_call, id="end-call"),
        pytest.param({}, _end_by_abort, id="abort"),
        pytest.param(
            {"after_turn": [assistant(TextBlock(text="late"))]},
            _end_by_interrupt,
            id="interrupt",
        ),
        pytest.param({"fail_follow_up": True}, _end_by_send, id="send-failure"),
        pytest.param({"finite": True}, _end_by_stream, id="stream-exhaustion"),
    ],
)
async def test_start_when_session_ended_does_disconnect_client_exactly_once_without_warning(
    client_options: _RacingOptions,
    end_session: Callable[[FakeClient], Awaitable[SessionOutcome]],
):
    client = _RacingClient(_one_turn(), **client_options)

    with _recorded_warnings() as caught:
        await end_session(client)

    assert client.disconnect_count == 1
    assert _disconnect_warnings(caught) == []


@pytest.mark.parametrize(
    ("finite", "end_session", "expected_reason"),
    [
        pytest.param(False, _end_by_end_call, "completed", id="end-call"),
        pytest.param(False, _end_by_abort, "interrupted", id="abort"),
        pytest.param(True, _end_by_stream, "completed", id="stream-exhaustion"),
    ],
)
async def test_start_when_disconnect_raises_does_keep_the_settled_outcome_with_one_warning(
    finite: bool,
    end_session: Callable[[FakeClient], Awaitable[SessionOutcome]],
    expected_reason: str,
):
    client = _DisconnectFailingClient(_one_turn(), finite=finite)

    with _recorded_warnings() as caught:
        outcome = await end_session(client)

    assert (outcome.reason, outcome.cost_usd) == (expected_reason, 0.01)
    assert len(_disconnect_warnings(caught)) == 1
