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
from gymrat.supervisor.events import SessionEvent, UsageUpdateEvent
from tests.supervisor._fixtures import (
    FakeClient,
    FiniteClient,
    assistant,
    result_message,
    settled_outcome,
    start_claude_session,
    start_interrupting_on_first_usage_update,
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
        super().__init__([*messages, *after_turn])
        self._finite = finite
        self._fail_follow_up = fail_follow_up
        self._stream: _Stream | None = _Stream(self._released)

    @override
    async def query(self, prompt: str) -> None:
        if self._fail_follow_up and self.query_prompts:
            message = "connection lost"
            raise RuntimeError(message)
        await super().query(prompt)

    @override
    async def receive_messages(self):
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if not self._finite:
            await self._released.wait()

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


class _FiniteDisconnectFailingClient(FiniteClient):
    """A client whose stream ends on its own, and whose ``disconnect`` then raises."""

    @override
    async def disconnect(self) -> None:
        self.disconnect_count += 1
        message = "teardown boom"
        raise RuntimeError(message)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _one_turn() -> list[object]:
    """The messages of a client that finishes a single turn."""
    return [result_message(total_cost_usd=0.01)]


async def _drain(n: int = 30) -> None:
    """Let the event loop process n rounds of pending callbacks."""
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
    await session.end()
    return await _outcome(session)


async def _end_by_abort(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` and abort it on its first usage update."""
    abort = asyncio.Event()

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            abort.set()

    return await _outcome(start_claude_session(client, observer, abort=abort))


async def _end_by_interrupt(client: FakeClient) -> SessionOutcome:
    """Run a session over ``client`` and interrupt it on its first usage update.

    The soft stop lands when the stream delivers its next message, so
    ``client`` must carry one after the first turn.
    """
    return await _outcome(start_interrupting_on_first_usage_update(client))


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
    """The text of every recorded warning that reports a failed disconnect."""
    return [str(w.message) for w in caught if _DISCONNECT_WARNING in str(w.message)]


# ---------------------------------------------------------------------------
# one disconnect per session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client_options", "end_session", "expected_reason", "expected_message"),
    [
        pytest.param({}, _end_by_end_call, "completed", None, id="end-call"),
        pytest.param({}, _end_by_abort, "interrupted", None, id="abort"),
        pytest.param(
            {"after_turn": [assistant(TextBlock(text="late"))]},
            _end_by_interrupt,
            "interrupted",
            None,
            id="interrupt",
        ),
        pytest.param(
            {"fail_follow_up": True}, _end_by_send, "error", "connection lost", id="send-failure"
        ),
        pytest.param({"finite": True}, _end_by_stream, "completed", None, id="stream-exhaustion"),
    ],
)
async def test_start_when_session_ended_does_disconnect_client_exactly_once_without_warning(
    client_options: _RacingOptions,
    end_session: Callable[[FakeClient], Awaitable[SessionOutcome]],
    expected_reason: str,
    expected_message: str | None,
):
    client = _RacingClient(_one_turn(), **client_options)

    with _recorded_warnings() as caught:
        outcome = await end_session(client)

    assert (outcome.reason, outcome.message) == (expected_reason, expected_message)
    assert client.disconnect_count == 1
    assert _disconnect_warnings(caught) == []


@pytest.mark.parametrize(
    ("client", "end_session", "expected_reason"),
    [
        pytest.param(
            _DisconnectFailingClient(_one_turn()), _end_by_end_call, "completed", id="end-call"
        ),
        pytest.param(
            _DisconnectFailingClient(_one_turn()), _end_by_abort, "interrupted", id="abort"
        ),
        pytest.param(
            _FiniteDisconnectFailingClient(_one_turn()),
            _end_by_stream,
            "completed",
            id="stream-exhaustion",
        ),
    ],
)
async def test_start_when_disconnect_raises_does_keep_the_settled_outcome_with_one_warning(
    client: FakeClient,
    end_session: Callable[[FakeClient], Awaitable[SessionOutcome]],
    expected_reason: str,
):
    with _recorded_warnings() as caught:
        outcome = await end_session(client)

    assert (outcome.reason, outcome.cost_usd) == (expected_reason, 0.01)
    assert len(_disconnect_warnings(caught)) == 1
