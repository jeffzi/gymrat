"""Shared builders and probes for the supervisor event tests.

These helpers are reused by later supervisor suites (the event log and the
stdio driver), so they live in one module rather than being duplicated per
test file. ``collecting_observer`` hands back an appending observer paired with
the list it fills; ``make_launch`` builds a fully-populated ``LaunchEvent`` from
overridable defaults; ``read_log_lines`` parses a JSONL log into dicts.
``seed_session_log``, ``seed_with_stop``, ``add_stop_async``, and
``supervise_fast`` share the turn-loop test boilerplate.
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NamedTuple, override

from gymrat.config import BenchlessConfig, Effort
from gymrat.session.clock import now_ms, now_ns
from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from gymrat.supervisor import (
    Driver,
    FollowUpEvent,
    supervise,
)
from gymrat.supervisor.context import SupervisedSession
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.events import (
    CapEvent,
    DirtyInfo,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    TurnEndEvent,
)
from gymrat.supervisor.supervise import SupervisionResult
from tests.session.records._fixtures import (
    session_record,
    stop_record,
)
from tests.supervisor._mock_driver import EmitStep, _MockSession


class ObserverProbe(NamedTuple):
    """An observer paired with the list it appends every received event to."""

    events: list[SessionEvent]
    observer: SessionObserver


def collecting_observer() -> ObserverProbe:
    """Return an observer that records each event it receives, and its list."""
    events: list[SessionEvent] = []
    return ObserverProbe(events, events.append)


def _cap_events(events: list[SessionEvent]) -> list[CapEvent]:
    return [event for event in events if isinstance(event, CapEvent)]


def make_launch(
    *,
    at: int = 1_000_000_000_000,
    head_sha: str = "abc123def",
    dirty: Literal[False] | DirtyInfo = False,
    max_minutes: float = 5,
    max_usd: float | None = None,
    model: str | None = None,
    effort: Effort | None = None,
    runbook_path: str = "/path/to/runbook.md",
    kickoff_summary: str = "test kickoff",
    session_id: str = "20260813-125044-34ec",
) -> LaunchEvent:
    """Build a ``LaunchEvent`` from shared defaults, overridden per keyword."""
    return LaunchEvent(
        at=at,
        schema_version=1,
        head_sha=head_sha,
        dirty=dirty,
        max_minutes=max_minutes,
        max_usd=max_usd,
        model=model,
        effort=effort,
        runbook_path=runbook_path,
        kickoff_summary=kickoff_summary,
        session_id=session_id,
    )


def read_log_lines(log_path: str | Path) -> list[dict[str, object]]:
    """Parse a JSONL log file into a list of decoded JSON objects."""
    text = Path(log_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def make_prompt(
    *,
    kickoff: str = "do the thing",
    cwd: str = "/tmp/test",
    system_prompt_append: str | None = None,
    model: str | None = None,
    effort: Effort | None = None,
    command_timeout_ms: int | None = None,
    max_budget_usd: float | None = None,
    traceparent: str | None = None,
) -> SessionPrompt:
    """Build a ``SessionPrompt`` from shared defaults, overridden per keyword."""
    return SessionPrompt(
        kickoff=kickoff,
        cwd=cwd,
        system_prompt_append=system_prompt_append,
        model=model,
        effort=effort,
        command_timeout_ms=command_timeout_ms,
        max_budget_usd=max_budget_usd,
        traceparent=traceparent,
    )


def noop_observer() -> SessionObserver:
    """Return an observer that discards every event it receives."""

    def _observer(_event: SessionEvent) -> None:
        return None

    return _observer


def result_message(
    *,
    subtype: str = "success",
    is_error: bool = False,
    num_turns: int = 1,
    total_cost_usd: float | None = None,
    result: str | None = None,
) -> SimpleNamespace:
    """Build a result message shaped like the SDK's ``ResultMessage``.

    A result message is identified by having both ``subtype`` and ``num_turns``
    attributes; a system message has ``subtype`` alone.
    """
    return SimpleNamespace(
        subtype=subtype,
        is_error=is_error,
        num_turns=num_turns,
        total_cost_usd=total_cost_usd,
        result=result,
    )


def system_message(
    *, subtype: str = "init", data: dict[str, object] | None = None
) -> SimpleNamespace:
    """Build a system message (has ``subtype`` but lacks ``num_turns``)."""
    ns = SimpleNamespace(subtype=subtype)
    if data is not None:
        ns.data = data
    return ns


# ---------------------------------------------------------------------------
# fake streaming client
# ---------------------------------------------------------------------------


class FakeClient:
    """A stand-in for the SDK streaming client that replays scripted messages.

    ``receive_messages`` yields each supplied message after an
    ``asyncio.sleep(0)`` handshake so an observer-scheduled interrupt or an
    abort lands deterministically between messages.  After the scripted
    messages, the stream blocks until ``disconnect`` releases it, mirroring
    the real SDK whose ``receive_messages`` iterator never terminates.
    """

    def __init__(
        self,
        messages: Sequence[object],
        *,
        throw: Exception | None = None,
    ) -> None:
        self.messages = messages
        self.throw = throw
        self.options: dict[str, object] | None = None
        self.query_prompts: list[str] = []
        self.interrupt_called = False
        self.disconnect_count = 0
        self._released = asyncio.Event()

    async def connect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.query_prompts.append(prompt)

    async def receive_messages(self):
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if self.throw is not None:
            raise self.throw
        await self._released.wait()

    async def interrupt(self) -> None:
        self.interrupt_called = True

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()


class FiniteClient(FakeClient):
    """A client whose stream ends naturally instead of blocking after the script."""

    @override
    async def receive_messages(self):
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if self.throw is not None:
            raise self.throw


class FactoryProbe:
    """A client factory that records its call count and the options it saw."""

    def __init__(self, client: FakeClient) -> None:
        self._client = client
        self.calls = 0

    def __call__(self, options: Mapping[str, object]) -> FakeClient:
        self.calls += 1
        self._client.options = dict(options)
        return self._client


def _default_benchless_config() -> BenchlessConfig:
    """A minimal ``BenchlessConfig`` for tests that need a context but not a real config."""
    return BenchlessConfig(
        adapter="mitata",
        samples=1,
        timeout_seconds=60,
        unstable_noise_pct=5.0,
        primary="geomean",
        runbook=None,
        stop=None,
    )


def make_context(
    *,
    root: str = "/tmp/test-repo",
    log_path: str = "/tmp/events.jsonl",
    lock_path: str = "/tmp/test-repo/.gymrat/lockfile",
    config: BenchlessConfig | None = None,
    deadline_ms: float | None = None,
    max_minutes: float = 10,
    max_usd: float | None = None,
) -> SupervisedSession:
    """Build a ``SupervisedSession`` from shared defaults, overridden per keyword.

    ``deadline_ms`` defaults to ``now_ms() + max_minutes * 60_000`` when not
    supplied, matching the computation ``_run_session`` performs.
    """
    if deadline_ms is None:
        deadline_ms = now_ms() + max_minutes * 60_000
    return SupervisedSession(
        root=root,
        log_path=log_path,
        lock_path=lock_path,
        config=config if config is not None else _default_benchless_config(),
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
        max_usd=max_usd,
    )


# ---------------------------------------------------------------------------
# turn-loop test helpers
# ---------------------------------------------------------------------------


def follow_up_events(events: list[SessionEvent]) -> list[FollowUpEvent]:
    """Return every ``FollowUpEvent`` in ``events``."""
    return [e for e in events if isinstance(e, FollowUpEvent)]


def follow_ups_with_action(events: list[SessionEvent], action: str) -> list[FollowUpEvent]:
    """Return every ``FollowUpEvent`` in ``events`` whose ``action`` matches."""
    return [e for e in follow_up_events(events) if e.action == action]


def seed_session_log(root: str) -> None:
    """Write a minimal session header so ``read_records`` / ``fold_session`` work."""
    jsonl_path = session_jsonl_path(root)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    append_record(jsonl_path, session_record())


def seed_with_stop(root: str) -> None:
    """Seed the session log and append a stop record so the classifier sees ``ends_on_stop``."""
    seed_session_log(root)
    append_record(session_jsonl_path(root), stop_record())


async def add_stop_async(root: str) -> None:
    """Append a stop record so the classifier sees ``ends_on_stop`` on the next turn end."""
    append_record(session_jsonl_path(root), stop_record())


def sent_texts(session: _MockSession) -> list[str | None]:
    """Return the text of every ``send`` call the mock session recorded."""
    return [text for call_type, text in session.calls if call_type == "send"]


def emit_turn_end(
    *,
    cost_usd: float = 0.01,
    origin: Literal["agent", "injected"] = "agent",
    delay_ms: int | None = None,
) -> EmitStep:
    """Build an ``EmitStep`` for a ``TurnEndEvent`` with the fields every caller shares."""
    return EmitStep(
        emit=TurnEndEvent(
            at=now_ns(),
            text="",
            cost_usd=cost_usd,
            origin=origin,
            budget_exhausted=False,
        ),
        delay_ms=delay_ms,
    )


async def supervise_fast(
    driver: Driver,
    prompt: SessionPrompt,
    *,
    context: SupervisedSession,
    launch: LaunchEvent,
    observer: SessionObserver | None = None,
    is_lock_held: Callable[[], bool] = lambda: False,
    grace_ms: int = 30_000,
    settle_window_ms: int = 0,
) -> SupervisionResult:
    """Call ``supervise`` with the settle-window and lock-poll defaults every fast test shares."""
    return await supervise(
        driver,
        prompt,
        context=context,
        launch=launch,
        observer=observer,
        settle_window_ms=settle_window_ms,
        lock_poll_ms=1,
        is_lock_held=is_lock_held,
        grace_ms=grace_ms,
    )
