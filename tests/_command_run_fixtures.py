"""Shared session seeding and command body for the ``command_run`` tests.

``test_command_run.py`` and ``test_command_run_tracing.py`` both exercise
``gymrat.command_run.with_repo_lock`` and need the same starting point: a
session log seeded with a header record, and a command body that does nothing.
Both modules import from here instead of duplicating the definitions.
"""

from gymrat.command_run import CommandTrace
from gymrat.session.records import SessionRecord
from tests.session.records._fixtures import session_record, write_session_log


def seeded_session(repo: str) -> SessionRecord:
    """Write a session header to ``repo``'s session log."""
    header = session_record()
    write_session_log(repo, header)
    return header


async def ok_body(trace: CommandTrace) -> str:
    """Trivial command body for tests that only inspect what the command leaves behind."""
    return "ok"
