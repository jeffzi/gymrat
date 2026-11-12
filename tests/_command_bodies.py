"""Shared command bodies for tests that run ``with_repo_lock``."""

from gymrat.command_run import CommandTrace


async def ok_body(trace: CommandTrace) -> str:
    """Trivial command body for tests that only inspect what the command leaves behind."""
    return "ok"
