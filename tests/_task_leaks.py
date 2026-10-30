"""Recorders for asyncio's forgotten-task diagnostics."""

import asyncio

# The two asyncio diagnostics that mark a task the code forgot to await or
# retrieve; both route through the loop's exception handler.
_LEAK_MARKERS = ("was destroyed but it is pending", "exception was never retrieved")


def install_task_leak_recorder() -> list[dict[str, object]]:
    """Route the running loop's exception handler into a list.

    Both "Task was destroyed but it is pending!" and "Task exception was never
    retrieved" are reported through ``loop.call_exception_handler`` when the
    offending task is finalized, so recording every context the handler sees —
    then forcing a collection — captures a forgotten task deterministically.

    Returns:
        The list every context the handler sees is appended to.
    """
    loop = asyncio.get_running_loop()
    records: list[dict[str, object]] = []

    def handler(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        records.append(context)

    loop.set_exception_handler(handler)
    return records


def task_leak_messages(records: list[dict[str, object]]) -> list[str]:
    """The recorded handler messages that name a forgotten-task diagnostic."""
    messages = [str(context.get("message", "")) for context in records]
    return [msg for msg in messages if any(marker in msg for marker in _LEAK_MARKERS)]
