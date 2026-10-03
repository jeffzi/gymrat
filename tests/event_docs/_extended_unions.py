"""Render the event docs in a fresh interpreter whose record or event union has one extra model.

Run as ``python -m tests.event_docs._extended_unions <channel>`` from the repo root, where
``<channel>`` is ``session-log`` or ``supervisor-log``. The script appends a probe model to the
matching union, rebinds every module attribute that holds the original union to the extended
one, then imports ``gymrat.event_docs`` and prints ``render_all()`` as JSON on stdout.

A module-level value that the generators read and that was derived from a union before ``_extend``
runs holds the original members, so it must be rebound here too, or the generators miss the probe:

- ``SESSION_LOG_MODELS``, the session-log member tuple, is rebound to the extended union's members.
- The supervisor event adapter is rebound to one built from the extended event union.

The substitution happens before ``gymrat.event_docs`` is first imported, so the generators see
the extended union exactly as they would see a model added to the union's source.
"""

import functools
import json
import operator
import sys
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field, TypeAdapter

from gymrat.session.records import SESSION_LOG_MODELS, SessionLogRecord
from gymrat.supervisor.events import SESSION_EVENT_ADAPTER, SessionEvent

#: Wire type of the probe model appended to the extended union.
PROBE_WIRE_TYPE = "probe"


class ProbeModel(BaseModel):
    """A probe model appended to a union to exercise the doc generators."""

    type: Literal["probe"] = PROBE_WIRE_TYPE
    at: str
    note: str


def _with_probe(members: tuple[object, ...]) -> Any:
    return functools.reduce(operator.or_, (*members, ProbeModel))


_ExtendedLogRecord = _with_probe(get_args(SessionLogRecord.__value__))


def _rebind(original: object, replacement: object) -> None:
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        for name, value in list(namespace.items()):
            if value is original:
                setattr(module, name, replacement)


def _extend(channel: str) -> None:
    if channel == "session-log":
        _rebind(SessionLogRecord, _ExtendedLogRecord)
        _rebind(SESSION_LOG_MODELS, get_args(_ExtendedLogRecord))
    elif channel == "supervisor-log":
        extended = _with_probe(get_args(SessionEvent))
        _rebind(SessionEvent, extended)
        adapter = TypeAdapter(Annotated[extended, Field(discriminator="type")])
        _rebind(SESSION_EVENT_ADAPTER, adapter)
    else:
        msg = f"unknown channel {channel!r}"
        raise SystemExit(msg)


if __name__ == "__main__":
    if "gymrat.event_docs" in sys.modules:
        msg = "gymrat.event_docs was imported before the union was extended"
        raise SystemExit(msg)
    _extend(sys.argv[1])

    # Import after _extend rebinds the union: the generators capture it at import time.
    from gymrat.event_docs import render_all

    json.dump(render_all(), sys.stdout)
