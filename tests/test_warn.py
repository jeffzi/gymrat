import pytest

from gymrat_py import warn as warn_module
from gymrat_py.adapters import types as adapters_types
from gymrat_py.warn import WarnSink, warn_to_stderr

# ---------------------------------------------------------------------------
# warn_to_stderr
# ---------------------------------------------------------------------------


def test_warn_to_stderr_when_called_does_write_message_with_newline_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    warn_to_stderr("hello")

    captured = capsys.readouterr()
    assert captured.err == "hello\n"
    assert captured.out == ""


# ---------------------------------------------------------------------------
# WarnSink
# ---------------------------------------------------------------------------


def test_warn_sink_when_assigned_plain_callable_does_receive_raw_message():
    collected: list[str] = []
    sink: WarnSink = collected.append

    sink("hello")

    assert collected == ["hello"]


# ---------------------------------------------------------------------------
# adapters.types re-export
# ---------------------------------------------------------------------------


def test_adapters_types_warn_to_stderr_when_referenced_does_reexport_same_object():
    assert adapters_types.warn_to_stderr is warn_module.warn_to_stderr
