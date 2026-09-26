"""Behavioral tests for the generic observer fan-out."""

from gymrat.observers import fan_out


class _Event:
    """A distinct object whose identity the subscribers can check."""


def _raise_boom(_: _Event) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_fan_out_when_called_does_dispatch_identical_event_to_subscribers_in_order():
    calls: list[tuple[str, _Event]] = []
    errors: list[Exception] = []

    def first(event: _Event) -> None:
        calls.append(("first", event))

    def second(event: _Event) -> None:
        calls.append(("second", event))

    dispatch = fan_out([first, second], errors.append)
    event = _Event()

    dispatch(event)

    assert [name for name, _ in calls] == ["first", "second"]
    assert all(received is event for _, received in calls)
    assert errors == []


def test_fan_out_when_subscriber_raises_does_report_error_and_call_remaining():
    received: list[_Event] = []
    errors: list[Exception] = []
    dispatch = fan_out([_raise_boom, received.append], errors.append)
    event = _Event()

    dispatch(event)

    assert received == [event]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "boom"


def test_fan_out_when_no_subscribers_does_not_report_errors():
    errors: list[Exception] = []
    dispatch = fan_out([], errors.append)

    dispatch(_Event())

    assert errors == []
