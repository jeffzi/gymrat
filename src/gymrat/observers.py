"""Fan one event out to several subscribers, isolating their failures."""

from collections.abc import Callable, Iterable


def fan_out[E](
    subscribers: Iterable[Callable[[E], None]],
    on_error: Callable[[Exception], None],
) -> Callable[[E], None]:
    """Build a callback that dispatches each event to every subscriber in order.

    Every subscriber receives the identical event object. A subscriber that
    raises never silences the others: its exception goes to ``on_error`` and
    dispatch continues with the next subscriber. With no subscribers the
    callback is a no-op.

    Args:
        subscribers: The callbacks to dispatch each event to, in call order.
            They are snapshotted when ``fan_out`` is called.
        on_error: The sink that receives each exception a subscriber raises.

    Returns:
        A callback that dispatches each event to all subscribers.
    """
    subs = tuple(subscribers)

    def dispatch(event: E) -> None:
        for subscriber in subs:
            try:
                subscriber(event)
            except Exception as error:  # noqa: BLE001 - reported through on_error; one failure must not break the chain
                on_error(error)

    return dispatch
