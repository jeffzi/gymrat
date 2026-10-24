"""The refusal of a sample count that cannot balance the run order of its targets."""

from gymrat.errors import GymratError
from gymrat.utils import balanced_counts_around, balancing_cycle, is_balanced


def require_balanced_samples(samples: int, targets: int) -> None:
    """Refuse ``samples`` unless it gives every target every run position equally often.

    Args:
        samples: The configured number of rounds.
        targets: How many targets each round benches.

    Raises:
        GymratError: ``samples`` is not a positive multiple of the balancing
            cycle for ``targets``. The message names the nearest valid counts
            and the error carries the ``unbalanced-samples`` reason.
    """
    if is_balanced(samples, targets):
        return
    choices = " or ".join(str(count) for count in balanced_counts_around(samples, targets))
    message = (
        f"samples = {samples} cannot balance {targets} targets: "
        f"use a multiple of {balancing_cycle(targets)} ({choices})"
    )
    raise GymratError(message, reason="unbalanced-samples")
