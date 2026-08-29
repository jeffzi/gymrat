"""Effect size value type."""

from dataclasses import dataclass
from typing import Literal

EffectUnit = Literal["percent"]
"""Unit an :class:`Effect` is expressed in.

Today only ``"percent"`` is admissible; the alias is shaped as a ``Literal`` union so a
percentage-point (``"pp"``) member can be added later without touching call sites.
"""


@dataclass(frozen=True, slots=True)
class Effect:
    """An observed effect size, immutable and compared by value.

    Attributes:
        value: The magnitude of the effect.
        unit: The unit ``value`` is expressed in.
    """

    value: float
    unit: EffectUnit
