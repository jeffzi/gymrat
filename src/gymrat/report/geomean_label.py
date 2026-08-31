"""Geomean labeling, parts extraction, and value styling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.model import Effect
from gymrat.plural import pluralize
from gymrat.report.display import QUIET_VERDICTS, DisplayClass
from gymrat.report.format import format_delta, format_noise_band_value
from gymrat.report.style import SCOPE_SEPARATOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import GeomeanResult

GEOMEAN_LABEL = "geomean"

GATED_GEOMEAN_LABEL = "gated geomean"

NO_GEOMEAN_FIGURE = "—"

NO_STABLE_METRICS = "no stable metrics"

NO_GEOMEAN_CELL = f"{NO_GEOMEAN_FIGURE}  {NO_STABLE_METRICS}"


def _delta_of(value: float) -> str:
    """A raw percentage figure as a signed delta, wrapping it in a percent effect.

    The aggregate figures carry a bare ratio rather than an :class:`Effect`, so
    they reuse :func:`format_delta` through this adapter.
    """
    return format_delta(Effect(value=value, unit="percent"))


def geomean_label(n: int) -> str:
    """The geomean row's label, carrying the count of metrics behind the figure.

    A table with one candidate names the count here, which frees its cells of
    everything but the aggregate itself. An empty geomean has no count to name
    and takes :data:`GEOMEAN_LABEL` alone.
    """
    return GEOMEAN_LABEL if n == 0 else f"{GEOMEAN_LABEL} ({pluralize(n, 'stable metric')})"


def geomean_scope_label(scope: str) -> str:
    """The label of an aggregate row covering one scope — a group or a kind."""
    return f"{GEOMEAN_LABEL} {SCOPE_SEPARATOR} {scope}"


def _geomean_provenance(geomean: GeomeanResult) -> str:
    """The provenance suffix behind a scope's figure.

    ``(n)`` when every scope metric stands behind the figure, ``(n/m)`` when
    exclusions thinned them.
    """
    total = geomean.n + len(geomean.excluded)
    return f"({geomean.n})" if total == geomean.n else f"({geomean.n}/{total})"


def scoped_geomean_label(scope: str, geomean: GeomeanResult) -> str:
    """A sectioned table's aggregate label with the provenance behind its figure."""
    return f"{geomean_scope_label(scope)} {_geomean_provenance(geomean)}"


@dataclass(frozen=True, slots=True)
class GeomeanParts:
    """The geomean's delta, the count behind it, and the band propagated from its metrics.

    Attributes:
        delta: The signed percentage the geomean moved.
        provenance: How many stable metrics stand behind the figure.
        band: The propagated band's figure, without the ``±`` a column pins in
            front of it, and empty where the metrics left it nothing to state.
    """

    delta: str
    provenance: str
    band: str


def geomean_parts(geomean: GeomeanResult) -> GeomeanParts | None:
    """The geomean's delta, band, and provenance, or ``None`` when nothing survived.

    A band of zero is what an aggregate over exact-only metrics propagates: there
    is no noise to state, and ``±0.0%`` would read as a measurement, so the band
    field is left empty.

    Args:
        geomean: The aggregate to take apart.

    Returns:
        The parts, or ``None`` when the geomean covers no metrics.
    """
    if geomean.n == 0:
        return None
    return GeomeanParts(
        delta=_delta_of(geomean.value),
        provenance=pluralize(geomean.n, "stable metric"),
        band=format_noise_band_value(geomean.band) if geomean.band > 0 else "",
    )


def _is_quiet_row(outcomes: Sequence[DisplayClass | None]) -> bool:
    """Whether every defined display class in a row is a quiet one.

    A row with no verdicts at all is left alone rather than counted as quiet.
    """
    defined = [outcome for outcome in outcomes if outcome is not None]
    return len(defined) > 0 and all(outcome in QUIET_VERDICTS for outcome in defined)


def geomean_value_style(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> str:
    """How a geomean's figure is styled: bold always, colored once it clears the noise band.

    The figure is an average of ratios, so it moves whether or not anything did.
    A value inside the band is emboldened and left uncolored. ``outcomes`` — the
    display class of each metric behind the figure — vetoes the color when every
    one is quiet, since coloring that would announce a win the rows all decline
    to claim. An empty ``outcomes`` leaves the band deciding alone.

    Args:
        geomean: The aggregate whose figure is being styled.
        outcomes: The display class of each metric behind the figure.

    Returns:
        A rich style string: ``"bold"``, ``"bold green"``, or ``"bold red"``.
    """
    if _is_quiet_row(outcomes):
        return "bold"
    if geomean.value < -geomean.band:
        return "bold green"
    if geomean.value > geomean.band:
        return "bold red"
    return "bold"
