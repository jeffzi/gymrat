"""Fit variant labels into a column without losing the ends that tell them apart.

Middle-ellipsis truncation (:func:`shorten_label`) and set-aware widening
(:func:`truncate_labels`) let a long branch name head a column while keeping
the ends that tell sibling branches apart.

Display widths are measured in terminal cells (:func:`rich.cells.cell_len`), so
a wide CJK character counts as two columns. Slicing follows rich's grapheme
clusters, so a cut never separates a combining mark, a variation selector or a
zero-width-joiner sequence from the character it modifies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.cells import cell_len, split_graphemes

if TYPE_CHECKING:
    from collections.abc import Sequence

# U+2026, one cell wide — three periods would cost two more columns.
_ELLIPSIS = "…"

LABEL_DISPLAY_WIDTH = 20
"""Widest a variant label prints, ellipsis included.

A branch name is free to be as long as git allows, but every column it heads is sized from it, so
an unbounded one would push the figures off the right edge of the terminal.
"""


def _clip_head(text: str, budget: int) -> str:
    spans, _ = split_graphemes(text)
    end = 0
    used = 0
    for _span_start, span_end, width in spans:
        if used + width > budget:
            break
        end = span_end
        used += width
    return text[:end]


def _clip_tail(text: str, budget: int) -> str:
    spans, _ = split_graphemes(text)
    start = len(text)
    used = 0
    for span_start, _span_end, width in reversed(spans):
        if used + width > budget:
            break
        start = span_start
        used += width
    return text[start:]


def shorten_label(text: str, max_width: int) -> str:
    """Fit ``text`` into ``max_width`` terminal cells, keeping both of its ends.

    The ends carry the identity: branch names that share a prefix differ in
    their tail, and a progress line names its step at the front and its target
    at the back. Cutting from the middle keeps both, where a plain slice would
    drop whichever end runs past the budget.

    Widening the budget can never lengthen the result. When a cut is needed,
    the cells left after the ellipsis split between head and tail, with the
    extra cell of an odd remainder going to the head.

    Width is measured in terminal cells, so a wide CJK character counts as two.
    The kept head and tail are sliced on grapheme clusters and each is filled
    only with clusters that fit whole, so a kept slice never splits a cluster
    and never overshoots its share.

    Args:
        text: The label to fit.
        max_width: The budget in terminal cells.

    Returns:
        ``text`` verbatim when it already fits, an empty string when
        ``max_width`` is zero or negative, ``"…"`` alone when only one cell is
        available, or ``head + "…" + tail`` otherwise.
    """
    if max_width <= 0:
        return ""

    if cell_len(text) <= max_width:
        return text

    kept = max_width - cell_len(_ELLIPSIS)
    if kept <= 0:
        return _ELLIPSIS

    head_budget = (kept + 1) // 2
    tail_budget = kept - head_budget
    head = _clip_head(text, head_budget)
    tail = _clip_tail(text, tail_budget)
    return f"{head}{_ELLIPSIS}{tail}"


def truncate_labels(labels: Sequence[str]) -> list[str]:
    """Shorten every label to the narrowest width that keeps the set as distinct.

    Labels are shortened as a set rather than one at a time because the ends are
    what tell sibling branches apart: ``feature/experiment-one-fastpath`` and
    ``feature/exploration-two-fastpath`` share both of theirs, so the narrowest
    width that keeps them distinct is the one worth spending. The width grows one
    cell at a time from :data:`LABEL_DISPLAY_WIDTH` up to the longest label until
    the shortened set has as many unique strings as the originals — two labels
    that were already identical stay that way, which is the run's own doing, not
    the display's.

    Widening never lengthens a label that already fits the wider budget:
    :func:`shorten_label` returns such a label verbatim.

    Args:
        labels: The labels to fit, in order.

    Returns:
        A new list of shortened labels aligned with ``labels``. Every label is
        returned verbatim when they all fit :data:`LABEL_DISPLAY_WIDTH`.
    """
    longest = max((cell_len(label) for label in labels), default=0)
    distinct = len(set(labels))
    for max_width in range(LABEL_DISPLAY_WIDTH, longest):
        shortened = [shorten_label(label, max_width) for label in labels]
        if len(set(shortened)) == distinct:
            return shortened
    return list(labels)
