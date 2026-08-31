"""English pluralization for count-prefixed labels."""

from __future__ import annotations

_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")

_VOWELS = "aeiou"  # cspell:ignore aeiou


def _regular_plural(noun: str) -> str:
    """``noun`` under the regular English suffix rules.

    A sibilant ending takes ``-es``; a consonant followed by ``y`` takes
    ``-ies``; everything else takes ``-s``. Multi-word nouns inflect on their
    last word, which the suffix tests already look at.
    """
    if noun.endswith(_SIBILANT_ENDINGS):
        return f"{noun}es"
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in _VOWELS:
        return f"{noun[:-1]}ies"
    return f"{noun}s"


def pluralize(count: int, noun: str, plural: str | None = None) -> str:
    """Inflect ``noun`` for ``count``, or use the explicit ``plural`` instead.

    A count of one keeps ``noun`` as given; any other count — zero and
    negatives included — takes the plural form.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {plural if plural is not None else _regular_plural(noun)}"
