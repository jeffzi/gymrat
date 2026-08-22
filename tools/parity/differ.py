"""Parsed-JSON comparator for the parity harness.

Compares two already-parsed JSON documents (the pinned reference CLI output and
the port's output) structurally and classifies every leaf difference into one of
three buckets:

- **ignored** — volatile or caller-suppressed paths that never affect the
  verdict,
- **p-value notes** — informational statistical differences that are surfaced but
  never fail a comparison,
- **differences** — everything else, which turns a report red.

The module is pure logic: no I/O, no subprocess, stdlib only.
"""

import math
from dataclasses import dataclass, field


class _Missing:
    """Sentinel for a key or index absent on one side of a comparison.

    Distinct from a JSON ``null`` (``None``) that is *present* on both sides, so
    a diff at a missing key is never confused with a diff between two nulls.
    """

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """A single leaf difference between two documents.

    Attributes:
        path: Dotted path to the leaf, with list indices as ``[i]`` segments,
            e.g. ``"metrics.lat/time.candidates[0].p"``.
        left: Value on the left document, or ``MISSING`` when absent there.
        right: Value on the right document, or ``MISSING`` when absent there.
    """

    path: str
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class DiffReport:
    """Outcome of comparing two documents.

    Attributes:
        differences: Comparison-failing differences, in document order.
        p_notes: Informational p-value differences that never fail the report.
    """

    differences: tuple[DiffEntry, ...]
    p_notes: tuple[DiffEntry, ...]

    @property
    def is_green(self) -> bool:
        """True iff there are no failing differences (p-notes are ignored)."""
        return not self.differences


# ---------------------------------------------------------------------------
# path segments and pattern tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Key:
    """A dict-key path segment, taken verbatim (``/`` and ``=`` are not split)."""

    name: str


@dataclass(frozen=True, slots=True)
class _Index:
    """A concrete list-index path segment."""

    pos: int


@dataclass(frozen=True, slots=True)
class _KeyWild:
    """Pattern token ``*``: matches exactly one non-index segment."""


@dataclass(frozen=True, slots=True)
class _IndexWild:
    """Pattern token ``[*]``: matches any list-index segment."""


_Segment = _Key | _Index
_Token = _Key | _Index | _KeyWild | _IndexWild

# Volatile fields that carry run-specific worktree state; never a real mismatch.
_VOLATILE_PATTERNS: tuple[str, ...] = (
    "worktrees.leftBehind[*].path",
    "worktrees.pruneError",
)

# Statistical p-values drift run-to-run; surfaced as notes, never as failures.
_P_VALUE_PATTERN = "metrics.*.candidates[*].p"


def _tokenize(pattern: str) -> list[_Token]:
    """Tokenize a dotted pattern into segment matchers.

    ``.`` separates key segments; ``[...]`` introduces an index segment. ``*``
    (as a whole key segment) becomes a key wildcard and ``[*]`` an index
    wildcard.
    """
    tokens: list[_Token] = []
    buf = ""
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == ".":
            if buf:
                tokens.append(_key_token(buf))
                buf = ""
            i += 1
        elif ch == "[":
            if buf:
                tokens.append(_key_token(buf))
                buf = ""
            end = pattern.index("]", i)
            inner = pattern[i + 1 : end]
            tokens.append(_IndexWild() if inner == "*" else _Index(int(inner)))
            i = end + 1
        else:
            buf += ch
            i += 1
    if buf:
        tokens.append(_key_token(buf))
    return tokens


def _key_token(segment: str) -> _Token:
    return _KeyWild() if segment == "*" else _Key(segment)


def _seg_matches(seg: _Segment, tok: _Token) -> bool:
    match tok:
        case _KeyWild():
            return isinstance(seg, _Key)
        case _IndexWild():
            return isinstance(seg, _Index)
        case _Key(name=name):
            return isinstance(seg, _Key) and seg.name == name
        case _Index(pos=pos):
            return isinstance(seg, _Index) and seg.pos == pos


def _matches(segments: list[_Segment], tokens: list[_Token]) -> bool:
    return len(segments) == len(tokens) and all(
        _seg_matches(seg, tok) for seg, tok in zip(segments, tokens, strict=True)
    )


def _render(segments: list[_Segment]) -> str:
    out = ""
    for seg in segments:
        if isinstance(seg, _Key):
            out += seg.name if not out else f".{seg.name}"
        else:
            out += f"[{seg.pos}]"
    return out


def _both_nan(left: object, right: object) -> bool:
    """True when both leaves are the float NaN, which never compares equal to itself."""
    return (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    )


@dataclass
class _Collector:
    ignore_patterns: list[list[_Token]]
    p_pattern: list[_Token]
    differences: list[DiffEntry] = field(default_factory=list)
    p_notes: list[DiffEntry] = field(default_factory=list)

    def record(self, segments: list[_Segment], left: object, right: object) -> None:
        # Ignore wins over every other classification, including p-notes.
        if any(_matches(segments, pat) for pat in self.ignore_patterns):
            return
        entry = DiffEntry(path=_render(segments), left=left, right=right)
        if _matches(segments, self.p_pattern):
            self.p_notes.append(entry)
        else:
            self.differences.append(entry)

    def walk(self, left: object, right: object, segments: list[_Segment]) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            self._walk_dict(left, right, segments)
        elif isinstance(left, list) and isinstance(right, list):
            self._walk_list(left, right, segments)
        elif left != right and not _both_nan(left, right):
            # Scalars that differ, or a container-vs-non-container type mismatch:
            # a single leaf difference, never recursed into.
            self.record(segments, left, right)

    def _walk_dict(
        self, left: dict[str, object], right: dict[str, object], segments: list[_Segment]
    ) -> None:
        for key, value in left.items():
            child = [*segments, _Key(key)]
            if key in right:
                self.walk(value, right[key], child)
            else:
                self.record(child, value, MISSING)
        for key, value in right.items():
            if key not in left:
                self.record([*segments, _Key(key)], MISSING, value)

    def _walk_list(self, left: list[object], right: list[object], segments: list[_Segment]) -> None:
        for i in range(max(len(left), len(right))):
            child = [*segments, _Index(i)]
            if i >= len(left):
                self.record(child, MISSING, right[i])
            elif i >= len(right):
                self.record(child, left[i], MISSING)
            else:
                self.walk(left[i], right[i], child)


def diff_json(
    left: object,
    right: object,
    *,
    ignore_paths: tuple[str, ...] = (),
) -> DiffReport:
    """Compare two parsed JSON documents and classify their differences.

    Args:
        left: Parsed JSON value from the left document.
        right: Parsed JSON value from the right document.
        ignore_paths: Additional dotted patterns to suppress. ``*`` matches one
            dict-key segment and ``[*]`` matches any list index; e.g.
            ``"metrics.*.baseline.spreadPct"`` suppresses that field for every
            metric.

    Returns:
        A ``DiffReport`` whose ``differences`` drive ``is_green`` and whose
        ``p_notes`` hold informational p-value differences. Volatile worktree
        fields and ``ignore_paths`` matches never appear in either bucket.
    """
    ignore_patterns = [_tokenize(pattern) for pattern in (*_VOLATILE_PATTERNS, *ignore_paths)]
    collector = _Collector(
        ignore_patterns=ignore_patterns,
        p_pattern=_tokenize(_P_VALUE_PATTERN),
    )
    collector.walk(left, right, [])
    return DiffReport(
        differences=tuple(collector.differences),
        p_notes=tuple(collector.p_notes),
    )
