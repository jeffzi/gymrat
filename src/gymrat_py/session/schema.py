"""Schema version and enum vocabulary for the session JSONL log.

This module is the single home of the log's on-disk vocabulary: the format
version and every closed string set a record field may hold. ``records.py``
imports these so that bumping the version or widening an enum touches one file.

The values mirror the wire form exactly -- ``"no-signal"``, ``"permutation"``,
``"nothing-to-commit"`` -- because they are what the JSONL log carries and what
a reader validates against. Because a schema-1 log is refused outright, a schema-2
record can never legitimately carry the retired ``"signed-rank"`` method, so that
arm is gone rather than kept as dead vocabulary.
"""

from typing import Literal

#: Version of the session JSONL format these schemas describe.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Metric verdict vocabulary
# ---------------------------------------------------------------------------

#: How a single metric moved, once its samples were judged.
Verdict = Literal["improved", "regressed", "no-signal", "unstable"]

#: The statistical test that produced a metric's verdict. Identical to the model's
#: own method union: the sign-flip permutation test is the default, with the band
#: and exact fallbacks.
Method = Literal["permutation", "band", "exact"]

# ---------------------------------------------------------------------------
# Iteration vocabulary
# ---------------------------------------------------------------------------

#: Whether an iteration's primary aggregates every gating metric or names one.
PrimaryKind = Literal["geomean", "metric"]

#: An iteration's overall outcome -- the tri-state an agent acts on. Unlike a
#: per-metric :data:`Verdict`, an iteration is never reported ``"unstable"``.
Outcome = Literal["improved", "regressed", "no-signal"]

# ---------------------------------------------------------------------------
# Keep vocabulary
# ---------------------------------------------------------------------------

#: Whether a kept iteration was committed or refused.
KeepStatus = Literal["committed", "blocked"]

#: Why a keep was blocked.
KeepReason = Literal[
    "checks-failed",
    "gating-regression",
    "nothing-measured",
    "nothing-to-commit",
]

# ---------------------------------------------------------------------------
# Hook vocabulary
# ---------------------------------------------------------------------------

#: Which side of an iteration a hook ran on.
HookStage = Literal["before", "after"]
