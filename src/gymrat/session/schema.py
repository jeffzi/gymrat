"""Schema version and enum vocabulary for the session JSONL log.

This module is the single home of the log's on-disk vocabulary: the format
version and every closed string set a record field may hold. ``records.py``
imports these so that bumping the version or widening an enum touches one file.

The values mirror the wire form exactly -- ``"no-signal"``, ``"permutation"``,
``"nothing-to-commit"`` -- because they are what the JSONL log carries and what
a reader validates against. Do not add ``"signed-rank"`` to :data:`Method`: no
released log ever carried it, so no reader needs to accept it — keeping it out
preserves a vocabulary free of dead values.
"""

from typing import Literal

SchemaVersion = Literal[1]
"""The session JSONL format versions a reader accepts, as a type."""

SCHEMA_VERSION: SchemaVersion = 1
"""Version of the session JSONL format these schemas describe."""

MetricOutcome = Literal["improved", "regressed", "no-signal", "unstable"]
"""How a single metric moved, once its samples were judged."""

Method = Literal["permutation", "band", "exact"]
"""The statistical test that produced a metric's verdict.

Identical to the model's own method union: the sign-flip permutation test is the default, with
the band and exact fallbacks.
"""

PrimaryKind = Literal["geomean", "metric"]
"""Whether an iteration's primary aggregates every gating metric or names one."""

Outcome = Literal["improved", "regressed", "no-signal"]
"""An iteration's overall outcome -- the tri-state an agent acts on.

Unlike a per-metric :data:`MetricOutcome`, an iteration is never reported ``"unstable"``.
"""

KeepStatus = Literal["committed", "blocked"]
"""Whether a kept iteration was committed or refused."""

KeepReason = Literal[
    "checks-failed",
    "gating-regression",
    "nothing-measured",
    "nothing-to-commit",
    "not-improved",
]
"""Why a keep was blocked."""

HookStage = Literal["before", "after"]
"""Which side of an iteration a hook ran on."""

CommandOrigin = Literal["cli", "tool"]
"""What invoked a command: a person at a terminal, or the supervisor's tool host.

- ``cli``  — run directly, by a person or a script
- ``tool`` — run by the supervised agent through the in-process tool host
"""

CommandReason = Literal[
    "stop-condition",
    "budget-exceeded",
    "unsettled",
    "gating-block",
    "already-stopped",
    "no-session",
    "finalized",
    "nothing-measured",
    "gating-regression",
    "nothing-to-commit",
    "checks-failed",
    "not-improved",
    "nothing-to-discard",
    "stale-session",
    "nothing-kept",
    "dirty-worktree",
    "unkept-commits",
    "bad-branch",
    "branch-exists",
    "fail-on",
    "no-filter",
    "no-baseline",
    "supervised-use-tool",
    "error",
]
"""Why a command exited non-zero.

Each value is produced by the command named after the dash-prefix:

- ``stop-condition``    — iterate (the stop condition fired)
- ``budget-exceeded``   — iterate (the iteration budget ran out)
- ``unsettled``         — iterate (unsettled iterations remain)
- ``gating-block``      — iterate (a gating metric regressed and blocked)
- ``already-stopped``   — iterate (a stop record already exists)
- ``no-session``        — any command requiring a session
- ``finalized``         — iterate / keep / discard (session already finalized)
- ``nothing-measured``  — keep (no iteration was measured)
- ``gating-regression`` — keep (gating regression blocked the keep)
- ``nothing-to-commit`` — keep (nothing to commit)
- ``checks-failed``     — keep (configured checks failed)
- ``not-improved``      — keep (the iteration was not improved)
- ``nothing-to-discard`` — discard (nothing to discard)
- ``stale-session``     — discard / keep (the session is stale)
- ``nothing-kept``      — finalize (no kept iterations)
- ``dirty-worktree``    — start (the worktree has uncommitted changes)
- ``unkept-commits``    — finalize (unkept commits remain)
- ``bad-branch``        — start (the branch is invalid)
- ``branch-exists``     — start (the branch already exists)
- ``fail-on``           — iterate (the fail-on condition fired)
- ``no-filter``         — probe (names given with no filter template configured)
- ``no-baseline``       — probe (the session has no baseline record)
- ``supervised-use-tool`` — any command typed outside the supervisor during a live supervised run
- ``error``             — any command (an unexpected error)
"""
