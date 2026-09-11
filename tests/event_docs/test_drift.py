"""Writer, artifact rendering, drift gate, and reader agreement for event docs.

``render_all()`` returns a dict mapping repo-relative path strings to text
content for the four generated artifacts.  ``write_all(root)`` writes them
under a directory tree.  ``python -m gymrat.event_docs`` invokes ``write_all``
against the repo root.

The drift test regenerates every artifact in memory and asserts each matches
its committed file byte for byte.  A mismatch means the committed file is
stale; the assertion message tells the reader to run ``task schemas``.

The reader-agreement tests derive the expected type sets from the actual
reader functions (fold_session, status rendering, outcome_record_count, and
the supervisor event union), then compare them to the ``READERS`` constant
so the two cannot diverge.
"""

import json
import subprocess
import sys
import typing
from pathlib import Path

from gymrat.config import BenchlessConfig
from gymrat.event_docs.asyncapi import READERS
from gymrat.loop.status import status_session
from gymrat.session import (
    BaselineRecord,
    BaselineRef,
    SessionLogRecord,
    SessionRecord,
    Worktrees,
    fold_session,
)
from gymrat.supervisor.events import SessionEvent
from gymrat.supervisor.turns import outcome_record_count
from tests.session.records._fixtures import (
    AT,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_record,
    stop_record,
    write_session_log,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPECTED_KEYS = frozenset({
    "schemas/session-log.schema.json",
    "schemas/supervisor-log.schema.json",
    "schemas/asyncapi.yaml",
    "docs/event-reference.md",
})


# ---------------------------------------------------------------------------
# render_all
# ---------------------------------------------------------------------------


def test_render_all_when_called_does_return_dict_with_exactly_four_artifact_keys():
    from gymrat.event_docs import render_all

    result = render_all()

    assert isinstance(result, dict)
    assert set(result.keys()) == _EXPECTED_KEYS


def test_render_all_when_called_does_return_nonempty_strings():
    from gymrat.event_docs import render_all

    result = render_all()

    for key, value in result.items():
        assert isinstance(value, str), f"{key}: expected str, got {type(value).__name__}"
        assert value, f"{key}: value is empty"


def test_render_all_when_called_does_return_valid_json_for_schema_artifacts():
    from gymrat.event_docs import render_all

    result = render_all()

    for key in (
        "schemas/session-log.schema.json",
        "schemas/supervisor-log.schema.json",
    ):
        parsed = json.loads(result[key])
        assert isinstance(parsed, dict), f"{key}: parsed JSON is not a dict"


def test_render_all_when_called_does_return_yaml_with_asyncapi_marker():
    from gymrat.event_docs import render_all

    result = render_all()

    assert "asyncapi:" in result["schemas/asyncapi.yaml"]


def test_render_all_when_called_does_return_markdown_starting_with_html_comment():
    from gymrat.event_docs import render_all

    result = render_all()

    assert result["docs/event-reference.md"].startswith("<!--")


# ---------------------------------------------------------------------------
# write_all
# ---------------------------------------------------------------------------


def test_write_all_when_called_does_write_four_nonempty_files(tmp_path: Path):
    from gymrat.event_docs import write_all

    paths = write_all(tmp_path)

    assert len(paths) == len(_EXPECTED_KEYS)
    for path in paths:
        assert path.exists(), f"missing: {path}"
        assert path.stat().st_size > 0, f"empty: {path}"


def test_write_all_when_called_does_create_files_at_expected_relative_paths(tmp_path: Path):
    from gymrat.event_docs import write_all

    write_all(tmp_path)

    for rel in _EXPECTED_KEYS:
        expected = tmp_path / rel
        assert expected.is_file(), f"expected file at {expected}"


def test_write_all_when_given_str_root_does_accept_it(tmp_path: Path):
    from gymrat.event_docs import write_all

    paths = write_all(str(tmp_path))

    assert len(paths) == len(_EXPECTED_KEYS)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


def test_main_module_when_run_does_exit_zero_with_four_output_lines():
    result = subprocess.run(
        [sys.executable, "-m", "gymrat.event_docs"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == len(_EXPECTED_KEYS)


def test_main_module_when_run_outside_repo_does_exit_nonzero_and_write_nothing(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "gymrat.event_docs"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )

    assert result.returncode != 0
    assert not result.stdout.strip()
    for rel in _EXPECTED_KEYS:
        assert not (tmp_path / rel).exists(), f"artifact written outside repo: {rel}"


# ---------------------------------------------------------------------------
# drift gate
# ---------------------------------------------------------------------------


def test_drift_when_committed_artifacts_match_render_all_does_pass():
    from gymrat.event_docs import render_all

    rendered = render_all()

    for rel_path, expected_content in rendered.items():
        committed = _REPO_ROOT / rel_path
        assert committed.is_file(), (
            f"{rel_path} does not exist on disk. Run `task schemas` to generate it."
        )
        actual = committed.read_text(encoding="utf-8")
        assert actual == expected_content, (
            f"{rel_path} is stale — committed file differs from render_all() output. "
            "Run `task schemas` to regenerate."
        )


def test_drift_when_artifact_is_modified_does_detect_mismatch(tmp_path: Path):
    from gymrat.event_docs import render_all, write_all

    write_all(tmp_path)
    rendered = render_all()

    first_key = next(iter(rendered))
    artifact = tmp_path / first_key
    original = artifact.read_text(encoding="utf-8")
    artifact.write_text(original + " ", encoding="utf-8")

    tampered = artifact.read_text(encoding="utf-8")
    assert tampered != rendered[first_key]


# ---------------------------------------------------------------------------
# reader agreement — READERS matches the actual reader functions
# ---------------------------------------------------------------------------


#: A minimal baseline measurement.
_BASELINE = BaselineRecord(
    type="baseline",
    at=AT,
    label="main",
    samples=({"total_ms": 15200},),
)


def _config() -> BenchlessConfig:
    """A benchless config for the status renderer."""
    return BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200.0,
        primary="geomean",
    )


def _worktrees(root: str) -> Worktrees:
    base = Path(root) / ".gymrat" / "worktrees"
    return Worktrees(experiment=str(base / "experiment"), baseline=str(base / "baseline"))


def _session_at_root(root: str) -> SessionRecord:
    return session_record(
        baseline=BaselineRef(ref="main", sha="a" * 40),
        worktrees=_worktrees(root),
    )


def _fold_session_types() -> set[str]:
    """Derive the set of wire types that change ``SessionState`` under fold_session.

    A type is in the set when folding a log containing it (in valid context)
    produces a different state than folding the same log without it.
    """
    probes: dict[str, tuple[tuple[SessionLogRecord, ...], SessionLogRecord]] = {
        "session": ((), session_record()),
        "iteration": ((session_record(),), iteration_record()),
        "keep": ((session_record(), iteration_record()), committed_keep(1)),
        "discard": ((session_record(), iteration_record()), discard_record(1)),
        "finalize": (
            (session_record(), iteration_record(), committed_keep(1)),
            finalize_record(),
        ),
        "stop": ((session_record(), iteration_record()), stop_record()),
        "baseline": ((session_record(),), _BASELINE),
        "hook": ((session_record(),), hook_record()),
        "command": ((session_record(),), command_record()),
    }

    changes: set[str] = set()
    for wire_type, (before, record) in probes.items():
        without = fold_session(list(before))
        with_record = fold_session([*before, record])
        if with_record != without:
            changes.add(wire_type)

    return changes


def _status_history_types(tmp_path: Path) -> set[str]:
    """Derive the set of wire types that change ``status_session`` output.

    For each record type, compare a log containing it (in valid context) against
    the same log without it, rendered from the same session header.  A type is
    in the set when its presence changes the rendered status output.

    ``finalize`` is excluded: its only effect on ``status_session`` is the
    trailing ``format_status_finalized`` line driven by ``state.finalized``,
    which belongs to the folded session state (already covered by the
    fold-session reader), not the ordered iteration history this reader
    builds. Probing it here would always register as a difference and produce
    a false positive against ``READERS["status-history"].types``.
    """
    probes: dict[str, tuple[tuple[SessionLogRecord, ...], SessionLogRecord]] = {
        "baseline": ((), _BASELINE),
        "iteration": ((), iteration_record()),
        "keep": ((iteration_record(),), committed_keep(1)),
        "discard": ((iteration_record(),), discard_record(1)),
        "stop": ((), stop_record()),
        "hook": ((), hook_record()),
        "command": ((), command_record()),
    }

    types: set[str] = set()
    for wire_type, (before, record) in probes.items():
        root_without = str(tmp_path / f"without-{wire_type}")
        root_with = str(tmp_path / f"with-{wire_type}")
        Path(root_without).mkdir(parents=True, exist_ok=True)
        Path(root_with).mkdir(parents=True, exist_ok=True)

        session = _session_at_root(root_without)
        without_history = before
        with_history = (*before, record)

        write_session_log(root_without, session, without_history)
        write_session_log(root_with, session, with_history)

        without_output = status_session(root_without, _config())
        with_output = status_session(root_with, _config())
        if with_output != without_output:
            types.add(wire_type)

    return types


def _supervisor_guard_types() -> set[str]:
    """Derive the set of wire types counted by ``outcome_record_count``.

    A type is in the set when a singleton list of that record produces a count
    of 1 (not 0, which would mean the type is excluded).
    """
    builders: dict[str, SessionLogRecord] = {
        "session": session_record(),
        "baseline": _BASELINE,
        "iteration": iteration_record(),
        "keep": committed_keep(1),
        "discard": discard_record(1),
        "hook": hook_record(),
        "finalize": finalize_record(),
        "stop": stop_record(),
        "command": command_record(),
    }

    types: set[str] = set()
    for wire_type, record in builders.items():
        if outcome_record_count([record]) > 0:
            types.add(wire_type)
    return types


def _dashboard_types() -> set[str]:
    """Derive the dashboard type set from the supervisor event type union."""
    types: set[str] = set()
    for cls in typing.get_args(SessionEvent):
        # Each event's wire type is its ``type`` field default.
        field_info = cls.model_fields.get("type")
        if field_info is not None:
            types.add(field_info.default)
    return types


def test_readers_fold_session_when_compared_to_fold_session_does_agree():
    expected = _fold_session_types()

    actual = set(READERS["fold-session"].types)

    assert actual == expected


def test_readers_status_history_when_compared_to_status_session_does_agree(tmp_path: Path):
    expected = _status_history_types(tmp_path)

    actual = set(READERS["status-history"].types)

    assert actual == expected


def test_readers_supervisor_guard_when_compared_to_outcome_record_count_does_agree():
    expected = _supervisor_guard_types()

    actual = set(READERS["supervisor-guard"].types)

    assert actual == expected


def test_readers_dashboard_when_compared_to_supervisor_event_union_does_agree():
    expected = _dashboard_types()

    actual = set(READERS["dashboard"].types)

    assert actual == expected
