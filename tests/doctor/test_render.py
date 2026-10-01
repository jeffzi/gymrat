"""Tests for the doctor text and JSON renderers.

No mocks: a real ``DoctorReport`` is rendered and its lines, glyphs, hint
indentation, caveat note, and summary are asserted directly. Color handling is
driven through ``NO_COLOR`` / ``FORCE_COLOR`` the way a shell would set it.
"""

import json
from collections.abc import Callable

import pytest
from syrupy.assertion import SnapshotAssertion

from gymrat.doctor import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    create_doctor_report,
    render_doctor_json,
    render_doctor_report,
)
from tests._ansi import strip_ansi


def lines(output: str) -> list[str]:
    return strip_ansi(output).split("\n")


def _env(**overrides: object) -> EnvironmentInfo:
    base: dict[str, object] = {
        "gymrat_version": "0.5.0",
        "python_version": "3.13.0",
        "platform": "darwin",
    }
    base.update(overrides)
    return EnvironmentInfo(**base)  # pyrefly: ignore


def _report(sections: list[CheckSection], **env_overrides: object) -> DoctorReport:
    return create_doctor_report(_env(**env_overrides), sections)


# ---------------------------------------------------------------------------
# environment header
# ---------------------------------------------------------------------------


def test_render_doctor_report_when_rendered_does_carry_version_python_and_platform_in_header():
    report = _report([], gymrat_version="1.2.3", python_version="3.13.1", platform="linux")

    header = lines(render_doctor_report(report))[0]

    assert "1.2.3" in header
    assert "3.13.1" in header
    assert "linux" in header


# ---------------------------------------------------------------------------
# section and check rendering
# ---------------------------------------------------------------------------


def test_render_doctor_report_when_section_present_does_show_title():
    report = _report([CheckSection(title="Environment", checks=[Check("git", "ok", "found")])])

    assert "Environment" in strip_ansi(render_doctor_report(report))


def test_render_doctor_report_when_mixed_statuses_does_mark_each_with_its_glyph():
    report = _report([
        CheckSection(
            title="Checks",
            checks=[
                Check("git", "ok", "found"),
                Check("repo", "warn", "not inside"),
                Check("config", "fail", "missing"),
            ],
        )
    ])

    rendered = lines(render_doctor_report(report))

    assert "✓" in next(line for line in rendered if "found" in line)
    assert "⚠" in next(line for line in rendered if "not inside" in line)
    assert "✗" in next(line for line in rendered if "missing" in line)


def test_render_doctor_report_when_hint_present_does_indent_four_with_backticks_stripped():
    report = _report([
        CheckSection(
            title="Env",
            checks=[Check("git", "fail", "not found", hint="run `gymrat init` to set up")],
        )
    ])

    hint_line = next(line for line in lines(render_doctor_report(report)) if "gymrat init" in line)

    assert hint_line.startswith("    ")
    assert "`" not in hint_line


def test_render_doctor_report_when_multiline_detail_does_indent_continuations_under_glyph():
    report = _report([
        CheckSection(
            title="Bench",
            checks=[Check("bench", "ok", "line one\nline two\nline three")],
        )
    ])

    rendered = lines(render_doctor_report(report))

    assert next(line for line in rendered if "line one" in line) == "  ✓ line one"
    assert "    line two" in rendered
    assert "    line three" in rendered


def test_render_doctor_report_when_check_has_no_hint_does_omit_hint_line():
    report = _report([CheckSection(title="Env", checks=[Check("git", "ok", "found")])])

    matched = [line for line in lines(render_doctor_report(report)) if "found" in line]

    assert len(matched) == 1


# ---------------------------------------------------------------------------
# caveat note
# ---------------------------------------------------------------------------


def _note(report_output: str) -> str:
    return next(line for line in lines(report_output) if "Note:" in line)


def test_render_doctor_report_when_default_does_mention_skill_file_location_in_note():
    report = _report([CheckSection(title="Env", checks=[Check("git", "ok", "found")])])

    note = _note(render_doctor_report(report))

    assert "skill file location" in note
    assert "presence ≠ loaded" in note


def test_render_doctor_report_when_workflow_skipped_does_switch_note():
    report = _report([
        CheckSection(
            title="Workflow",
            checks=[Check("workflow", "ok", "Skipped — fix config errors first")],
        )
    ])

    note = _note(render_doctor_report(report))

    assert "skipped" in note.lower()
    assert "skill file location" not in note


def test_render_doctor_report_when_workflow_ran_own_checks_does_keep_default_note():
    report = _report([
        CheckSection(
            title="Workflow",
            checks=[Check("skill file", "ok", "Skill file is installed")],
        )
    ])

    assert "skill file location" in _note(render_doctor_report(report))


# ---------------------------------------------------------------------------
# summary line
# ---------------------------------------------------------------------------


def test_render_doctor_report_when_mixed_statuses_does_report_all_three_counts():
    report = _report([
        CheckSection(
            title="Mixed",
            checks=[
                Check("a", "ok", ""),
                Check("b", "ok", ""),
                Check("c", "warn", ""),
                Check("d", "fail", ""),
            ],
        )
    ])

    output = strip_ansi(render_doctor_report(report))

    assert "2 ok" in output
    assert "1 warning" in output
    assert "1 failure" in output


def test_render_doctor_report_when_multiple_per_status_does_pluralize_counts():
    report = _report([
        CheckSection(
            title="All",
            checks=[
                Check("a", "ok", ""),
                Check("b", "warn", ""),
                Check("c", "warn", ""),
                Check("d", "fail", ""),
                Check("e", "fail", ""),
                Check("f", "fail", ""),
            ],
        )
    ])

    output = strip_ansi(render_doctor_report(report))

    assert "1 ok" in output
    assert "2 warnings" in output
    assert "3 failures" in output


# ---------------------------------------------------------------------------
# color handling
# ---------------------------------------------------------------------------


# CSI introducer; present in the output iff any ANSI escape was emitted.
_ESCAPE_PREFIX = "\x1b["


def _two_status_report() -> DoctorReport:
    """A report with one passing and one failing check."""
    return _report([
        CheckSection(title="Env", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])
    ])


@pytest.mark.parametrize(
    ("env", "color", "expect_ansi"),
    [
        pytest.param("FORCE_COLOR", None, True, id="default-defers-to-env"),
        pytest.param(None, True, True, id="forced-on"),
        pytest.param(None, False, False, id="forced-off"),
    ],
)
def test_render_doctor_report_when_color_resolved_does_control_ansi(
    monkeypatch: pytest.MonkeyPatch, env: str | None, color: bool | None, expect_ansi: bool
):
    if env is not None:
        monkeypatch.setenv(env, "1")
    report = _two_status_report()

    output = render_doctor_report(report, color=color)

    assert (_ESCAPE_PREFIX in output) is expect_ansi


def _mixed_status_report() -> DoctorReport:
    """A report with a passing, a warning and a failing check, each kind with a hint where it applies."""
    return _report([
        CheckSection(
            title="Environment",
            checks=[
                Check("git", "ok", "git 2.45.0"),
                Check("skill", "warn", "skill not installed", hint="run gymrat init"),
            ],
        ),
        CheckSection(
            title="Bench",
            checks=[Check("bench", "fail", "bench not set", hint="set bench in gymrat.toml")],
        ),
    ])


def _warning_only_report() -> DoctorReport:
    """A report whose only problem is a warning."""
    return _report([
        CheckSection(
            title="Environment",
            checks=[
                Check("git", "ok", "git 2.45.0"),
                Check("skill", "warn", "skill not installed", hint="run gymrat init"),
            ],
        )
    ])


@pytest.mark.parametrize("color", [True, False], ids=["color-on", "color-off"])
@pytest.mark.parametrize(
    "make_report",
    [
        pytest.param(_two_status_report, id="ok-and-fail"),
        pytest.param(_mixed_status_report, id="ok-warn-fail"),
        pytest.param(_warning_only_report, id="warn-only"),
    ],
)
def test_render_doctor_report_when_rendered_does_match_the_snapshot(
    make_report: Callable[[], DoctorReport], color: bool, snapshot: SnapshotAssertion
):
    report = make_report()

    output = render_doctor_report(report, color=color)

    assert output.split("\n") == snapshot


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def test_render_doctor_json_when_force_color_env_does_carry_no_ansi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    report = _two_status_report()

    output = render_doctor_json(report)

    assert _ESCAPE_PREFIX not in output


def test_render_doctor_json_when_rendered_does_emit_full_document():
    report = _report(
        [
            CheckSection(
                title="Environment",
                checks=[
                    Check("git", "ok", "available"),
                    Check("repo", "fail", "not in repo", hint="run inside repo"),
                ],
            )
        ],
        gymrat_version="1.0.0",
    )

    parsed = json.loads(render_doctor_json(report))

    assert parsed == {
        "environment": {
            "gymrat_version": "1.0.0",
            "python_version": "3.13.0",
            "platform": "darwin",
        },
        "sections": [
            {
                "title": "Environment",
                "checks": [
                    {"name": "git", "status": "ok", "detail": "available"},
                    {
                        "name": "repo",
                        "status": "fail",
                        "detail": "not in repo",
                        "hint": "run inside repo",
                    },
                ],
            }
        ],
        "ok_count": 1,
        "warn_count": 0,
        "fail_count": 1,
        "has_failures": True,
    }
    section = parsed["sections"][0]
    plain_check, hinted_check = section["checks"]
    assert list(parsed) == [
        "environment",
        "sections",
        "ok_count",
        "warn_count",
        "fail_count",
        "has_failures",
    ]
    assert list(parsed["environment"]) == ["gymrat_version", "python_version", "platform"]
    assert list(section) == ["title", "checks"]
    assert list(plain_check) == ["name", "status", "detail"]
    assert list(hinted_check) == ["name", "status", "detail", "hint"]
