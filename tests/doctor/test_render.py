"""Tests for the doctor text and JSON renderers.

No mocks: a real ``DoctorReport`` is rendered and its lines, glyphs, hint
indentation, caveat note, and summary are asserted directly. Color handling is
driven through ``NO_COLOR`` / ``FORCE_COLOR`` the way a shell would set it.
"""

import json

import pytest

from gymrat.doctor.checks import (
    Check,
    CheckSection,
    EnvironmentInfo,
    create_doctor_report,
)
from gymrat.doctor.render import render_doctor_json, render_doctor_report
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


def _report(sections: list[CheckSection], **env_overrides: object):
    return create_doctor_report(_env(**env_overrides), sections)


@pytest.fixture(autouse=True)
def _neutral_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited color env so each test controls it explicitly."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


# ---------------------------------------------------------------------------
# environment header
# ---------------------------------------------------------------------------


def test_render_doctor_report_header_carries_version_python_and_platform():
    report = _report([], gymrat_version="1.2.3", python_version="3.13.1", platform="linux")

    header = lines(render_doctor_report(report))[0]

    assert "1.2.3" in header
    assert "3.13.1" in header
    assert "linux" in header


# ---------------------------------------------------------------------------
# section and check rendering
# ---------------------------------------------------------------------------


def test_render_doctor_report_shows_section_title():
    report = _report([CheckSection(title="Environment", checks=[Check("git", "ok", "found")])])

    assert "Environment" in strip_ansi(render_doctor_report(report))


def test_render_doctor_report_marks_each_status_with_its_glyph():
    report = _report(
        [
            CheckSection(
                title="Checks",
                checks=[
                    Check("git", "ok", "found"),
                    Check("repo", "warn", "not inside"),
                    Check("config", "fail", "missing"),
                ],
            )
        ]
    )

    rendered = lines(render_doctor_report(report))

    assert "✓" in next(line for line in rendered if "found" in line)
    assert "⚠" in next(line for line in rendered if "not inside" in line)
    assert "✗" in next(line for line in rendered if "missing" in line)


def test_render_doctor_report_renders_hint_indented_four_with_backticks_stripped():
    report = _report(
        [
            CheckSection(
                title="Env",
                checks=[Check("git", "fail", "not found", hint="run `gymrat init` to set up")],
            )
        ]
    )

    hint_line = next(line for line in lines(render_doctor_report(report)) if "gymrat init" in line)

    assert hint_line.startswith("    ")
    assert "`" not in hint_line


def test_render_doctor_report_indents_multiline_detail_continuations_under_the_glyph():
    report = _report(
        [
            CheckSection(
                title="Bench",
                checks=[Check("bench", "ok", "line one\nline two\nline three")],
            )
        ]
    )

    rendered = lines(render_doctor_report(report))

    assert next(line for line in rendered if "line one" in line) == "  ✓ line one"
    assert "    line two" in rendered
    assert "    line three" in rendered


def test_render_doctor_report_omits_hint_line_when_check_has_none():
    report = _report([CheckSection(title="Env", checks=[Check("git", "ok", "found")])])

    matched = [line for line in lines(render_doctor_report(report)) if "found" in line]

    assert len(matched) == 1


# ---------------------------------------------------------------------------
# caveat note
# ---------------------------------------------------------------------------


def _note(report_output: str) -> str:
    return next(line for line in lines(report_output) if "Note:" in line)


def test_render_doctor_report_note_mentions_skill_file_location_by_default():
    report = _report([CheckSection(title="Env", checks=[Check("git", "ok", "found")])])

    note = _note(render_doctor_report(report))

    assert "skill file location" in note
    assert "presence ≠ loaded" in note


def test_render_doctor_report_note_switches_when_workflow_section_was_skipped():
    report = _report(
        [
            CheckSection(
                title="Workflow",
                checks=[Check("workflow", "ok", "Skipped — fix config errors first")],
            )
        ]
    )

    note = _note(render_doctor_report(report))

    assert "skipped" in note.lower()
    assert "skill file location" not in note


def test_render_doctor_report_note_stays_default_when_workflow_ran_its_own_checks():
    report = _report(
        [
            CheckSection(
                title="Workflow",
                checks=[Check("skill file", "ok", "Skill file is installed")],
            )
        ]
    )

    assert "skill file location" in _note(render_doctor_report(report))


# ---------------------------------------------------------------------------
# summary line
# ---------------------------------------------------------------------------


def test_render_doctor_report_summary_reports_all_three_counts():
    report = _report(
        [
            CheckSection(
                title="Mixed",
                checks=[
                    Check("a", "ok", ""),
                    Check("b", "ok", ""),
                    Check("c", "warn", ""),
                    Check("d", "fail", ""),
                ],
            )
        ]
    )

    output = strip_ansi(render_doctor_report(report))

    assert "2 ok" in output
    assert "1 warning" in output
    assert "1 failure" in output


def test_render_doctor_report_summary_pluralizes_singular_and_plural_counts():
    report = _report(
        [
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
        ]
    )

    output = strip_ansi(render_doctor_report(report))

    assert "1 ok" in output
    assert "2 warnings" in output
    assert "3 failures" in output


# ---------------------------------------------------------------------------
# color handling
# ---------------------------------------------------------------------------


def test_render_doctor_report_when_no_color_suppresses_ansi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NO_COLOR", "1")
    report = _report(
        [CheckSection(title="Env", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])]
    )

    assert "\x1b[" not in render_doctor_report(report)


def test_render_doctor_report_when_force_color_emits_ansi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    report = _report(
        [CheckSection(title="Env", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])]
    )

    assert "\x1b[" in render_doctor_report(report)


def test_render_doctor_report_when_color_false_does_suppress_ansi():
    report = _report(
        [CheckSection(title="Env", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])]
    )

    assert "\x1b[" not in render_doctor_report(report, color=False)


def test_render_doctor_report_when_color_true_does_emit_ansi():
    report = _report(
        [CheckSection(title="Env", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])]
    )

    assert "\x1b[" in render_doctor_report(report, color=True)


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def test_render_doctor_json_when_force_color_env_does_carry_no_ansi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    report = _report(
        [CheckSection(title="Env", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])]
    )

    output = render_doctor_json(report)

    assert "\x1b[" not in output


def test_render_doctor_json_carries_environment_sections_and_counts():
    report = _report(
        [
            CheckSection(
                title="Environment",
                checks=[
                    Check("git", "ok", "available"),
                    Check("repo", "warn", "not in repo", hint="run inside repo"),
                ],
            )
        ],
        gymrat_version="1.0.0",
    )

    parsed = json.loads(render_doctor_json(report))

    assert parsed["environment"]["pythonVersion"] == "3.13.0"
    assert "nodeVersion" not in parsed["environment"]
    assert parsed["okCount"] == 1
    assert parsed["warnCount"] == 1
    assert parsed["failCount"] == 0


def test_render_doctor_json_carries_check_status_and_hint():
    report = _report(
        [
            CheckSection(
                title="Config",
                checks=[Check("file", "fail", "missing", hint="create gymrat.json")],
            )
        ]
    )

    parsed = json.loads(render_doctor_json(report))
    check = parsed["sections"][0]["checks"][0]

    assert check["status"] == "fail"
    assert check["hint"] == "create gymrat.json"
