"""Text and JSON renderers for a doctor report.

The text renderer styles each check with a status glyph, indents continuation
lines and hints under it, and closes with a caveat note and a status summary.
Color follows the project's :func:`render_lines` resolution — ``NO_COLOR`` /
``FORCE_COLOR`` and stdout's TTY status decide whether ANSI escapes appear.
"""

import json

from rich.markup import escape

from gymrat_py.doctor.checks import Check, CheckStatus, DoctorReport
from gymrat_py.report import pluralize
from gymrat_py.report.style import (
    RENDER_WIDTH,
    highlight_inline_code,
    markup,
    render_lines,
)

_STATUS_GLYPHS: dict[CheckStatus, str] = {"ok": "✓", "warn": "⚠", "fail": "✗"}
_STATUS_STYLES: dict[CheckStatus, str] = {"ok": "green", "warn": "yellow", "fail": "red"}

_WORKFLOW_SECTION_TITLE = "Workflow"

# The synthetic check name build_workflow_section emits in place of its real checks;
# the real ones are named after what they inspect ("skill file", "checks", …).
_WORKFLOW_SKIP_CHECK_NAME = "workflow"

# Continuation lines and hints align under the status glyph, two spaces past its
# two-space indent.
_DETAIL_INDENT = "    "

_NOTE_WORKFLOW_RAN = (
    "Note: prepare scripts were not run; only the Claude skill file location "
    "was checked (presence ≠ loaded)."
)
_NOTE_WORKFLOW_SKIPPED = (
    "Note: prepare scripts were not run; workflow checks (including the Claude "
    "skill file check) were skipped because config errors block them."
)


def _workflow_was_skipped(report: DoctorReport) -> bool:
    """Whether config problems made the workflow section collapse to the skip placeholder.

    When that happened the skill file was never looked at, so the caveat note
    drops the skill-file claim.
    """
    workflow = next(
        (section for section in report.sections if section.title == _WORKFLOW_SECTION_TITLE),
        None,
    )
    if workflow is None or not workflow.checks:
        return False
    return all(check.name == _WORKFLOW_SKIP_CHECK_NAME for check in workflow.checks)


def _header_line(report: DoctorReport) -> str:
    env = report.environment
    name = markup(f"gymrat v{env.gymrat_version}", "bold")
    rest = markup(f" · python {env.python_version} · {env.platform}", "dim")
    return f"{name}{rest}"


def _check_lines(status: CheckStatus, detail: str, hint: str | None) -> list[str]:
    glyph = markup(_STATUS_GLYPHS[status], _STATUS_STYLES[status])
    first, *continuations = detail.split("\n")
    lines = [f"  {glyph} {escape(first)}"]
    lines.extend(f"{_DETAIL_INDENT}{escape(line)}" for line in continuations)
    if hint is not None:
        lines.append(f"{_DETAIL_INDENT}[dim]{highlight_inline_code(hint)}[/dim]")
    return lines


def render_doctor_report(report: DoctorReport) -> str:
    """Render a doctor report as styled text for the terminal."""
    lines: list[str] = [_header_line(report), ""]

    for section in report.sections:
        lines.append(markup(section.title, "bold"))
        for check in section.checks:
            lines.extend(_check_lines(check.status, check.detail, check.hint))
        lines.append("")

    note = _NOTE_WORKFLOW_SKIPPED if _workflow_was_skipped(report) else _NOTE_WORKFLOW_RAN
    lines.append(markup(note, "dim"))
    lines.append("")

    summary = " · ".join(
        [
            pluralize(report.ok_count, "ok", "ok"),
            pluralize(report.warn_count, "warning"),
            pluralize(report.fail_count, "failure"),
        ]
    )
    lines.append(escape(summary))

    return render_lines(*lines, width=RENDER_WIDTH)


def render_doctor_json(report: DoctorReport) -> str:
    """Serialize the report as JSON for machine consumption, keyed as the shipped surface."""
    document = {
        "environment": {
            "gymratVersion": report.environment.gymrat_version,
            "pythonVersion": report.environment.python_version,
            "platform": report.environment.platform,
        },
        "sections": [
            {
                "title": section.title,
                "checks": [_check_json(check) for check in section.checks],
            }
            for section in report.sections
        ],
        "okCount": report.ok_count,
        "warnCount": report.warn_count,
        "failCount": report.fail_count,
        "hasFailures": report.has_failures,
    }
    return json.dumps(document, ensure_ascii=False)


def _check_json(check: Check) -> dict[str, str]:
    payload: dict[str, str] = {
        "name": check.name,
        "status": check.status,
        "detail": check.detail,
    }
    if check.hint is not None:
        payload["hint"] = check.hint
    return payload
