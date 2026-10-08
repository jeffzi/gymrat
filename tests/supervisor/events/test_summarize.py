"""Behavioral tests for the one-line summaries of event text and tool input.

``summarize`` collapses text and truncates it with a bare ellipsis at
``SUMMARY_MAX_CHARS``; ``summarize_input`` summarizes a tool's
input by its JSON form, extracting the one field a known tool is identified by
(a file path, a command, an agent type, a skill, or gymrat's own MCP tools).
"""

import json
from pathlib import Path

import pytest

from gymrat.supervisor.events import (
    ITERATE_TOOL,
    PROBE_TOOL,
    SUMMARY_MAX_CHARS,
    summarize,
    summarize_input,
)
from tests.supervisor._fixtures import NotJsonEncodable

# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("short text", "short text", id="plain-fit"),
        pytest.param("hello   world", "hello world", id="internal-whitespace-collapsed"),
        pytest.param("  trimmed  ", "trimmed", id="leading-trailing-trimmed"),
        pytest.param("line1\nline2\nline3", "line1 line2 line3", id="newlines-collapsed"),
        pytest.param("a" * SUMMARY_MAX_CHARS, "a" * SUMMARY_MAX_CHARS, id="exactly-at-budget"),
    ],
)
def test_summarize_when_within_budget_does_return_collapsed_text(text: str, expected: str):
    assert summarize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "a" * (SUMMARY_MAX_CHARS + 250), "a" * SUMMARY_MAX_CHARS + "…", id="ascii-over-budget"
        ),
        pytest.param(
            "\n".join(["line"] * SUMMARY_MAX_CHARS),
            " ".join(["line"] * SUMMARY_MAX_CHARS)[:SUMMARY_MAX_CHARS] + "…",
            id="multiline-collapsed-then-truncated",
        ),
        pytest.param(
            "\U0001f3af" * (SUMMARY_MAX_CHARS + 3),
            "\U0001f3af" * SUMMARY_MAX_CHARS + "…",
            id="all-emoji",
        ),
    ],
)
def test_summarize_when_over_budget_does_truncate_with_bare_ellipsis(text: str, expected: str):
    assert summarize(text) == expected


# ---------------------------------------------------------------------------
# summarize_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"key": "value"}, '{"key":"value"}', id="dict-no-spaces"),
        pytest.param(None, "null", id="none-to-json-null"),
        pytest.param(NotJsonEncodable(), "not-json-encodable", id="non-serializable-str-fallback"),
    ],
)
def test_summarize_input_when_given_value_does_summarize_its_json_form(
    value: object, expected: str
):
    assert summarize_input(value) == expected


# ---------------------------------------------------------------------------
# summarize_input — tool-specific extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        pytest.param("Read", {"file_path": "/a/b.py"}, "/a/b.py", id="read"),
        pytest.param(
            "Edit",
            {"file_path": "/a/b.py", "old_string": "x", "new_string": "y"},
            "/a/b.py",
            id="edit",
        ),
        pytest.param("Write", {"file_path": "/a/b.py", "content": "..."}, "/a/b.py", id="write"),
        pytest.param(
            "MultiEdit",
            {"file_path": "/a/b.py", "edits": [{"old_string": "x", "new_string": "y"}]},
            "/a/b.py",
            id="multi-edit",
        ),
        pytest.param(
            "NotebookEdit",
            {"notebook_path": "/a/nb.ipynb"},
            "/a/nb.ipynb",
            id="notebook-edit",
        ),
        pytest.param("Bash", {"command": "echo hello"}, "echo hello", id="bash-command"),
        pytest.param(
            "Agent",
            {
                "subagent_type": "Explore",
                "description": "Explore ECS source architecture",
                "prompt": "long prompt body that should never appear",
            },
            "Explore: Explore ECS source architecture",
            id="agent-subagent-type",
        ),
        pytest.param(
            "Task",
            {
                "type": "Explore",
                "description": "Explore ECS source architecture",
                "prompt": "long prompt body that should never appear",
            },
            "Explore: Explore ECS source architecture",
            id="task-type",
        ),
        pytest.param(
            "Agent",
            {"description": "Explore ECS source architecture", "prompt": "..."},
            "Explore ECS source architecture",
            id="agent-without-type-shows-description-only",
        ),
        pytest.param("Skill", {"skill": "gymrat"}, "gymrat", id="skill-name-only"),
        pytest.param(
            "Skill",
            {"skill": "gymrat", "args": "some args"},
            "gymrat some args",
            id="skill-name-with-args",
        ),
        pytest.param(ITERATE_TOOL, {"some": "data"}, "gymrat iterate", id="gymrat-iterate"),
    ],
)
def test_summarize_input_when_known_tool_does_extract_its_identifying_fields(
    tool_name: str, tool_input: dict[str, object], expected: str
):
    assert summarize_input(tool_input, tool_name=tool_name) == expected


@pytest.mark.parametrize(
    ("file_path", "supervised_root", "expected"),
    [
        pytest.param("/project/src/main.py", "/project", "src/main.py", id="under-root"),
        pytest.param(
            "{home}/Documents/notes.md",
            "/other/project",
            "~/Documents/notes.md",
            id="under-home",
        ),
        pytest.param(
            "{home}/project/src/main.py",
            "{home}/project",
            "src/main.py",
            id="under-root-and-home-prefers-root",
        ),
        pytest.param("/etc/config.ini", "/project", "/etc/config.ini", id="outside-both-verbatim"),
        pytest.param(
            "/project/..cache/data.json",
            "/project",
            "..cache/data.json",
            id="two-dot-directory-under-root",
        ),
        pytest.param(
            "{home}/work/sibling/main.py",
            "{home}/work/project",
            "~/work/sibling/main.py",
            id="sibling-of-root-under-home",
        ),
        pytest.param(
            "{home}/work", "{home}/work/project", "~/work", id="parent-of-root-under-home"
        ),
    ],
)
def test_summarize_input_when_file_path_given_does_render_it_from_root_or_home(
    file_path: str, supervised_root: str, expected: str
):
    home = str(Path.home())

    result = summarize_input(
        {"file_path": file_path.format(home=home)},
        tool_name="Read",
        supervised_root=supervised_root.format(home=home),
    )

    assert result == expected


# ---------------------------------------------------------------------------
# summarize_input — gymrat MCP tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [
        pytest.param(
            {"names": ["bench", "squat"], "samples": 6},
            "gymrat probe bench squat --samples 6",
            id="names-and-samples",
        ),
        pytest.param(
            {"names": ["bench"]},
            "gymrat probe bench",
            id="single-name-no-samples",
        ),
        pytest.param(
            {},
            "gymrat probe",
            id="empty-dict",
        ),
        pytest.param(
            {"names": "not-a-list"},
            "gymrat probe",
            id="names-not-a-list-dropped",
        ),
        pytest.param(
            {"names": [1, 2]},
            "gymrat probe",
            id="names-not-strings-dropped",
        ),
        pytest.param(
            {"names": ["a"], "samples": "five"},
            "gymrat probe a",
            id="samples-not-int-dropped",
        ),
    ],
)
def test_summarize_input_when_probe_tool_does_build_cli_summary(
    tool_input: dict[str, object], expected: str
):
    result = summarize_input(tool_input, tool_name=PROBE_TOOL)

    assert result == expected


def test_summarize_input_when_probe_names_long_does_truncate_via_length_cap():
    long_names = [f"exercise_{i}" for i in range(100)]
    tool_input: dict[str, object] = {"names": long_names}
    full_command = " ".join(["gymrat", "probe", *long_names])

    result = summarize_input(tool_input, tool_name=PROBE_TOOL)

    assert result == f"{full_command[:SUMMARY_MAX_CHARS]}…"


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param("Read", {"not_file_path": "/a.py"}, id="read-missing-file-path"),
        pytest.param("Bash", {"not_command": "echo"}, id="bash-missing-command"),
        pytest.param("Agent", {"prompt": "..."}, id="agent-missing-description"),
        pytest.param("Skill", {"not_skill": "foo"}, id="skill-missing-skill"),
        pytest.param("UnknownTool", {"some_key": "some_value"}, id="unknown-tool"),
    ],
)
def test_summarize_input_when_no_known_field_to_extract_does_fall_back_to_json(
    tool_name: str, tool_input: dict[str, object]
):
    result = summarize_input(tool_input, tool_name=tool_name)

    assert result == json.dumps(tool_input, separators=(",", ":"))
