"""Behavioral tests for the one-line summaries of event text and tool input.

``summarize`` collapses text and truncates it with a bare ellipsis at
``SUMMARY_MAX_CHARS`` by default; ``summarize_input`` summarizes a tool's
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
# SUMMARY_MAX_CHARS
# ---------------------------------------------------------------------------


def test_summarize_when_called_without_max_chars_does_truncate_to_summary_max_chars():
    overflow = 50
    text = "a" * (SUMMARY_MAX_CHARS + overflow)

    result = summarize(text)

    assert result == "a" * SUMMARY_MAX_CHARS + "…"


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
    ],
)
def test_summarize_when_within_budget_does_return_collapsed_text(text: str, expected: str):
    assert summarize(text, 100) == expected


def test_summarize_when_over_budget_does_truncate_with_bare_ellipsis():
    overflow = 250
    max_chars = 50

    result = summarize("a" * (max_chars + overflow), max_chars)

    assert result == "a" * max_chars + "…"


def test_summarize_when_multiline_over_budget_does_truncate_with_bare_ellipsis():
    result = summarize("line1\nline2\nline3\nline4", 20)

    assert result == "line1 line2 line3 li…"


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        pytest.param(
            "\U0001f3af" * 8,
            5,
            "\U0001f3af\U0001f3af\U0001f3af\U0001f3af\U0001f3af…",
            id="all-emoji",
        ),
        pytest.param(  # cspell:disable-next-line
            "ab\U0001f3af\U0001f3afcd\U0001f3af", 3, "ab\U0001f3af…", id="mixed-width"
        ),
    ],
)
def test_summarize_when_truncating_does_split_on_code_point_boundaries(
    text: str, max_chars: int, expected: str
):
    assert summarize(text, max_chars) == expected


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
    assert summarize_input(value, 200) == expected


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
    ],
)
def test_summarize_input_when_file_tool_does_extract_path_only(
    tool_name: str, tool_input: dict[str, object], expected: str
):
    assert summarize_input(tool_input, tool_name=tool_name) == expected


def test_summarize_input_when_path_under_root_does_render_relative():
    result = summarize_input(
        {"file_path": "/project/src/main.py"},
        tool_name="Read",
        supervised_root="/project",
    )

    assert result == "src/main.py"


def test_summarize_input_when_path_under_home_does_render_tilde_prefixed():
    home = str(Path.home())

    result = summarize_input(
        {"file_path": f"{home}/Documents/notes.md"},
        tool_name="Read",
        supervised_root="/other/project",
    )

    assert result == "~/Documents/notes.md"


def test_summarize_input_when_path_under_root_and_home_does_prefer_root_relative():
    home = str(Path.home())
    root = f"{home}/project"

    result = summarize_input(
        {"file_path": f"{root}/src/main.py"},
        tool_name="Read",
        supervised_root=root,
    )

    assert result == "src/main.py"


def test_summarize_input_when_path_outside_root_and_home_does_render_verbatim():
    result = summarize_input(
        {"file_path": "/etc/config.ini"},
        tool_name="Read",
        supervised_root="/project",
    )

    assert result == "/etc/config.ini"


def test_summarize_input_when_bash_does_extract_command():
    result = summarize_input({"command": "echo hello"}, tool_name="Bash")

    assert result == "echo hello"


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param(
            "Agent",
            {
                "subagent_type": "Explore",
                "description": "Explore ECS source architecture",
                "prompt": "long prompt body that should never appear",
            },
            id="agent-subagent-type",
        ),
        pytest.param(
            "Task",
            {
                "type": "Explore",
                "description": "Explore ECS source architecture",
                "prompt": "long prompt body that should never appear",
            },
            id="task-type",
        ),
    ],
)
def test_summarize_input_when_agent_or_task_does_extract_type_and_description(
    tool_name: str, tool_input: dict[str, object]
):
    result = summarize_input(tool_input, tool_name=tool_name)

    assert result == "Explore: Explore ECS source architecture"


def test_summarize_input_when_agent_has_no_subagent_type_does_show_description_only():
    result = summarize_input(
        {"description": "Explore ECS source architecture", "prompt": "..."},
        tool_name="Agent",
    )

    assert result == "Explore ECS source architecture"


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        pytest.param("Skill", {"skill": "gymrat"}, "gymrat", id="skill-name-only"),
        pytest.param(
            "Skill",
            {"skill": "gymrat", "args": "some args"},
            "gymrat some args",
            id="skill-name-with-args",
        ),
    ],
)
def test_summarize_input_when_skill_tool_does_extract_skill_and_args(
    tool_name: str, tool_input: dict[str, object], expected: str
):
    assert summarize_input(tool_input, tool_name=tool_name) == expected


# ---------------------------------------------------------------------------
# summarize_input — gymrat MCP tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param({}, id="no-payload"),
        pytest.param({"some": "data"}, id="with-payload"),
    ],
)
def test_summarize_input_when_iterate_tool_does_return_gymrat_iterate(
    tool_input: dict[str, object],
):
    result = summarize_input(tool_input, tool_name=ITERATE_TOOL)

    assert result == "gymrat iterate"


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

    result = summarize_input(tool_input, tool_name=PROBE_TOOL)

    assert len(result) <= SUMMARY_MAX_CHARS + 1  # +1 for the ellipsis character
    assert result.startswith("gymrat probe exercise_0")
    assert result.endswith("…")


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param("Read", {"not_file_path": "/a.py"}, id="read-missing-file-path"),
        pytest.param("Bash", {"not_command": "echo"}, id="bash-missing-command"),
        pytest.param("Agent", {"prompt": "..."}, id="agent-missing-type"),
        pytest.param("Skill", {"not_skill": "foo"}, id="skill-missing-skill"),
    ],
)
def test_summarize_input_when_expected_field_missing_does_fall_back_to_json(
    tool_name: str, tool_input: dict[str, object]
):
    result = summarize_input(tool_input, tool_name=tool_name)

    assert result == json.dumps(tool_input, separators=(",", ":"))


def test_summarize_input_when_unknown_tool_does_fall_back_to_json():
    tool_input = {"some_key": "some_value"}

    result = summarize_input(tool_input, tool_name="UnknownTool")

    assert result == '{"some_key":"some_value"}'
