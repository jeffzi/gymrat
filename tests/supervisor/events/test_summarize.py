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
    ITERATE_SUMMARY,
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


def test_summarize_when_exactly_at_budget_does_return_the_text_whole():
    text = "a" * SUMMARY_MAX_CHARS

    result = summarize(text)

    assert result == text


def test_summarize_when_over_budget_does_truncate_with_bare_ellipsis():
    text = "\n".join(["line"] * SUMMARY_MAX_CHARS)

    result = summarize(text)

    assert result == " ".join(["line"] * SUMMARY_MAX_CHARS)[:SUMMARY_MAX_CHARS] + "…"


# ---------------------------------------------------------------------------
# summarize_input
# ---------------------------------------------------------------------------


def test_summarize_input_when_value_not_json_encodable_does_fall_back_to_its_str():
    result = summarize_input(NotJsonEncodable())

    assert result == "not-json-encodable"


# ---------------------------------------------------------------------------
# summarize_input — tool-specific extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        pytest.param("Read", {"file_path": "/a/b.py"}, "/a/b.py", id="read"),
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
        pytest.param(ITERATE_TOOL, {"some": "data"}, ITERATE_SUMMARY, id="gymrat-iterate"),
    ],
)
def test_summarize_input_when_known_tool_does_extract_its_identifying_fields(
    tool_name: str, tool_input: dict[str, object], expected: str
):
    result = summarize_input(tool_input, tool_name=tool_name)

    assert result == expected


@pytest.mark.parametrize(
    ("file_path", "supervised_root", "expected"),
    [
        pytest.param(
            "{home}/project/src/main.py",
            "{home}/project",
            "src/main.py",
            id="under-root-and-home-prefers-root",
        ),
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
    file_path: str,
    supervised_root: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
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
