import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import (
    ConfigFile,
    ConfigFileResult,
    load_config_file_collecting,
    validate_config_dict,
)
from gymrat.errors import GymratError
from tests.config._toml import write_config, write_raw

#: A document whose decimal integer literal has more digits than the
#: interpreter converts, so the TOML parser raises a plain ``ValueError``.
DIGIT_LIMIT_DOCUMENT = "samples = 1" + "0" * 5000

#: A document nested deeper than the interpreter's recursion limit, so the TOML
#: parser raises ``RecursionError``.
DEEP_NESTING_DOCUMENT = "bench = " + "[" * 100_000 + "]" * 100_000

# ---------------------------------------------------------------------------
# valid TOML with known keys
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_integral_float_given_does_parse_it_as_an_integer(
    tmp_path: Path,
):
    config_path = write_raw(tmp_path, "samples = 5.0")

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(config_file=ConfigFile(samples=5), exists=True, problems=[])


# ---------------------------------------------------------------------------
# invalid TOML / non-finite literals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(DIGIT_LIMIT_DOCUMENT, id="integer-past-digit-limit"),
        pytest.param(DEEP_NESTING_DOCUMENT, id="nesting-past-recursion-limit"),
    ],
)
def test_load_config_file_collecting_when_parser_rejects_document_does_report_parse_problem(
    tmp_path: Path, document: str
):
    config_path = write_raw(tmp_path, document)

    result = load_config_file_collecting(config_path, required=False)

    assert (result.config_file, result.exists, len(result.problems)) == (None, True, 1)
    assert str(config_path) in result.problems[0]


# ---------------------------------------------------------------------------
# unknown keys
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_key_unknown_does_report_its_path(tmp_path: Path):
    config_path = write_config(tmp_path, {"bench": "name", "badKey": "value"})

    result = load_config_file_collecting(config_path, required=False)

    assert (result.config_file, len(result.problems)) == (None, 1)
    assert result.problems[0].endswith(": badKey")


# ---------------------------------------------------------------------------
# values of the wrong type or out of range
# ---------------------------------------------------------------------------


def _wrong_type(content: dict[str, object], key_path: str, expected: str, label: str) -> object:
    return pytest.param(content, key_path, expected, id=f"{key_path}-{label}")


@pytest.mark.parametrize(
    ("content", "key_path", "expected"),
    [
        _wrong_type({"metrics": 42}, "metrics", "an object", "integer"),
        _wrong_type({"bench": " "}, "bench", "a non-empty string", "space"),
        _wrong_type({"bench": "\u00a0"}, "bench", "a non-empty string", "no-break-space"),
        _wrong_type({"samples": 1.5}, "samples", "an integer", "fraction"),
    ],
)
def test_load_config_file_collecting_when_value_invalid_does_name_key_path_and_expected_shape(
    tmp_path: Path, content: dict[str, object], key_path: str, expected: str
):
    config_path = write_config(tmp_path, content)

    result = load_config_file_collecting(config_path, required=False)

    assert (result.config_file, len(result.problems)) == (None, 1)
    assert key_path in result.problems[0]
    assert expected in result.problems[0]


def test_load_config_file_collecting_when_multiple_fields_invalid_does_report_every_problem(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"bench": 42, "samples": 0})

    result = load_config_file_collecting(config_path, required=False)

    assert result.config_file is None
    assert len(result.problems) == 2
    assert any("bench" in problem for problem in result.problems)
    assert any("samples" in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# section keys
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_section_key_embeds_line_break_does_report_key_not_shape(
    tmp_path: Path,
):
    smuggled = "latency\ndirection: 999, gating: 0"
    config_path = write_config(tmp_path, {"metrics": {smuggled: {"gating": False}}})

    result = load_config_file_collecting(config_path, required=False)

    # The rejected value is an object; only its key is bad, so the message names
    # the escaped key and its line break instead of demanding an object.
    assert (result.config_file, len(result.problems)) == (None, 1)
    assert "metrics" in result.problems[0]
    assert json.dumps(smuggled) in result.problems[0]
    assert "line break" in result.problems[0]


# ---------------------------------------------------------------------------
# validate_config_dict
# ---------------------------------------------------------------------------


def test_validate_config_dict_when_config_valid_does_not_raise():
    validate_config_dict({"bench": "npm run bench"})


@pytest.mark.parametrize(
    ("config", "first", "later"),
    [
        pytest.param(
            {"samples": "bad", "timeout_seconds": "bad", "filter": "npm run bench"},
            "samples",
            "timeout_seconds",
            id="schema",
        ),
        pytest.param(
            {"filter": "npm run bench", "primary": "geomean", "stop": {"target_value": 1.5}},
            "filter",
            "stop.target_value",
            id="loop-keys",
        ),
    ],
)
def test_validate_config_dict_when_several_problems_does_raise_the_first(
    config: dict[str, object], first: str, later: str
):
    with pytest.raises(GymratError) as exc_info:
        validate_config_dict(config)

    message = str(exc_info.value)
    assert (first in message, later in message) == (True, False)


# ---------------------------------------------------------------------------
# missing, unreadable and undecodable files
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_optional_file_missing_does_return_empty_config(
    tmp_path: Path,
):
    missing = tmp_path / "nonexistent.toml"

    result = load_config_file_collecting(missing, required=False)

    assert result == ConfigFileResult(config_file=ConfigFile(), exists=False, problems=[])


def _read_failure_reason(path: Path) -> str:
    # The OS and the codec word these failures themselves (a directory is refused
    # differently on Windows), so the expected reason comes from the running ones.
    try:
        path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return exc.strerror or str(exc)
    except ValueError as exc:
        return str(exc)
    pytest.fail(f"{path} read without error")


def _directory(tmp_path: Path) -> Path:
    return tmp_path


def _utf16_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "gymrat.toml"
    config_path.write_bytes('bench = "hello"'.encode("utf-16"))
    return config_path


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_directory, id="path-is-a-directory"),
        pytest.param(_utf16_file, id="file-is-utf16"),
    ],
)
def test_load_config_file_collecting_when_file_cannot_be_read_does_report_path_and_reason(
    tmp_path: Path, arrange: Callable[[Path], Path]
):
    config_path = arrange(tmp_path)

    result = load_config_file_collecting(config_path, required=False)

    assert (result.config_file, result.exists, len(result.problems)) == (None, True, 1)
    assert str(config_path) in result.problems[0]
    assert result.problems[0].endswith(f": {_read_failure_reason(config_path)}")
