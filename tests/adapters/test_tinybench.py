import json
import re
from pathlib import Path

import pytest

from gymrat.adapters.contract import TIME_SUFFIX, AdapterError
from gymrat.adapters.tinybench import tinybench_adapter
from tests.adapters._inputs import VALID_TASK, build_tinybench_stdout, task

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tinybench.txt"

# ---------------------------------------------------------------------------
# basic parsing
# ---------------------------------------------------------------------------


def test_parse_when_tasks_completed_does_emit_one_time_metric_per_task_in_nanoseconds_keeping_integers():
    stdout = build_tinybench_stdout([task("parse/small", p50=0.5), task("encode", p50=2)])

    result = tinybench_adapter.parse(stdout)

    assert result == {f"parse/small{TIME_SUFFIX}": 500_000, f"encode{TIME_SUFFIX}": 2_000_000}
    assert type(result[f"encode{TIME_SUFFIX}"]) is int


def test_parse_when_median_is_zero_does_emit_zero_with_one_warning():
    stdout = build_tinybench_stdout([task("noop", p50=0)])
    warnings: list[str] = []

    result = tinybench_adapter.parse(stdout, warnings.append)

    assert result == {f"noop{TIME_SUFFIX}": 0}
    assert len(warnings) == 1
    assert '"noop"' in warnings[0]


# ---------------------------------------------------------------------------
# locating the payload in stdout
# ---------------------------------------------------------------------------

_PAYLOAD = build_tinybench_stdout([task("encode", p50=0.5)])


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param(
            f"{build_tinybench_stdout([task('logged', p50=9)])}\n{_PAYLOAD}",
            id="earlier-tinybench-carrier",
        ),
        pytest.param(
            f"{_PAYLOAD}\n{json.dumps({'tinybench': 'done'})}",
            id="later-tinybench-key-not-a-list",
        ),
    ],
)
def test_parse_when_payload_surrounded_by_other_output_does_read_the_last_carrier(stdout: str):
    assert tinybench_adapter.parse(stdout) == {f"encode{TIME_SUFFIX}": 500_000}


# ---------------------------------------------------------------------------
# skipped tasks
# ---------------------------------------------------------------------------

# Each row pairs a task with a regex for the computed parts of its skip warning.
_NOT_COMPLETED_TASKS = [
    pytest.param(task("t", state="banana"), r'"t".*"banana"', id="unknown-state"),
    pytest.param(task("t", result={"state": None}), r'"t".*null', id="state-null"),
    pytest.param(
        task("enc#ode", state="aborted"),
        r'"enc#ode".*"aborted"',
        id="not-completed-name-with-hash",
    ),
    pytest.param(
        task("t", result={"latency": {"p50": 0.5}}), r'"t".*result\.state', id="state-missing"
    ),
    pytest.param(
        task("t", result={"state": "errored", "error": "boom"}),
        r'"t".*boom',
        id="errored-string",
    ),
    pytest.param(
        task("t", result={"state": "errored", "error": {}}), r'"t"$', id="errored-empty-object"
    ),
    pytest.param(
        task("t", result={"state": "errored", "error": ""}), r'"t"$', id="errored-empty-string"
    ),
    pytest.param(task("t", result={"state": "errored"}), r'"t"$', id="errored-missing"),
]

_MALFORMED_TASKS = [
    pytest.param(None, r"null$", id="task-null"),
    pytest.param(
        {"name": 42, "result": {"state": "completed", "latency": {"p50": 1}}},
        r"42$",
        id="name-not-string",
    ),
    pytest.param(
        {"result": {"state": "completed", "latency": {"p50": 1}}}, r"name$", id="name-missing"
    ),
    pytest.param({"name": "t"}, r'"t".*result$', id="result-missing"),
    pytest.param({"name": "t", "result": None}, r'"t".*result.*null$', id="result-null"),
    pytest.param(
        task("t", result={"state": "completed"}), r'"t".*result\.latency$', id="latency-missing"
    ),
    pytest.param(
        task("t", result={"state": "completed", "latency": 5}),
        r'"t".*result\.latency.*5$',
        id="latency-not-object",
    ),
    pytest.param(
        task("t", result={"state": "completed", "latency": {}}),
        r'"t".*result\.latency\.p50$',
        id="p50-missing",
    ),
    pytest.param(task("t", p50=True), r'"t".*result\.latency\.p50.*true$', id="p50-bool"),
    pytest.param(
        task("t", p50=float("inf")), r'"t".*result\.latency\.p50.*Infinity$', id="p50-infinity"
    ),
]

_BAD_NAME_TASKS = [
    pytest.param(
        task("enc\u2028ode"),
        r'"enc\\u2028ode"',
        id="line-terminator",
    ),
    pytest.param(task("a//b", p50=0), r'"a//b"', id="empty-path-segment-zero-median"),
]


@pytest.mark.parametrize(
    ("bad_task", "pattern"), _NOT_COMPLETED_TASKS + _MALFORMED_TASKS + _BAD_NAME_TASKS
)
def test_parse_when_task_is_skipped_does_skip_only_that_task_with_one_warning(
    bad_task: object, pattern: str, capsys: pytest.CaptureFixture[str]
):
    stdout = build_tinybench_stdout([bad_task, VALID_TASK])
    warnings: list[str] = []

    result = tinybench_adapter.parse(stdout, warnings.append)

    assert result == {f"valid{TIME_SUFFIX}": 500_000}
    assert len(warnings) == 1
    assert re.search(pattern, warnings[0])
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# task names
# ---------------------------------------------------------------------------


def test_parse_when_task_name_contains_hash_does_raise_adapter_error_naming_the_task():
    stdout = build_tinybench_stdout([task("enc#ode")])

    with pytest.raises(AdapterError) as exc_info:
        tinybench_adapter.parse(stdout)

    assert str(exc_info.value).startswith('Task name "enc#ode"')


def test_parse_when_task_names_collide_does_keep_last_value_with_one_warning():
    stdout = build_tinybench_stdout([task("encode", p50=0.5), task("encode", p50=2)])
    warnings: list[str] = []

    result = tinybench_adapter.parse(stdout, warnings.append)

    assert result == {f"encode{TIME_SUFFIX}": 2_000_000}
    assert len(warnings) == 1
    assert f"encode{TIME_SUFFIX}" in warnings[0]
    assert "give the tasks distinct names" in warnings[0]


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_parse_when_last_tinybench_array_is_empty_does_raise_adapter_error():
    stdout = f"{build_tinybench_stdout([VALID_TASK])}\n{build_tinybench_stdout([])}"

    with pytest.raises(AdapterError, match=r"tinybench array"):
        tinybench_adapter.parse(stdout)


# ---------------------------------------------------------------------------
# real fixture
# ---------------------------------------------------------------------------


def test_parse_when_given_real_fixture_does_read_only_the_completed_task():
    stdout = _FIXTURE_PATH.read_text()
    warnings: list[str] = []

    result = tinybench_adapter.parse(stdout, warnings.append)

    assert result == pytest.approx({f"parse/small{TIME_SUFFIX}": 207.99999998644125})
    assert len(warnings) == 1
    assert '"explode"' in warnings[0]
