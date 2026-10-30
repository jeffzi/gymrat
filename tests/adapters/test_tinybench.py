import re

import pytest

from gymrat.adapters.contract import TIME_SUFFIX
from gymrat.adapters.tinybench import tinybench_adapter
from tests.adapters._inputs import VALID_TASK, build_tinybench_stdout, task

# ---------------------------------------------------------------------------
# basic parsing
# ---------------------------------------------------------------------------


def test_parse_when_median_is_zero_does_emit_zero_with_one_warning():
    stdout = build_tinybench_stdout([task("noop", p50=0)])
    warnings: list[str] = []

    result = tinybench_adapter.parse(stdout, warnings.append)

    assert result == {f"noop{TIME_SUFFIX}": 0}
    assert len(warnings) == 1
    assert '"noop"' in warnings[0]


# ---------------------------------------------------------------------------
# skipped tasks
# ---------------------------------------------------------------------------

# Each row pairs a task with a regex for the computed parts of its skip warning.
_NOT_COMPLETED_TASKS = [
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
    pytest.param(task("t", result={"state": "errored"}), r'"t"$', id="errored-missing"),
]

_MALFORMED_TASKS = [
    pytest.param({"name": "t"}, r'"t".*result$', id="result-missing"),
    pytest.param(
        task("t", result={"state": "completed"}), r'"t".*result\.latency$', id="latency-missing"
    ),
    pytest.param(
        task("t", p50=1e305),
        r'"t".*result\.latency\.p50',
        id="p50-overflows-on-conversion",
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
