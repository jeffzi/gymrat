import pytest

from gymrat.config_env import EnvResult, env_positive_int_result

# ---------------------------------------------------------------------------
# positive int overflow (CPython integer-string conversion limit)
# ---------------------------------------------------------------------------


def test_env_positive_int_result_when_digit_string_exceeds_conversion_limit_does_report_problem(
    monkeypatch: pytest.MonkeyPatch,
):
    huge = "1" * 4301
    monkeypatch.setenv("GYMRAT_SAMPLES", huge)

    result = env_positive_int_result("GYMRAT_SAMPLES")

    assert isinstance(result, EnvResult)
    assert result.problem is not None
    assert result.value is None
