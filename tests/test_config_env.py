import pytest

from gymrat_py.config_env import EnvResult, env_positive_int_result

# Every GYMRAT_* variable used by these tests. Cleared before each test so an
# ambient shell value cannot bleed in.
GYMRAT_ENV_VARS = ("GYMRAT_SAMPLES",)


@pytest.fixture(autouse=True)
def _clear_gymrat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in GYMRAT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


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
