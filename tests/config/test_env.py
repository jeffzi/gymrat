import pytest

from gymrat.config.env import EnvResult, env_positive_int_result, is_positive_integer

_ENV_VAR = "GYMRAT_SAMPLES"
_CEILING = 10


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(" 5", id="leading-space"),
        pytest.param("5 ", id="trailing-space"),
        pytest.param("+5", id="plus-sign"),
        pytest.param("1_000", id="underscore"),
        pytest.param("\N{ARABIC-INDIC DIGIT FIVE}", id="non-ascii-digit"),
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
    ],
)
def test_is_positive_integer_when_not_bare_positive_digits_does_reject(raw: str):
    assert is_positive_integer(raw) is False


def test_env_positive_int_result_when_digit_string_exceeds_conversion_limit_does_report_problem(
    monkeypatch: pytest.MonkeyPatch,
):
    huge = "1" * 4301
    monkeypatch.setenv("GYMRAT_SAMPLES", huge)

    result = env_positive_int_result("GYMRAT_SAMPLES")

    assert isinstance(result, EnvResult)
    assert result.problem is not None
    assert result.value is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [pytest.param("1", 1, id="one"), pytest.param("10", _CEILING, id="ceiling")],
)
def test_env_positive_int_result_when_bare_digits_within_ceiling_does_accept(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
):
    monkeypatch.setenv(_ENV_VAR, raw)

    result = env_positive_int_result(_ENV_VAR, maximum=_CEILING)

    assert (result.value, result.problem) == (expected, None)


@pytest.mark.parametrize(
    "raw",
    [pytest.param(" 5", id="malformed"), pytest.param("11", id="above-ceiling")],
)
def test_env_positive_int_result_when_malformed_or_above_ceiling_does_report_problem(
    monkeypatch: pytest.MonkeyPatch, raw: str
):
    monkeypatch.setenv(_ENV_VAR, raw)

    result = env_positive_int_result(_ENV_VAR, maximum=_CEILING)

    assert result.value is None
    assert result.problem is not None
