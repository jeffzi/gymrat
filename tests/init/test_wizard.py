"""Tests for the interactive ``init`` wizard.

The wizard is driven with in-memory text streams: a :class:`_Stream` for input
whose ``isatty`` the test pins, and a plain :class:`io.StringIO` for output. That
lets each case assert the prompt text, the re-prompt-on-invalid behavior, the
EOF-cancels-the-wizard distinction, the default-drop of the adapter, and the way
flags pre-answer (and thereby skip) individual prompts.
"""

import io
import math
from typing import override

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.init.wizard import (
    DEFAULT_RUNBOOK_PATH,
    RunbookChoice,
    WizardOptions,
    WizardResult,
    run_wizard,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Stream(io.StringIO):
    """A text stream whose TTY status the test controls."""

    def __init__(self, initial: str = "", *, tty: bool):
        super().__init__(initial)
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


def _non_interactive(**overrides: object) -> WizardOptions:
    """Build options for non-interactive mode (``yes=True`` unless overridden)."""
    overrides.setdefault("yes", True)
    return WizardOptions(input=_Stream("", tty=False), output=io.StringIO(), **overrides)  # type: ignore[arg-type]


def _interactive(lines: list[str], **overrides: object) -> WizardOptions:
    """Build options for interactive mode, feeding ``lines`` as answered prompts."""
    text = "".join(f"{line}\n" for line in lines)
    return WizardOptions(input=_Stream(text, tty=True), output=io.StringIO(), **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# non-interactive mode
# ---------------------------------------------------------------------------


def test_run_wizard_when_bench_via_flag_does_return_settled_defaults():
    result = run_wizard(_non_interactive(bench="npm run bench"))

    assert result == WizardResult(
        bench="npm run bench",
        runbook=RunbookChoice(path=DEFAULT_RUNBOOK_PATH),
        install_skill=True,
    )


def test_run_wizard_when_bench_missing_does_raise_naming_bench_flag():
    with pytest.raises(GymratError, match="--bench"):
        run_wizard(_non_interactive())


def test_run_wizard_when_adapter_flag_does_include_adapter():
    result = run_wizard(_non_interactive(bench="npm run bench", adapter="mitata"))

    assert result.adapter == "mitata"


def test_run_wizard_when_adapter_flag_equals_default_does_drop_it():
    result = run_wizard(_non_interactive(bench="npm run bench", adapter="metric-lines"))

    assert result.adapter is None


def test_run_wizard_when_invalid_adapter_flag_does_raise_naming_value_and_valid_names():
    with pytest.raises(GymratError) as exc:
        run_wizard(_non_interactive(bench="npm run bench", adapter="nope"))

    assert '"nope"' in str(exc.value)
    assert exc.value.hint is not None
    assert "valid adapters are:" in exc.value.hint


def test_run_wizard_when_stop_target_flag_non_finite_does_raise_naming_stop_target():
    with pytest.raises(GymratError, match="stop-target"):
        run_wizard(_non_interactive(bench="npm run bench", stop_target=math.nan, primary="latency"))


def test_run_wizard_when_stop_target_without_primary_does_raise_naming_primary():
    with pytest.raises(GymratError, match="--primary"):
        run_wizard(_non_interactive(bench="npm run bench", stop_target=1.5))


def test_run_wizard_when_stop_target_with_primary_does_include_both():
    result = run_wizard(_non_interactive(bench="npm run bench", stop_target=1.5, primary="latency"))

    assert result.stop_target == 1.5
    assert result.primary == "latency"


def test_run_wizard_when_stop_max_iterations_flag_non_positive_does_raise_naming_flag():
    with pytest.raises(GymratError, match="stop-max-iterations"):
        run_wizard(_non_interactive(bench="npm run bench", stop_max_iterations=0))


@pytest.mark.parametrize(
    ("runbook", "expected"),
    [
        pytest.param(False, False, id="no-runbook-false"),
        pytest.param("custom-runbook.md", RunbookChoice(path="custom-runbook.md"), id="path"),
        pytest.param(True, RunbookChoice(path=DEFAULT_RUNBOOK_PATH), id="bare-true-default"),
    ],
)
def test_run_wizard_when_runbook_flag_varies_does_settle_the_choice(
    runbook: object, expected: object
):
    result = run_wizard(_non_interactive(bench="npm run bench", runbook=runbook))

    assert result.runbook == expected


@pytest.mark.parametrize("skill", [True, False])
def test_run_wizard_when_skill_flag_set_does_settle_install_skill(skill: bool):
    result = run_wizard(_non_interactive(bench="npm run bench", skill=skill))

    assert result.install_skill is skill


def test_run_wizard_when_non_interactive_does_write_no_prompts():
    options = _non_interactive(bench="npm run bench")

    run_wizard(options)

    assert isinstance(options.output, io.StringIO)
    assert options.output.getvalue() == ""


def test_run_wizard_when_bare_primary_without_stop_target_does_include_primary_only():
    result = run_wizard(_non_interactive(bench="npm run bench", primary="latency"))

    assert result.primary == "latency"
    assert result.stop_target is None


def test_run_wizard_when_non_tty_stdin_without_yes_does_behave_non_interactively():
    options = WizardOptions(
        input=_Stream("", tty=False),
        output=io.StringIO(),
        bench="npm run bench",
    )

    result = run_wizard(options)

    assert result == WizardResult(
        bench="npm run bench",
        runbook=RunbookChoice(path=DEFAULT_RUNBOOK_PATH),
        install_skill=True,
    )


# ---------------------------------------------------------------------------
# interactive mode
# ---------------------------------------------------------------------------


def test_run_wizard_when_all_answers_via_prompts_does_settle_each():
    result = run_wizard(
        _interactive(["npm run bench", "y", "metric-lines", "npm run lint", "", "", "y", "", "y"])
    )

    assert result.bench == "npm run bench"
    assert result.checks == "npm run lint"
    assert result.install_skill is True
    assert result.runbook == RunbookChoice(path=DEFAULT_RUNBOOK_PATH)


def test_run_wizard_when_bench_empty_interactively_does_re_prompt_until_non_empty():
    result = run_wizard(
        _interactive(["", "", "npm run bench", "y", "metric-lines", "", "", "", "n", "n"])
    )

    assert result.bench == "npm run bench"


def test_run_wizard_when_adapter_invalid_interactively_does_re_prompt_with_error_and_names():
    options = _interactive(["npm run bench", "y", "nope", "metric-lines", "", "", "", "n", "n"])

    result = run_wizard(options)

    assert result.adapter is None
    assert isinstance(options.output, io.StringIO)
    out = options.output.getvalue()
    assert 'Unknown adapter: "nope".' in out
    assert "valid adapters are:" in out


def test_run_wizard_when_flag_pre_answers_a_question_does_skip_that_prompt():
    result = run_wizard(
        _interactive(["y", "metric-lines", "", "", "", "n", "y"], bench="npm run bench")
    )

    assert result.bench == "npm run bench"


def test_run_wizard_when_stop_target_provided_interactively_does_prompt_for_primary():
    result = run_wizard(
        _interactive(["npm run bench", "y", "metric-lines", "", "1.5", "latency", "3", "n", "y"])
    )

    assert result.stop_target == 1.5
    assert result.primary == "latency"
    assert result.stop_max_iterations == 3


def test_run_wizard_when_stop_target_empty_interactively_does_not_prompt_for_primary():
    result = run_wizard(_interactive(["npm run bench", "y", "metric-lines", "", "", "", "n", "y"]))

    assert result.stop_target is None
    assert result.primary is None


def test_run_wizard_when_primary_is_geomean_interactively_does_re_prompt():
    result = run_wizard(
        _interactive(
            ["npm run bench", "y", "metric-lines", "", "1.5", "geomean", "latency", "", "n", "y"]
        )
    )

    assert result.primary == "latency"


def test_run_wizard_when_runbook_declined_interactively_does_set_runbook_false():
    result = run_wizard(_interactive(["npm run bench", "y", "metric-lines", "", "", "", "n", "y"]))

    assert result.runbook is False


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("abc", id="non-numeric"),
        pytest.param("Infinity", id="infinity"),
        pytest.param("0x10", id="hex"),
        pytest.param("  ", id="whitespace"),
    ],
)
def test_run_wizard_when_stop_target_invalid_interactively_does_re_prompt_until_valid(
    invalid: str,
):
    result = run_wizard(
        _interactive(
            ["npm run bench", "y", "metric-lines", "", invalid, "2.0", "latency", "", "n", "y"]
        )
    )

    assert result.stop_target == 2.0


def test_run_wizard_when_stop_max_iterations_invalid_interactively_does_re_prompt_until_valid():
    result = run_wizard(
        _interactive(
            ["npm run bench", "y", "metric-lines", "", "1.5", "latency", "0", "-1", "3", "n", "y"]
        )
    )

    assert result.stop_max_iterations == 3


@pytest.mark.parametrize("gate", ["y", "Y"])
def test_run_wizard_when_advanced_gate_accepted_does_prompt_for_advanced_settings(gate: str):
    result = run_wizard(
        _interactive(
            ["npm run bench", gate, "mitata", "npm run lint", "1.5", "latency", "3", "n", "y"]
        )
    )

    assert result.adapter == "mitata"
    assert result.checks == "npm run lint"
    assert result.stop_target == 1.5
    assert result.primary == "latency"
    assert result.stop_max_iterations == 3


@pytest.mark.parametrize("gate", ["n", ""])
def test_run_wizard_when_advanced_gate_declined_does_skip_advanced_and_use_defaults(gate: str):
    result = run_wizard(_interactive(["npm run bench", gate, "n", "y"]))

    assert result.adapter is None
    assert result.checks is None
    assert result.stop_target is None
    assert result.primary is None
    assert result.stop_max_iterations is None


def test_run_wizard_when_gate_declined_does_still_honor_flag_supplied_advanced_settings():
    result = run_wizard(
        _interactive(
            ["npm run bench", "n", "n", "y"],
            adapter="mitata",
            stop_target=1.5,
            primary="latency",
        )
    )

    assert result.adapter == "mitata"
    assert result.stop_target == 1.5
    assert result.primary == "latency"


# ---------------------------------------------------------------------------
# EOF (Ctrl-D) cancels the wizard
# ---------------------------------------------------------------------------


def test_run_wizard_when_eof_at_bench_prompt_does_raise_cancelled_not_bench_flag_error():
    with pytest.raises(GymratError) as exc:
        run_wizard(_interactive([]))

    message = str(exc.value)
    assert "Cancelled." in message
    assert "--bench" not in message


@pytest.mark.parametrize(
    ("lines", "overrides"),
    [
        pytest.param(["npm run bench"], {}, id="advanced-gate"),
        pytest.param(["npm run bench", "y"], {}, id="adapter-after-gate"),
        pytest.param(
            ["y", "metric-lines", ""],
            {"bench": "npm run bench", "stop_target": 1.5},
            id="primary-metric",
        ),
    ],
)
def test_run_wizard_when_eof_at_a_later_prompt_does_raise_cancelled(
    lines: list[str], overrides: dict[str, object]
):
    with pytest.raises(GymratError, match="Cancelled."):
        run_wizard(_interactive(lines, **overrides))
