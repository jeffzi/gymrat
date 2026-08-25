"""The interactive ``init`` wizard that settles a scaffold's answers.

Each answer is settled from its flag, from an interactive prompt, or from a
default under ``--yes`` / a non-TTY stdin. The wizard reads lines synchronously
from its input stream: a returned ``""`` is end-of-stream (Ctrl-D), which cancels
the whole wizard, while an empty line falls back to the prompt's default. Advanced
settings hide behind a gate; the runbook and skill questions are always asked
interactively. The settled :class:`WizardResult` is what :func:`scaffold` writes.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TextIO

import typer

from gymrat_py.adapters import ADAPTER_NAMES, get_adapter
from gymrat_py.cli.shared import parse_positive_integer_up_to, parse_stop_target_value
from gymrat_py.config import CONFIG_DEFAULTS, GEOMEAN_PRIMARY
from gymrat_py.errors import GymratError, message_of

#: The runbook filename scaffolded when the wizard creates one without a supplied path.
DEFAULT_RUNBOOK_PATH = "gymrat-runbook.md"

#: Upper bound on the interactive max-iterations answer, mirroring the flag coercer.
_MAX_ITERATIONS = 2**53 - 1

type _Validator = Callable[[str], str | None]

_CANCELLED = "Cancelled."
_MISSING_BENCH = "Missing --bench flag."
_MISSING_PRIMARY = "Missing --primary flag."
_MISSING_PRIMARY_HINT = "--stop-target requires --primary to name a metric."
_INVALID_STOP_TARGET = "Invalid --stop-target: not a number."
_INVALID_MAX_ITERATIONS = "Invalid --stop-max-iterations: must be an integer >= 1."


@dataclass(frozen=True, slots=True)
class RunbookChoice:
    """A settled decision to create a runbook at ``path``."""

    path: str


@dataclass(frozen=True, slots=True)
class WizardOptions:
    """Pre-filled answers and the I/O streams the wizard prompts over.

    Any field left ``None`` is prompted for interactively unless ``yes`` is set,
    in which case a default is applied instead. ``runbook`` is tri-state: ``False``
    skips the runbook, ``True`` writes it to :data:`DEFAULT_RUNBOOK_PATH`, and a
    string writes it to that path.
    """

    input: TextIO
    output: TextIO
    bench: str | None = None
    adapter: str | None = None
    checks: str | None = None
    stop_target: float | None = None
    stop_max_iterations: int | None = None
    primary: str | None = None
    runbook: str | bool | None = None
    skill: bool | None = None
    yes: bool = False


@dataclass(frozen=True, slots=True)
class WizardResult:
    """The settled wizard output that :func:`scaffold` turns into files.

    Unlike :class:`WizardOptions`, ``runbook`` is resolved to either ``False``
    (skip) or a concrete :class:`RunbookChoice`, with no tri-state left.
    """

    bench: str
    runbook: RunbookChoice | Literal[False]
    install_skill: bool
    adapter: str | None = None
    checks: str | None = None
    stop_target: float | None = None
    stop_max_iterations: int | None = None
    primary: str | None = None


class _Wizard:
    """One run of the wizard over a fixed set of options and streams."""

    def __init__(self, options: WizardOptions) -> None:
        self._options = options
        self._input = options.input
        self._output = options.output
        self._interactive = (not options.yes) and options.input.isatty()

    def run(self) -> WizardResult:
        bench = self._settle_bench()
        advanced = self._settle_advanced_gate()

        adapter = self._settle_adapter(advanced=advanced)
        checks = self._settle_checks(advanced=advanced)
        stop_target = self._settle_stop_target(advanced=advanced)
        primary = self._settle_primary(stop_target, advanced=advanced)
        stop_max_iterations = self._settle_max_iterations(advanced=advanced)
        runbook = self._settle_runbook()
        install_skill = self._settle_install_skill()

        return WizardResult(
            bench=bench,
            runbook=runbook,
            install_skill=install_skill,
            adapter=adapter if adapter and adapter != CONFIG_DEFAULTS.adapter else None,
            checks=checks or None,
            stop_target=stop_target,
            primary=primary or None,
            stop_max_iterations=stop_max_iterations,
        )

    # -- prompt loop --------------------------------------------------------

    def _ask(
        self,
        question: str,
        *,
        default: str | None = None,
        validate: _Validator | None = None,
    ) -> str | None:
        """Prompt once (re-prompting on validation failure) and return the answer.

        An empty line yields ``default``; end-of-stream cancels the wizard.
        """
        suffix = f" [{default}]" if default is not None else ""
        prompt = f"{question}{suffix} "
        while True:
            self._output.write(prompt)
            raw = self._input.readline()
            if raw == "":
                raise GymratError(_CANCELLED)
            answer = raw.rstrip("\r\n")
            if answer == "":
                return default
            if validate is not None:
                error = validate(answer)
                if error is not None:
                    self._output.write(f"{error}\n")
                    continue
            return answer

    # -- per-field settlement ----------------------------------------------

    def _settle_bench(self) -> str:
        bench = self._options.bench
        if not bench and self._interactive:
            while not bench:
                bench = self._ask("Bench command:")
        if not bench:
            raise GymratError(_MISSING_BENCH)
        return bench

    def _settle_advanced_gate(self) -> bool:
        if not self._interactive:
            return False
        return self._ask("Configure advanced settings? (y/N)") in ("y", "Y")

    def _settle_adapter(self, *, advanced: bool) -> str | None:
        adapter = self._options.adapter
        if adapter is not None:
            get_adapter(adapter)
            return adapter
        if advanced:
            return self._ask(
                f"Adapter ({', '.join(ADAPTER_NAMES)}):",
                default=CONFIG_DEFAULTS.adapter,
                validate=self._validate_adapter,
            )
        return None

    def _settle_checks(self, *, advanced: bool) -> str | None:
        if self._options.checks is not None:
            return self._options.checks
        if advanced:
            return self._ask("Checks command (optional):")
        return None

    def _settle_stop_target(self, *, advanced: bool) -> float | None:
        flag = self._options.stop_target
        if flag is not None:
            if not math.isfinite(flag):
                raise GymratError(_INVALID_STOP_TARGET)
            return flag
        if advanced:
            self._output.write("Stop the loop when the primary metric reaches this threshold.\n")
            raw = self._ask("Stop target (optional):", validate=self._validate_stop_target)
            if raw:
                return float(raw)
        return None

    def _settle_primary(self, stop_target: float | None, *, advanced: bool) -> str | None:
        primary = self._options.primary
        if stop_target is not None and not primary:
            if advanced:
                while not primary:
                    primary = self._ask("Primary metric:", validate=self._validate_primary)
            if not primary:
                raise GymratError(_MISSING_PRIMARY, hint=_MISSING_PRIMARY_HINT)
        return primary

    def _settle_max_iterations(self, *, advanced: bool) -> int | None:
        flag = self._options.stop_max_iterations
        if flag is not None:
            if flag < 1:
                raise GymratError(_INVALID_MAX_ITERATIONS)
            return flag
        if advanced:
            raw = self._ask("Max iterations (optional):", validate=self._validate_max_iterations)
            if raw:
                return int(raw)
        return None

    def _settle_runbook(self) -> RunbookChoice | Literal[False]:
        runbook = self._options.runbook
        if runbook is False:
            return False
        if runbook is True:
            return RunbookChoice(path=DEFAULT_RUNBOOK_PATH)
        if isinstance(runbook, str):
            return RunbookChoice(path=runbook)
        if self._interactive:
            if self._ask("Create runbook? (y/N)") in ("y", "Y"):
                path = self._ask("Runbook path:", default=DEFAULT_RUNBOOK_PATH)
                return RunbookChoice(path=path or DEFAULT_RUNBOOK_PATH)
            return False
        return RunbookChoice(path=DEFAULT_RUNBOOK_PATH)

    def _settle_install_skill(self) -> bool:
        if self._options.skill is not None:
            return self._options.skill
        if self._interactive:
            return self._ask("Install skill? (y/N)") in ("y", "Y")
        return True

    # -- validators ---------------------------------------------------------

    @staticmethod
    def _validate_adapter(name: str) -> str | None:
        try:
            get_adapter(name)
        except GymratError as err:
            return " ".join(part for part in (message_of(err), err.hint) if part)
        return None

    @staticmethod
    def _validate_stop_target(value: str) -> str | None:
        try:
            parse_stop_target_value(value)
        except typer.BadParameter as err:
            return str(err.message)
        return None

    @staticmethod
    def _validate_max_iterations(value: str) -> str | None:
        try:
            parse_positive_integer_up_to(_MAX_ITERATIONS)(value)
        except typer.BadParameter as err:
            return str(err.message)
        return None

    @staticmethod
    def _validate_primary(value: str) -> str | None:
        if value == GEOMEAN_PRIMARY:
            return f'Primary metric cannot be "{GEOMEAN_PRIMARY}".'
        return None


def run_wizard(options: WizardOptions) -> WizardResult:
    """Settle every scaffold answer from flags, prompts, or defaults.

    Non-interactive mode (``yes`` or a non-TTY stdin) asks nothing; a missing
    required answer raises a :class:`GymratError` naming the flag. End-of-stream
    during any interactive prompt raises ``GymratError("Cancelled.")``.
    """
    return _Wizard(options).run()
