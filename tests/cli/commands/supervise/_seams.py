"""The seam kit the ``gymrat supervise`` command tests share.

``install_seams`` replaces every seam ``commands.supervise`` composes over —
the doctor gate, config resolution, pre-flight, kickoff, the Claude driver, the
supervisor run, the run-end exit sequence, the progress reporter, and the signal
cleanup — at the names the command imports them under, and returns the
recorders a test asserts on. The git-exclude write stays real.
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest
from typer.testing import Result

from gymrat.cli.app import app
from gymrat.cli.supervise.preflight import PreflightFlags, doctor_gate, run_preflight
from gymrat.cli.supervise.progress import SuperviseReporter, create_supervise_reporter
from gymrat.config import ResolvedConfig, StopConfig, SuperviseConfig
from gymrat.loop.start import StartResult
from gymrat.session.store import ReadSessionResult
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.events import SessionObserver
from gymrat.supervisor.exit_sequence import ExitReport, ExitStep, run_exit_sequence
from gymrat.supervisor.kickoff import compose_kickoff
from gymrat.supervisor.supervise import SupervisionResult, supervise
from tests._config import resolved_config
from tests.cli._command_stubs import (
    stub_config,
)
from tests.cli._runner import (
    runner,
)
from tests.cli.supervise._fixtures import make_supervision_result
from tests.session.records._fixtures import empty_session_state, session_record, worktrees_at

#: The wall-clock cap the ``--max-minutes``-driven tests assert against.
CAP_MINUTES = 10
CAP_MS = CAP_MINUTES * 60_000


# ---------------------------------------------------------------------------
# seam installation
# ---------------------------------------------------------------------------


def _uninstall_cleanup() -> None:
    """Stand for the zero-argument handle ``install_termination_cleanup`` returns."""


@dataclass(slots=True)
class Seams:
    """The recorders and doubles a single command run wires up.

    ``supervise_calls`` / ``reporter_calls`` / ``exit_calls`` capture the keyword
    payloads their seams received; ``compose_calls`` records the prompt of each
    call. The ``create_driver`` mock stands in for the side-effecting driver
    factory so a test can read what it received; ``install_cleanup`` keeps the run
    from installing a real signal handler. The reporter's observer and warn sinks
    append to ``observed_events`` and ``warnings``. The fake supervisor run calls ``supervise_hook`` (when set) with
    its recorded call before returning, and the fake exit sequence returns
    ``exit_report`` after calling ``exit_hook`` (when set) with its recorded call,
    so a test can act from inside either.
    ``session_result`` is what the reporter's session reader returns right now:
    the reporter shows it at construction and again only after
    ``refresh_session``, so a change made later reaches the summary only
    through a refresh.
    """

    observed_events: list[object] = field(default_factory=list)
    observer: SessionObserver = field(init=False)
    warnings: list[str] = field(default_factory=list)
    exit_report: ExitReport = field(
        default_factory=lambda: ExitReport(
            steps=(ExitStep(kind="nothing", text="nothing to settle"),)
        )
    )
    supervise_hook: Callable[[dict[str, Any]], None] | None = None
    exit_hook: Callable[[dict[str, Any]], None] | None = None
    exit_calls: list[dict[str, Any]] = field(default_factory=list)
    session_result: ReadSessionResult | None = None
    final_text: str | None = None
    create_driver: Mock = field(init=False)
    install_cleanup: Mock = field(
        default_factory=lambda: create_autospec(
            install_termination_cleanup,
            name="install_termination_cleanup",
            return_value=create_autospec(_uninstall_cleanup),
        )
    )
    doctor_gate: Mock = field(
        default_factory=lambda: create_autospec(doctor_gate, name="doctor_gate")
    )
    preflight_calls: list[dict[str, object]] = field(default_factory=list)
    supervise_calls: list[dict[str, object]] = field(default_factory=list)
    reporter_calls: list[dict[str, object]] = field(default_factory=list)
    compose_calls: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.observer = self.observed_events.append
        self.create_driver = create_autospec(
            create_claude_driver, name="create_claude_driver", return_value=object()
        )

    def _record_supervise_call(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """Record one supervisor-run call, folding its positional driver and prompt into keywords.

        Args:
            args: The positional arguments the supervisor run received.
            kwargs: The keyword arguments the supervisor run received.
        """
        call = {**kwargs, **dict(zip(("driver", "prompt"), args, strict=False))}
        self.supervise_calls.append(call)


def command_config(
    *, stop: StopConfig | None = None, supervise: SuperviseConfig | None = None
) -> ResolvedConfig:
    """A resolved config the pre-flight returns and the kickoff/reporter read fields off of."""
    return resolved_config(runbook="runbook.md", stop=stop, supervise=supervise)


def make_start_result(
    root: str = "/repo", branch: str | None = None, *, resumed: bool = False
) -> StartResult:
    """Build a ``StartResult`` carrying sensible defaults, its session on ``branch`` when given."""
    overrides = {} if branch is None else {"branch": branch}
    rec = session_record(
        worktrees=worktrees_at(root),
        **overrides,
    )
    return StartResult(
        session=rec,
        state=empty_session_state(),
        resumed=resumed,
    )


def _patch_supervise(
    monkeypatch: pytest.MonkeyPatch,
    fake: Callable[..., Coroutine[object, object, SupervisionResult]],
) -> None:
    """Replace the command's supervisor run with ``fake``, called against the real signature.

    The replacement is an autospec of :func:`gymrat.supervisor.supervise.supervise`
    whose side effect is ``fake``, so a call the real function would reject fails
    the test instead of reaching ``fake``.

    Args:
        monkeypatch: The fixture the replacement is installed through.
        fake: The coroutine function that stands in for the supervisor run.
    """
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.supervise",
        create_autospec(supervise, side_effect=fake),
    )


def _install_fake_preflight(
    monkeypatch: pytest.MonkeyPatch, seams: Seams, *, branch: str | None, resumed: bool
) -> None:
    def fake_preflight(*, root: str, config: object, flags: PreflightFlags) -> StartResult:
        seams.preflight_calls.append({
            "root": root,
            "baseline_ref": flags.baseline_ref,
            "max_minutes": flags.max_minutes,
            "force": flags.force,
            "allow_dirty": flags.allow_dirty,
        })
        return make_start_result(root, branch, resumed=resumed)

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.run_preflight",
        create_autospec(run_preflight, side_effect=fake_preflight),
    )


def _install_fake_exit_sequence(monkeypatch: pytest.MonkeyPatch, seams: Seams) -> None:
    """Replace the exit sequence with a fake that records each call and returns ``exit_report``."""

    async def fake_run_exit_sequence(context: object, **kwargs: object) -> ExitReport:
        call = {"context": context, **kwargs}
        seams.exit_calls.append(call)
        if seams.exit_hook is not None:
            seams.exit_hook(call)
        return seams.exit_report

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.run_exit_sequence",
        create_autospec(run_exit_sequence, side_effect=fake_run_exit_sequence),
    )


def _install_fake_reporter(monkeypatch: pytest.MonkeyPatch, seams: Seams) -> None:
    """Replace the dashboard reporter with one whose sinks feed ``seams``' recorders."""

    def fake_reporter(**kwargs: object) -> SuperviseReporter:
        seams.reporter_calls.append(kwargs)
        shown = SimpleNamespace(session=seams.session_result)

        def refresh_session() -> None:
            shown.session = seams.session_result

        return SuperviseReporter(
            observer=seams.observer,
            stop=lambda: None,
            frame=lambda: "",
            exit_phase=lambda _phase: None,
            warn=seams.warnings.append,
            refresh_session=refresh_session,
            session_result=lambda: shown.session,
            final_text=lambda: seams.final_text,
        )

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.create_supervise_reporter",
        create_autospec(create_supervise_reporter, side_effect=fake_reporter),
    )


def install_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: ResolvedConfig | None = None,
    result: SupervisionResult | None = None,
    session_result: ReadSessionResult | None = None,
    final_text: str | None = None,
    raises: Exception | None = None,
    branch: str | None = None,
    resumed: bool = False,
    real_reporter: bool = False,
) -> Seams:
    """Replace every seam ``commands.supervise`` composes over, returning the recorders.

    Args:
        monkeypatch: The fixture the replacements are installed through.
        config: The config the resolve seam returns; a default command config when omitted.
        result: The supervision result the supervisor seam hands back.
        session_result: What the reporter's session reader returns at first.
        final_text: The agent's final message the reporter reports.
        raises: An error the supervisor seam raises instead of returning ``result``.
        branch: The session branch the fake pre-flight's start result carries.
        resumed: Whether the fake pre-flight reports a resumed session.
        real_reporter: Keep the real dashboard reporter instead of the recording
            fake. ``reporter_calls``, ``observed_events`` and ``warnings`` stay
            empty.

    Returns:
        The recorders and doubles the run is wired to.
    """
    seams = Seams()
    seams.session_result = session_result
    seams.final_text = final_text
    resolved = config if config is not None else command_config()
    handed_back = result if result is not None else make_supervision_result()

    def fake_compose(
        cfg: object,
        prompt: object = None,
        *,
        experiment_worktree: object,
    ) -> SimpleNamespace:
        seams.compose_calls.append(prompt)
        return SimpleNamespace(kickoff="begin optimization", system_prompt_append="system prompt")

    async def fake_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams._record_supervise_call(args, kwargs)
        if seams.supervise_hook is not None:
            seams.supervise_hook(seams.supervise_calls[-1])
        if raises is not None:
            raise raises
        return handed_back

    monkeypatch.setattr("gymrat.cli.commands.supervise.doctor_gate", seams.doctor_gate)
    stub_config(monkeypatch, "supervise", resolved)
    _install_fake_preflight(monkeypatch, seams, branch=branch, resumed=resumed)
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.compose_kickoff",
        create_autospec(compose_kickoff, side_effect=fake_compose),
    )
    monkeypatch.setattr("gymrat.cli.commands.supervise.create_claude_driver", seams.create_driver)
    _patch_supervise(monkeypatch, fake_supervise)
    _install_fake_exit_sequence(monkeypatch, seams)
    if not real_reporter:
        _install_fake_reporter(monkeypatch, seams)
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.install_termination_cleanup", seams.install_cleanup
    )
    monkeypatch.setattr("gymrat.session.budget.install_termination_cleanup", seams.install_cleanup)
    return seams


def run(*extra: str, prompt: str = "optimize it", max_minutes: str = str(CAP_MINUTES)) -> Result:
    """Invoke ``gymrat supervise`` through the shared CLI runner.

    Args:
        *extra: Further flags, appended after the required cap.
        prompt: The kickoff prompt argument.
        max_minutes: The required ``--max-minutes`` cap.

    Returns:
        The runner's result.
    """
    return runner.invoke(app, ["supervise", prompt, "--max-minutes", max_minutes, *extra])
