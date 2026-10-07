"""The seam kit the ``gymrat supervise`` command tests share.

``install_seams`` replaces every seam ``commands.supervise`` composes over —
config resolution, pre-flight, kickoff, the Claude driver, the supervisor run,
the run-end exit sequence, the progress reporter, the git-exclude write, and
the signal cleanup — at the names the command imports them under, and returns
the recorders a test asserts on.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest
from typer.testing import Result

from gymrat.cli.app import app
from gymrat.cli.exit import write_stdout
from gymrat.cli.supervise.preflight import PreflightFlags, doctor_gate
from gymrat.cli.supervise.progress import SuperviseReporter, create_supervise_reporter
from gymrat.cli.supervise.types import ReadSessionResult
from gymrat.config import ResolvedConfig, StopConfig, SuperviseConfig
from gymrat.errors import GymratError
from gymrat.loop.start import StartResult
from gymrat.session.workspace import ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.exit_sequence import ExitPhase, ExitReport, ExitStep
from gymrat.supervisor.supervise import SupervisionResult
from tests._ansi import strip_ansi
from tests._config import resolved_config
from tests._process_helpers import CleanupRegistry
from tests.cli._session import runner
from tests.cli.supervise._fixtures import make_supervision_result
from tests.session.records._fixtures import empty_session_state, session_record, worktrees_at

#: The wall-clock cap the ``--max-minutes``-driven tests assert against.
CAP_MINUTES = 10
CAP_MS = CAP_MINUTES * 60_000

#: The message the tracing-setup seam fails with when a test makes it explode.
TRACING_FAILURE = "tracing exporter unreachable"


# ---------------------------------------------------------------------------
# seam installation
# ---------------------------------------------------------------------------


def _real_reporter() -> SuperviseReporter:
    """The production reporter, built only as an autospec source for ``stop``.

    ``SuperviseReporter.stop`` is a nested closure with no importable name, so it
    can't be targeted directly by ``create_autospec``. Building a real
    (side-effect-free, plain-mode) reporter and taking its attribute gives
    ``create_autospec`` the actual production callable to bind against.
    """
    return create_supervise_reporter(root="/tmp/repo", max_minutes=1.0, mode="plain")


class Seams:
    """The recorders and doubles a single command run wires up.

    ``supervise_calls`` / ``reporter_calls`` / ``exit_calls`` capture the keyword
    payloads their seams received; ``compose_calls`` records ``(config, prompt)``
    per call. The ``reporter_stop``, ``ensure_git_exclude``, ``install_cleanup``,
    and ``create_driver`` mocks stand in for the side-effecting seams so a test can
    assert they fired. The reporter's observer, exit-phase, and warn sinks append
    to ``observed_events``, ``exit_phases``, and ``warnings``. The fake exit
    sequence returns ``exit_report`` after calling ``exit_hook`` (when set) with
    its recorded call, so a test can act from inside the sequence.
    ``session_result`` is what the reporter's session reader returns right now:
    the reporter shows it at construction and again only after
    ``refresh_session``, so a change made later reaches the summary only
    through a refresh.
    """

    def __init__(self) -> None:
        self.driver = object()
        self.observed_events: list[object] = []
        self.observer: Callable[[object], None] = self.observed_events.append
        self.exit_phases: list[ExitPhase] = []
        self.warnings: list[str] = []
        self.exit_report = ExitReport(steps=(ExitStep(kind="nothing", text="nothing to settle"),))
        self.exit_hook: Callable[[dict[str, Any]], None] | None = None
        self.exit_calls: list[dict[str, Any]] = []
        self.session_result: ReadSessionResult | None = None
        self.final_text: str | None = None
        real_reporter = _real_reporter()
        self.reporter_stop = create_autospec(real_reporter.stop, name="reporter.stop")
        self.ensure_git_exclude = create_autospec(ensure_git_exclude, name="ensure_git_exclude")
        self.create_driver = create_autospec(
            create_claude_driver, name="create_claude_driver", return_value=self.driver
        )
        self.install_cleanup = create_autospec(
            install_termination_cleanup, name="install_termination_cleanup", return_value=Mock()
        )
        self.doctor_gate = create_autospec(doctor_gate, name="doctor_gate")
        self.preflight_calls: list[dict[str, object]] = []
        self.supervise_calls: list[dict[str, object]] = []
        self.reporter_calls: list[dict[str, object]] = []
        self.compose_calls: list[tuple[object, object]] = []

    def record_supervise_call(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        call = {**kwargs, **dict(zip(("driver", "prompt"), args, strict=False))}
        self.supervise_calls.append(call)

    def installed_cleanups(self) -> list[Callable[[], None]]:
        """Return every termination cleanup the run installed, in install order."""
        return [call.args[0] for call in self.install_cleanup.call_args_list]


def command_config(
    *,
    stop: StopConfig | None = None,
    runbook: str | None = "runbook.md",
    supervise: SuperviseConfig | None = None,
) -> ResolvedConfig:
    """A resolved config the pre-flight returns and the kickoff/reporter read fields off of."""
    return resolved_config(runbook=runbook, stop=stop, supervise=supervise)


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
) -> Seams:
    """Replace every seam ``commands.supervise`` composes over, returning the recorders."""
    seams = Seams()
    seams.session_result = session_result
    seams.final_text = final_text
    resolved = config if config is not None else command_config()
    handed_back = result if result is not None else make_supervision_result()

    def fake_preflight(*, root: str, config: object, flags: PreflightFlags) -> StartResult:
        seams.preflight_calls.append({
            "root": root,
            "config": config,
            "baseline_ref": flags.baseline_ref,
            "max_minutes": flags.max_minutes,
            "force": flags.force,
            "allow_dirty": flags.allow_dirty,
        })
        return make_start_result(root, branch, resumed=resumed)

    def fake_compose(
        cfg: object,
        prompt: object = None,
        *,
        experiment_worktree: object = None,
    ) -> SimpleNamespace:
        seams.compose_calls.append((cfg, prompt))
        return SimpleNamespace(kickoff="begin optimization", system_prompt_append="system prompt")

    async def fake_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        if raises is not None:
            raise raises
        return handed_back

    async def fake_run_exit_sequence(context: object, **kwargs: object) -> ExitReport:
        call = {"context": context, **kwargs}
        seams.exit_calls.append(call)
        if seams.exit_hook is not None:
            seams.exit_hook(call)
        return seams.exit_report

    def fake_reporter(**kwargs: object) -> SimpleNamespace:
        seams.reporter_calls.append(kwargs)
        shown = SimpleNamespace(session=seams.session_result)

        def refresh_session() -> None:
            shown.session = seams.session_result

        return SimpleNamespace(
            observer=seams.observer,
            stop=seams.reporter_stop,
            exit_phase=seams.exit_phases.append,
            warn=seams.warnings.append,
            refresh_session=refresh_session,
            session_result=lambda: shown.session,
            final_text=lambda: seams.final_text,
        )

    def fake_resolve(_flags: object, _base_dir: object = None) -> ResolvedConfig:
        return resolved

    monkeypatch.setattr("gymrat.cli.commands.supervise.doctor_gate", seams.doctor_gate)
    monkeypatch.setattr("gymrat.cli.commands.supervise.resolve_config", fake_resolve)
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", fake_preflight)
    monkeypatch.setattr("gymrat.cli.commands.supervise.compose_kickoff", fake_compose)
    monkeypatch.setattr("gymrat.cli.commands.supervise.create_claude_driver", seams.create_driver)
    monkeypatch.setattr("gymrat.cli.commands.supervise.supervise", fake_supervise)
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_exit_sequence", fake_run_exit_sequence)
    monkeypatch.setattr("gymrat.cli.commands.supervise.create_supervise_reporter", fake_reporter)
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.ensure_git_exclude", seams.ensure_git_exclude
    )
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.install_termination_cleanup", seams.install_cleanup
    )
    return seams


def track_cleanups(monkeypatch: pytest.MonkeyPatch) -> CleanupRegistry:
    """Swap the command's cleanup installer for a registry a test can read at any point."""
    registry = CleanupRegistry()
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.install_termination_cleanup", registry.install
    )
    return registry


def record_stdout_writes(monkeypatch: pytest.MonkeyPatch, order: list[str], label: str) -> None:
    """Append ``label`` to ``order`` at every stdout write the command makes, then forward it."""

    def tracking_write(data: str) -> None:
        order.append(label)
        write_stdout(data)

    monkeypatch.setattr("gymrat.cli.commands.supervise.write_stdout", tracking_write)


def run(*args: str) -> Result:
    """Invoke ``gymrat supervise`` with ``args`` through the shared CLI runner."""
    return runner.invoke(app, ["supervise", *args])


def err_text(result: Result) -> str:
    """The combined stdout+stderr of a run, for flag-name and message probes."""
    return strip_ansi((result.stdout or "") + (result.stderr or ""))


def exploding_setup_tracing(*_args: object, **_kwargs: object) -> tuple[object, ...]:
    """Stand in for the tracing setup, failing the way a bad exporter does."""
    raise GymratError(TRACING_FAILURE)
