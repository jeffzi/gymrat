"""Failure formatting for command errors during sampling."""

from gymrat.errors import CommandError
from gymrat.exec import ExecResult, ExecTimeoutError
from gymrat.sampling.types import _LABEL_WIDTH, _REF_HINT, TargetContext
from gymrat.targets import RefTarget


def to_command_error(  # noqa: PLR0913, PLR0917 -- one field per failure axis
    phase: str,
    sample_index: int | None,
    command: str,
    ctx: TargetContext,
    result: ExecResult | ExecTimeoutError,
    request_timeout_ms: int,
) -> CommandError:
    """Map a command failure to a target-specific :class:`CommandError`.

    A ref target contributes ``ref`` and ``worktree`` location lines plus the
    hint that the worktree only holds tracked files; a plain directory
    contributes a single ``dir`` line and no hint.

    Args:
        phase: Name of the phase the command ran in, e.g. ``"prepare"``.
        sample_index: The sample number to mention in the header, or ``None``
            when the failure is not tied to a specific sample.
        command: The command line that failed.
        ctx: The target context supplying the header label, position, and
            location lines.
        result: The execution outcome, either a failed ``ExecResult`` or a
            timeout.
        request_timeout_ms: The configured timeout, used when ``result`` did
            not time out.

    Returns:
        The formatted command error with target-specific location context.
    """
    timed_out = isinstance(result, ExecTimeoutError)
    if timed_out:
        timeout_ms = result.timeout_ms
        exit_code: int | None = None
    else:
        timeout_ms = request_timeout_ms
        exit_code = result.exit_code

    target = ctx.target
    if isinstance(target, RefTarget):
        location = [_field("ref", target.ref), _field("worktree", ctx.dir)]
        hint = _REF_HINT
    else:
        location = [_field("dir", ctx.dir)]
        hint = None

    position = f"{ctx.position}, " if ctx.position is not None else ""
    sample = f", sample {sample_index}" if sample_index is not None else ""
    outcome_label = "timed out" if timed_out else "failed"
    header = f'{phase} command {outcome_label} ({position}"{ctx.label}"{sample})'

    lines = [header, *location, _field("command", command)]
    if timed_out:
        lines.append(_field("timeout", f"{timeout_ms}ms"))
    else:
        lines.append(_field("exit code", exit_code))
    lines.extend(
        _captured_output(result.stdout, result.stdout_bytes, result.stderr, result.stderr_bytes)
    )

    return CommandError("\n".join(lines), hint=hint)


def _field(label: str, value: object) -> str:
    """Format an indented, column-aligned ``label: value`` detail line."""
    return f"  {(label + ':').ljust(_LABEL_WIDTH)}{value}"


def _captured_output(stdout: str, stdout_bytes: int, stderr: str, stderr_bytes: int) -> list[str]:
    """Render the captured output of a failed command.

    A lone non-empty stream is emitted bare unless its captured text was
    truncated, in which case it — like every stream when both are present —
    becomes a labeled entry annotated with the true byte total.

    Args:
        stdout: The captured stdout text.
        stdout_bytes: The true byte total of stdout before truncation.
        stderr: The captured stderr text.
        stderr_bytes: The true byte total of stderr before truncation.

    Returns:
        Lines of rendered output, ready for joining into the error message.
    """
    streams = [
        ("stderr", stderr, stderr_bytes),
        ("stdout", stdout, stdout_bytes),
    ]
    present = [(label, text, total) for label, text, total in streams if text]

    if len(present) == 1:
        label, text, total = present[0]
        if not _is_truncated(text, total):
            return [text]
        return _labeled(label, text, total)

    return [line for entry in present for line in _labeled(*entry)]


def _is_truncated(text: str, total_bytes: int) -> bool:
    """Whether ``total_bytes`` exceeds what survived capture in ``text``."""
    return total_bytes > len(text.encode())


def _labeled(label: str, text: str, total: int) -> list[str]:
    """Build a labeled stream entry, flagging truncation against the byte total."""
    suffix = f" (truncated, {total} bytes total)" if _is_truncated(text, total) else ""
    return [f"--- {label}{suffix} ---", text]
