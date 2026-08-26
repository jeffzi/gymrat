"""Hook-script builders and record helpers shared by the hooks tests.

Every builder writes a numbered Python script into a scratch directory and
returns the command that runs it under the current interpreter: no shell
builtin, no executable bit, just ``"<python>" "<script>"`` with both paths
quoted. Large payloads are parked in files beside the script rather than baked
into the source, so a channel far larger than a source literal wants to be
still reaches the runner byte for byte.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.loop._hooks``.
"""

import json
import sys
from pathlib import Path
from typing import Any, Literal

from gymrat_py.loop.hooks import HookInvocation
from gymrat_py.session import HookRecord, Worktrees
from gymrat_py.session.schema import HookStage
from tests.session._records import SESSION_ID, iteration_record, session_record

Channel = Literal["stdout", "stderr"]


class HookScripts:
    """Builders for hook commands scoped to a scratch directory.

    A private counter numbers the scripts so repeated builders in one test never
    collide on a filename. ``experiment_dir`` is the worktree an invocation runs
    in; ``temp_dir`` holds the scripts and any parked payload files.
    """

    def __init__(self, temp_dir: str, experiment_dir: str) -> None:
        self.temp_dir = temp_dir
        self.experiment_dir = experiment_dir
        self._count = 0

    def hook_command(self, body: str) -> str:
        """Write ``body`` as a script and give back the command that runs it."""
        self._count += 1
        script_path = Path(self.temp_dir) / f"hook-{self._count}.py"
        script_path.write_text(body, encoding="utf-8")
        return f'"{sys.executable}" "{script_path}"'

    def printing(self, *lines: str) -> str:
        r"""A command printing each of ``lines`` on its own line on stdout.

        Writes go through ``sys.stdout.buffer`` so ``\n`` stays a single byte
        on Windows — text-mode stdout would translate it to ``\r\n``.
        """
        writes = "\n".join(
            f"sys.stdout.buffer.write({json.dumps(line + chr(10))}.encode())" for line in lines
        )
        return self.hook_command(f"import sys\n{writes}\n")

    def printing_line(self, name: str, channel: Channel, content: str) -> str:
        """Park ``content`` in ``name`` beside the hook and return source printing it.

        The text lives in a file rather than inside the script so a payload far
        larger than a source literal wants to be still reaches the runner byte
        for byte.
        """
        data_path = Path(self.temp_dir) / name
        data_path.write_bytes(content.encode())
        return f"sys.{channel}.buffer.write(open({json.dumps(str(data_path))}, 'rb').read())\n"

    def printing_content_of(self, file_name: str, content: str) -> str:
        """A command that prints parked ``content`` verbatim on stdout."""
        return self.hook_command("import sys\n" + self.printing_line(file_name, "stdout", content))

    def failing_content_of(self, stem: str, stdout: str, stderr: str) -> str:
        """A command that prints parked text on both channels and then exits 3."""
        body = (
            "import sys\n"
            + self.printing_line(f"{stem}-stdout.txt", "stdout", stdout)
            + self.printing_line(f"{stem}-stderr.txt", "stderr", stderr)
            + "sys.exit(3)\n"
        )
        return self.hook_command(body)

    def invocation_of(self, command: str, **overrides: Any) -> HookInvocation:
        """A ``before`` invocation on the scratch worktree, overridable field by field."""
        session = session_record(
            session_id=SESSION_ID,
            worktrees=Worktrees(
                experiment=self.experiment_dir,
                baseline=str(Path(self.temp_dir) / "side-baseline"),
            ),
        )
        fields: dict[str, Any] = {
            "command": command,
            "stage": "before",
            "seq": 2,
            "session": session,
            "last_iteration": None,
            "iteration_count": 1,
        }
        fields.update(overrides)
        return HookInvocation(**fields)


def labeled_lines(report: str, stage: HookStage) -> list[str]:
    """The report's lines with their ``[stage]`` label stripped off.

    Every line the runner emits carries the label, so a line without one is a
    leak of unlabeled hook output rather than something to quietly pass through.
    """
    if report == "":
        return []
    prefix = f"[{stage}] "
    lines: list[str] = []
    for line in report.split("\n"):
        if not line.startswith(prefix):
            message = f"expected every hook report line to be labeled {prefix.strip()}: {line}"
            raise AssertionError(message)
        lines.append(line[len(prefix) :])
    return lines


def expected_hook_record(**overrides: Any) -> HookRecord:
    """The ``HookRecord`` ``run_hook`` produces, with ``duration_ms`` normalized to 0.

    ``duration_ms`` is nondeterministic, so it defaults to 0 here and callers
    compare against a record whose own ``duration_ms`` they have replaced with 0.
    ``timed_out`` defaults to ``False`` and ``stderr_bytes`` to 0; every other
    field is the caller's to name.
    """
    fields: dict[str, Any] = {
        "type": "hook",
        "timed_out": False,
        "stderr_bytes": 0,
        "duration_ms": 0,
    }
    fields.update(overrides)
    return HookRecord(**fields)


__all__ = [
    "HookScripts",
    "expected_hook_record",
    "iteration_record",
    "labeled_lines",
    "session_record",
]
