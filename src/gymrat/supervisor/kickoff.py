"""Compose the system-prompt append and kickoff message for a supervised session.

Unlike the shell command, a supervised session has no human to answer the
skill's interactive fallback, so it reads the bundled skill and requires a
runbook. ``compose_kickoff`` stitches the skill text and the runbook body into
the system-prompt append and settles the opening kickoff message.
"""

from dataclasses import dataclass
from pathlib import Path

from gymrat.bundled_skill import read_bundled_skill, strip_frontmatter
from gymrat.config import BenchlessConfig
from gymrat.errors import GymratError

DEFAULT_KICKOFF = (
    "Drive the optimization session. This session has a wall-clock cap; "
    "`iterate`, `keep`, `discard`, `status`, `sync`, `compare`, and `measure` print the time left. "
    "Follow the skill instructions and the runbook to guide your work."
)


_CLOCK_RULE = (
    "This session has a wall-clock cap. The `keep`, `discard`, `status`, `sync`, `compare`, "
    "and `measure` Bash commands print a time-left line; the `iterate` and `probe` tools "
    "return the same clock as `budget.remaining_seconds` in their JSON document. Read the "
    "time left after every one of them and plan from it. Never estimate elapsed time "
    "yourself. A measurement the cap kills records nothing."
)
"""How the agent reads the session's remaining time, in both the command and tool forms."""

_CONTRACT = (
    "No human reads the turns of this session. Ask-first rules resolve to deciding "
    "from the runbook — the runbook is the authority. When the work is done, run "
    '`gymrat stop -m "<report>"` and only then end the turn. The supervisor replies '
    "after every turn and the session continues, so ending a turn never waits for "
    "anything. Never run a gymrat command in the background."
)
"""What an unattended session may decide on its own, and how it signals it is done."""

_TOOLS = (
    "Use the `probe` and `iterate` tools instead of running `gymrat probe` or "
    "`gymrat iterate` through Bash. The `measure`, `compare`, `keep`, `discard`, "
    "`status`, and `stop` commands stay Bash commands. A tool call runs in the "
    "foreground and returns the command's JSON document."
)
"""Which actions reach gymrat as in-process tools and which stay Bash commands."""


def _read_runbook(path: str) -> str:
    """Read the runbook at ``path``, mapping every read failure to a ``GymratError``."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        message = f"Runbook is not valid UTF-8: {path}"
        hint = "Re-save the runbook as UTF-8 or remove non-UTF-8 bytes."
        raise GymratError(message, hint=hint) from err
    except FileNotFoundError as err:
        message = f"Runbook not found at {path}."
        hint = "Verify the file exists at the path configured for `runbook` in gymrat.toml."
        raise GymratError(message, hint=hint) from err
    except OSError as err:
        message = f"Cannot read runbook at {path}."
        hint = "Check file permissions and ensure the path is a regular file."
        raise GymratError(message, hint=hint) from err


@dataclass(frozen=True, slots=True)
class KickoffResult:
    """The prompt additions a supervised session starts from."""

    system_prompt_append: str
    kickoff: str


def compose_kickoff(
    config: BenchlessConfig,
    prompt: str | None = None,
    *,
    experiment_worktree: str,
) -> KickoffResult:
    """Build the supervised session's system-prompt append and kickoff message.

    The bundled skill is read first: a broken installation is surfaced before
    any runbook validation. Its frontmatter is activation metadata, not
    instructions, so only the body reaches the prompt. The runbook is then
    required and read, and its body is appended under a heading naming its
    path.

    Args:
        config: The settled benchless configuration; its ``runbook`` must name a
            readable file.
        prompt: An explicit kickoff message; when omitted, a default is used.
        experiment_worktree: Absolute path to the experiment worktree directory.

    Returns:
        The system-prompt append and kickoff message for the agent session.

    Raises:
        GymratError: When the bundled skill cannot be read, when no runbook is
            configured, or when the configured runbook file cannot be read.
    """
    skill_content = strip_frontmatter(read_bundled_skill())

    if config.runbook is None:
        message = "No runbook configured — set `runbook` in gymrat.toml."
        hint = (
            "A supervised session has no human to answer the skill's "
            "fallback; a runbook is required."
        )
        raise GymratError(message, hint=hint)

    runbook_content = _read_runbook(config.runbook)

    system_prompt_append = (
        f"{skill_content}\n\n"
        "**The gymrat skill is already loaded above. Do not call `Skill(gymrat)`.**\n\n"
        f"{_CONTRACT}\n\n"
        f"{_TOOLS}\n\n"
        f"{_CLOCK_RULE}\n\n"
        f"## Runbook: {config.runbook}\n\n{runbook_content}"
    )

    preflight_done = (
        f"The session is open and the baseline is recorded. "
        f"The experiment worktree is at {experiment_worktree}. "
        f"Skip steps 1 and 2 of the skill and begin with the runbook."
    )
    base_message = prompt if prompt is not None else DEFAULT_KICKOFF
    kickoff = f"{base_message}\n\n{preflight_done}"

    return KickoffResult(
        system_prompt_append=system_prompt_append,
        kickoff=kickoff,
    )
