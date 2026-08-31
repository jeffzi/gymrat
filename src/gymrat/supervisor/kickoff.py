"""Compose the system-prompt append and kickoff message for a supervised session.

Unlike the shell command, a supervised session has no human to answer the
skill's interactive fallback, so it reads the bundled skill and requires a
runbook. ``compose_kickoff`` stitches the skill text and the runbook body into
the system-prompt append and settles the opening kickoff message.
"""

from dataclasses import dataclass
from pathlib import Path

from gymrat.bundled_skill import read_bundled_skill
from gymrat.config import BenchlessConfig
from gymrat.errors import GymratError

DEFAULT_KICKOFF = (
    "Drive the optimization session. Follow the skill instructions and the "
    "runbook to guide your work."
)


@dataclass(frozen=True, slots=True)
class KickoffResult:
    """The prompt additions a supervised session starts from."""

    system_prompt_append: str
    kickoff: str


def compose_kickoff(config: BenchlessConfig, prompt: str | None = None) -> KickoffResult:
    """Build the supervised session's system-prompt append and kickoff message.

    The bundled skill is read first: a broken installation is surfaced before
    any runbook validation. The runbook is then required and read, and its body
    is appended under a heading naming its path.

    Args:
        config: The settled benchless configuration; its ``runbook`` must name a
            readable file.
        prompt: An explicit kickoff message; when omitted, a default is used.

    Raises:
        GymratError: When the bundled skill cannot be read, when no runbook is
            configured, or when the configured runbook file cannot be read.
    """
    skill_content = read_bundled_skill()

    if config.runbook is None:
        message = "No runbook configured — set `runbook` in gymrat.toml."
        hint = (
            "A supervised session has no human to answer the skill's "
            "fallback; a runbook is required."
        )
        raise GymratError(message, hint=hint)

    try:
        runbook_content = Path(config.runbook).read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        message = f"Runbook is not valid UTF-8: {config.runbook}"
        hint = "Re-save the runbook as UTF-8 or remove non-UTF-8 bytes."
        raise GymratError(message, hint=hint) from err
    except FileNotFoundError as err:
        message = f"Runbook not found at {config.runbook}."
        hint = "Verify the file exists at the path configured for `runbook` in gymrat.toml."
        raise GymratError(message, hint=hint) from err
    except OSError as err:
        message = f"Cannot read runbook at {config.runbook}."
        hint = "Check file permissions and ensure the path is a regular file."
        raise GymratError(message, hint=hint) from err

    system_prompt_append = f"{skill_content}\n## Runbook: {config.runbook}\n\n{runbook_content}"

    return KickoffResult(
        system_prompt_append=system_prompt_append,
        kickoff=prompt if prompt is not None else DEFAULT_KICKOFF,
    )
