"""Reader for the skill file shipped as package data.

The skill file lives alongside the package as ``skills/gymrat/SKILL.md`` and is
resolved through :mod:`importlib.resources` so it is found whether gymrat runs
from a source checkout, a wheel, or a zipapp.
"""

from importlib import resources
from importlib.resources.abc import Traversable
from zipfile import BadZipFile

from gymrat.errors import GymratError

_PACKAGE = "gymrat"
_SKILL_RELATIVE_PATH = "skills/gymrat/SKILL.md"
_FRONTMATTER_DELIMITER = "---"


def strip_frontmatter(text: str) -> str:
    """Return ``text`` without its leading YAML frontmatter block.

    The frontmatter is Claude Code activation metadata — a ``---`` line, YAML
    fields whose values may fold across several lines, and a closing ``---``
    line — so consumers that feed the skill to a model as plain instructions
    drop it. Text that does not open with a delimiter line, or whose block is
    never closed, is returned unchanged.
    """
    if not text.startswith(f"{_FRONTMATTER_DELIMITER}\n"):
        return text

    _, closing, body = text.partition(f"\n{_FRONTMATTER_DELIMITER}\n")
    return body if closing else text


def _skill_resource() -> Traversable:
    """Resolve the packaged skill file as an :mod:`importlib.resources` traversable."""
    return resources.files(_PACKAGE).joinpath(_SKILL_RELATIVE_PATH)


def read_bundled_skill() -> str:
    """Return the text of the skill file shipped as package data.

    Raises:
        GymratError: When the packaged skill file cannot be resolved or read — a
            sign the installation is incomplete. The message names the file, and
            its resolved location too when resolution got that far; the hint
            points at reinstalling.
    """
    resource: Traversable | None = None
    try:
        resource = _skill_resource()
        return resource.read_text(encoding="utf-8")
    except (OSError, ValueError, BadZipFile) as err:
        location = "" if resource is None else f" (resolved to {resource})"
        message = f"Could not read the bundled skill file {_SKILL_RELATIVE_PATH}{location}."
        hint = "The gymrat installation looks incomplete; reinstall the package to restore it."
        raise GymratError(message, hint=hint) from err
