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


def _skill_resource() -> Traversable:
    """Resolve the packaged skill file as an :mod:`importlib.resources` traversable."""
    resource = resources.files(_PACKAGE)
    for part in _SKILL_RELATIVE_PATH.split("/"):
        resource = resource / part
    return resource


def read_bundled_skill() -> str:
    """Return the text of the skill file shipped as package data.

    Raises:
        GymratError: When the packaged skill file cannot be read — a sign the
            installation is incomplete. The message names the file and its
            resolved location, and the hint points at reinstalling.
    """
    resource = _skill_resource()
    try:
        return resource.read_text(encoding="utf-8")
    except (OSError, ValueError, BadZipFile) as err:
        message = (
            f"Could not read the bundled skill file {_SKILL_RELATIVE_PATH} "
            f"(resolved to {resource})."
        )
        hint = "The gymrat installation looks incomplete; reinstall the package to restore it."
        raise GymratError(message, hint=hint) from err
