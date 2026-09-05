"""Path display helpers shared across CLI and supervisor output."""

from pathlib import Path


def abbreviate_home(path: str) -> str:
    """Shorten a path under the user's home directory to a ``~`` prefix.

    Returns *path* unchanged when it does not fall under the home directory.
    """
    try:
        rel = Path(path).relative_to(Path.home()).as_posix()
    except (ValueError, RuntimeError):
        return path
    return "~" if rel == "." else f"~/{rel}"
