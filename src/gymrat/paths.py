"""Path display helpers shared across CLI and supervisor output."""

from pathlib import Path


def abbreviate_home(path: str) -> str:
    """Shorten a path under the user's home directory to a ``~`` prefix.

    Args:
        path: The path to abbreviate.

    Returns:
        The ``~``-prefixed path, or *path* unchanged when it is not under home.
    """
    try:
        rel = Path(path).relative_to(Path.home()).as_posix()
    except (ValueError, RuntimeError):
        return path
    return "~" if rel == "." else f"~/{rel}"
