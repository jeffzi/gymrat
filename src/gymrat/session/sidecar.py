"""Reading the JSON sidecar files a session keeps beside its log."""

from pathlib import Path

from pydantic import BaseModel, ValidationError


def read_sidecar[Model: BaseModel](path: Path, model: type[Model]) -> Model | None:
    """Read and validate one sidecar file.

    A sidecar is advisory state another process may be rewriting or may have
    left behind, so a file that cannot be used reads as no file at all.

    Args:
        path: The sidecar file to read.
        model: The pydantic model the file's JSON must validate against.

    Returns:
        The validated model, or ``None`` when the file is missing, unreadable,
        not UTF-8, or does not validate.
    """
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError, UnicodeDecodeError):
        return None
