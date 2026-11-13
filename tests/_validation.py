"""Shared helper that captures a pydantic validation failure."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError


def validation_error(annotation: Any, value: object) -> ValidationError:
    """Validate ``value`` against ``annotation`` and return the error it raises.

    Args:
        annotation: The type to validate against.
        value: A value expected to fail validation.

    Returns:
        The raised ``ValidationError``; the calling test fails when none is raised.
    """
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError as error:
        return error
    pytest.fail(f"{value!r} unexpectedly validated against {annotation!r}")
