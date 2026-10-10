"""Behavioral tests for reading a JSON sidecar file.

``read_sidecar`` validates a file against a pydantic model and reads any file it
cannot use as no file at all: missing, unreadable, not UTF-8, not JSON, or not a
JSON object. Which keys and value types a model accepts belongs to the model, so
those cases live beside each sidecar's own tests.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from gymrat.session.sidecar import read_sidecar


class _Sample(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    count: int
    name: str


def _contents(raw: bytes) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        path.write_bytes(raw)

    return write


def _absent(_path: Path) -> None:
    """Leave the sidecar file absent."""


def _directory(path: Path) -> None:
    path.mkdir()


def test_read_sidecar_when_file_validates_does_return_the_model(tmp_path: Path):
    path = tmp_path / "sidecar.json"
    path.write_text('{"count":3,"name":"apple"}', encoding="utf-8")

    result = read_sidecar(path, _Sample)

    assert result == _Sample(count=3, name="apple")


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_absent, id="file-absent"),
        pytest.param(_directory, id="path-is-a-directory"),
        pytest.param(_contents(b""), id="empty"),
        pytest.param(_contents(b'{"count":3,"na'), id="truncated-json"),
        pytest.param(_contents(b"not valid json{{{"), id="invalid-json"),
        pytest.param(_contents(b"\x80\x81\x82"), id="not-utf8"),
        pytest.param(_contents(b'[3, "apple"]'), id="array-not-object"),
        pytest.param(_contents(b"42"), id="number-not-object"),
    ],
)
def test_read_sidecar_when_file_unusable_does_return_none(
    tmp_path: Path, arrange: Callable[[Path], None]
):
    path = tmp_path / "sidecar.json"
    arrange(path)

    result = read_sidecar(path, _Sample)

    assert result is None
