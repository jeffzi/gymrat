"""Behavioral tests for reading and removing a JSON sidecar file.

``read_sidecar`` validates a file against a pydantic model and reads any file it
cannot use as no file at all: missing, unreadable, not UTF-8, or does not
validate. Which keys and value types a model accepts belongs to the model, so
those cases live beside each sidecar's own tests.

``remove_sidecar`` removes a file best-effort: a missing file is a silent
success, and a removal the OS refuses is warned about, never raised. Each
sidecar's own tests pin only the path and label it hands over.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from gymrat.utils import read_sidecar, remove_sidecar
from tests.session._progress_file import fail_unlink_of


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


# ---------------------------------------------------------------------------
# read_sidecar
# ---------------------------------------------------------------------------


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
        pytest.param(_contents(b"not valid json{{{"), id="invalid-json"),
        pytest.param(_contents(b"\x80\x81\x82"), id="not-utf8"),
    ],
)
def test_read_sidecar_when_file_unusable_does_return_none(
    tmp_path: Path, arrange: Callable[[Path], None]
):
    path = tmp_path / "sidecar.json"
    arrange(path)

    result = read_sidecar(path, _Sample)

    assert result is None


# ---------------------------------------------------------------------------
# remove_sidecar
# ---------------------------------------------------------------------------


def test_remove_sidecar_when_file_exists_does_remove_it(tmp_path: Path):
    path = tmp_path / "sidecar.json"
    path.write_text("{}", encoding="utf-8")
    warnings: list[str] = []

    remove_sidecar(path, "sample sidecar", warnings.append)

    assert (path.exists(), warnings) == (False, [])


def test_remove_sidecar_when_file_absent_does_not_warn(tmp_path: Path):
    warnings: list[str] = []

    remove_sidecar(tmp_path / "sidecar.json", "sample sidecar", warnings.append)

    assert warnings == []


def test_remove_sidecar_when_os_refuses_removal_does_warn_once_naming_label_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "sidecar.json"
    path.write_text("{}", encoding="utf-8")
    fail_unlink_of(str(path), monkeypatch)
    warnings: list[str] = []

    remove_sidecar(path, "sample sidecar", warnings.append)

    assert len(warnings) == 1
    assert warnings[0].startswith(f"warning: could not remove the sample sidecar {path}: ")
    assert "The process cannot access the file" in warnings[0]
