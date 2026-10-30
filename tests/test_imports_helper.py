"""Tests for the import-isolation probe the seam tests run."""

import subprocess

import pytest

from tests._imports import modules_loaded_after


def test_modules_loaded_after_when_statements_raise_does_report_child_stderr():
    with pytest.raises(subprocess.CalledProcessError) as caught:
        modules_loaded_after("import banana")

    assert caught.value.__notes__ == [caught.value.stderr]
