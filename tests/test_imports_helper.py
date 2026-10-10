"""Tests for the import-isolation probe the seam tests run."""

import subprocess

import pytest

from tests._imports import modules_loaded_after


def test_modules_loaded_after_when_statements_raise_does_report_child_stderr():
    with pytest.raises(
        subprocess.CalledProcessError,
        match=r"ModuleNotFoundError: No module named 'banana'",
    ):
        modules_loaded_after("import banana")
