"""Tests for the config subsystem simplification.

Verifies that:
- The three-dataclass read-outcome union is replaced by tuple returns in load.py
- Throwing duplicates are removed from validate.py
- The config package loads without circular imports
- resolve_benchless_config delegates to inspect_config
"""

import subprocess
import sys
from pathlib import Path

import pytest
import tomli_w

from gymrat.config import (
    BenchlessConfig,
    CliFlags,
    HooksConfig,
    StopConfig,
    resolve_benchless_config,
)
from gymrat.errors import GymratError


def write_config(directory: Path, content: dict[str, object]) -> Path:
    config_path = directory / "gymrat.toml"
    config_path.write_text(tomli_w.dumps(content), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# _ReadOk / _ReadAbsent / _ReadError removed from load.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("_ReadOk", id="read-ok"),
        pytest.param("_ReadAbsent", id="read-absent"),
        pytest.param("_ReadError", id="read-error"),
    ],
)
def test_load_module_when_read_outcome_class_accessed_does_raise_attribute_error(name: str):
    from gymrat.config import load

    assert not hasattr(load, name)


# ---------------------------------------------------------------------------
# assert_flag_not_empty / validate_loop_keys / assert_runbook_exists removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("assert_flag_not_empty", id="assert-flag-not-empty"),
        pytest.param("validate_loop_keys", id="validate-loop-keys"),
        pytest.param("assert_runbook_exists", id="assert-runbook-exists"),
    ],
)
def test_validate_module_when_removed_function_accessed_does_raise_attribute_error(name: str):
    from gymrat.config import validate

    assert not hasattr(validate, name)


# ---------------------------------------------------------------------------
# No circular import at package load time
# ---------------------------------------------------------------------------


def test_config_package_when_imported_fresh_does_not_raise_import_error():
    """Import gymrat.config in a subprocess to detect circular imports.

    A subprocess is necessary because the test process has already imported the
    package, which masks cycles that only manifest on first import.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import gymrat.config"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, (
        f"Importing gymrat.config failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_config_inspect_module_when_imported_fresh_does_not_raise_import_error():
    """Import gymrat.config.inspect in a subprocess to detect circular imports.

    inspect.py is the most likely candidate for a circular import because it
    imports from the config package while the package init imports from
    resolve.py, which now delegates to inspect_config.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import gymrat.config.inspect"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, (
        f"Importing gymrat.config.inspect failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# resolve_benchless_config delegates to inspect pipeline
# ---------------------------------------------------------------------------


def test_resolve_benchless_config_when_valid_config_does_return_settled_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {
            "bench": "config-bench",
            "adapter": "custom-adapter",
            "samples": 20,
            "timeout_seconds": 3600,
            "unstable_noise_pct": 150.5,
        },
    )
    monkeypatch.chdir(tmp_path)

    result = resolve_benchless_config(CliFlags())

    assert result == BenchlessConfig(
        adapter="custom-adapter",
        samples=20,
        timeout_seconds=3600,
        unstable_noise_pct=150.5,
        primary="geomean",
    )


def test_resolve_benchless_config_when_multiple_problems_does_raise_first_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": 42, "samples": "bad"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError):
        resolve_benchless_config(CliFlags())


def test_resolve_benchless_config_when_filter_omits_names_does_raise_naming_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "filter": "npm run bench"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_benchless_config(CliFlags())

    message = str(exc.value)
    assert "filter" in message
    assert "{names}" in message


def test_resolve_benchless_config_when_runbook_missing_does_raise_naming_runbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "runbook": "missing.md"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_benchless_config(CliFlags())

    message = str(exc.value)
    assert "runbook" in message
    assert "missing.md" in message


def test_resolve_benchless_config_when_loop_keys_valid_does_carry_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {
            "bench": "config-bench",
            "checks": "npm test",
            "filter": "npm run bench -- {names}",
            "primary": "decode/time",
            "stop": {"target_value": 1.5, "max_iterations": 20},
            "hooks": {"before": "npm run warm-cache", "after": "npm run cool-down"},
        },
    )
    monkeypatch.chdir(tmp_path)

    result = resolve_benchless_config(CliFlags())

    assert result == BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="decode/time",
        checks="npm test",
        filter="npm run bench -- {names}",
        stop=StopConfig(target_value=1.5, max_iterations=20),
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down"),
    )


def test_resolve_benchless_config_when_target_value_with_geomean_does_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "stop": {"target_value": 1.5}})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_benchless_config(CliFlags())

    message = str(exc.value)
    assert "target_value" in message
    assert "geomean" in message
