"""Tests for the doctor bench section.

The adapter registry is the only mock — ``get_adapter`` is replaced so the test
controls whether resolution succeeds. PATH lookup uses ``shutil.which`` which is
patched to control executable availability.
"""

import pytest

from gymrat.doctor.bench import build_bench_section
from gymrat.errors import GymratError


def _patch_adapter_raises(monkeypatch: pytest.MonkeyPatch, error: GymratError) -> None:
    def boom(_name: str) -> object:
        raise error

    monkeypatch.setattr("gymrat.doctor.bench.get_adapter", boom)


def _patch_adapter_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gymrat.doctor.bench.get_adapter", lambda _name: None)  # pyrefly: ignore


# ---------------------------------------------------------------------------
# adapter resolution
# ---------------------------------------------------------------------------


def test_build_bench_section_when_adapter_unknown_does_fail_with_message_and_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    error = GymratError('Unknown adapter "bogus".', hint="valid adapters are: metric-lines, mitata")
    _patch_adapter_raises(monkeypatch, error)

    section = build_bench_section(bench="node bench.js", adapter="bogus")

    assert len(section.checks) == 1
    check = section.checks[0]
    assert check.name == "adapter"
    assert check.status == "fail"
    assert "Unknown adapter" in check.detail
    assert check.hint == "valid adapters are: metric-lines, mitata"


def test_build_bench_section_when_adapter_valid_does_report_ok(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_adapter_ok(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/node")  # pyrefly: ignore

    section = build_bench_section(bench="node bench.js", adapter="metric-lines")

    assert section.title == "Bench"
    adapter_check = section.checks[0]
    assert adapter_check.name == "adapter"
    assert adapter_check.status == "ok"
    assert "metric-lines" in adapter_check.detail


# ---------------------------------------------------------------------------
# bench command set
# ---------------------------------------------------------------------------


def test_build_bench_section_when_bench_unresolved_does_fail_naming_flag_and_config_key(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_adapter_ok(monkeypatch)

    section = build_bench_section(bench=None, adapter="metric-lines")

    bench_check = next(c for c in section.checks if c.name == "bench")
    assert bench_check.status == "fail"
    assert bench_check.hint is not None
    assert "--bench" in bench_check.hint


def test_build_bench_section_when_bench_set_does_report_ok_with_command(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_adapter_ok(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/node")  # pyrefly: ignore

    section = build_bench_section(bench="node bench.js", adapter="metric-lines")

    bench_check = next(c for c in section.checks if c.name == "bench")
    assert bench_check.status == "ok"
    assert "node bench.js" in bench_check.detail


# ---------------------------------------------------------------------------
# executable on PATH
# ---------------------------------------------------------------------------


def test_build_bench_section_when_executable_found_does_report_ok(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_adapter_ok(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/npx")  # pyrefly: ignore

    section = build_bench_section(bench="npx tsx bench.ts", adapter="metric-lines")

    exe_check = next(c for c in section.checks if c.name == "executable")
    assert exe_check.status == "ok"
    assert "npx" in exe_check.detail


def test_build_bench_section_when_executable_missing_does_warn(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_adapter_ok(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda _cmd: None)  # pyrefly: ignore

    section = build_bench_section(bench="node bench.js", adapter="metric-lines")

    exe_check = next(c for c in section.checks if c.name == "executable")
    assert exe_check.status == "warn"
    assert "node" in exe_check.detail


def test_build_bench_section_when_bench_unresolved_does_not_check_executable(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_adapter_ok(monkeypatch)

    section = build_bench_section(bench=None, adapter="metric-lines")

    assert not any(c.name == "executable" for c in section.checks)


# ---------------------------------------------------------------------------
# early return on adapter failure
# ---------------------------------------------------------------------------


def test_build_bench_section_when_adapter_fails_does_not_check_bench_or_executable(
    monkeypatch: pytest.MonkeyPatch,
):
    error = GymratError('Unknown adapter "bogus".')
    _patch_adapter_raises(monkeypatch, error)

    section = build_bench_section(bench="node bench.js", adapter="bogus")

    assert len(section.checks) == 1
    assert section.checks[0].name == "adapter"


# ---------------------------------------------------------------------------
# config problems → skip
# ---------------------------------------------------------------------------


def test_build_bench_section_when_config_problems_and_bench_none_does_skip():
    section = build_bench_section(bench=None, adapter="metric-lines", config_problems=True)

    assert len(section.checks) == 1
    check = section.checks[0]
    assert check.status == "ok"
    assert "Skipped" in check.detail


# ---------------------------------------------------------------------------
# PATH probe with shell command strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bench", "expected_exe"),
    [
        pytest.param('"path with spaces/node" bench.js', "path with spaces/node", id="quoted-exe"),
        pytest.param("VAR=1 make test", "make", id="env-var-prefix"),
        pytest.param("VAR=1 FOO=2 node bench.js", "node", id="multiple-env-vars"),
    ],
)
def test_build_bench_section_when_shell_command_does_extract_real_executable(
    monkeypatch: pytest.MonkeyPatch,
    bench: str,
    expected_exe: str,
):
    _patch_adapter_ok(monkeypatch)
    which_calls: list[str] = []

    def fake_which(cmd: str) -> str:
        which_calls.append(cmd)
        return f"/usr/bin/{cmd}"

    monkeypatch.setattr("shutil.which", fake_which)

    section = build_bench_section(bench=bench, adapter="metric-lines")

    exe_check = next(c for c in section.checks if c.name == "executable")
    assert exe_check.status == "ok"
    assert which_calls[0] == expected_exe


@pytest.mark.parametrize(
    "bench",
    [
        pytest.param("cd src && make bench", id="shell-operator-cd"),
        pytest.param("{ make bench; }", id="shell-brace-group"),
    ],
)
def test_build_bench_section_when_shell_metacharacters_does_skip_path_check(
    monkeypatch: pytest.MonkeyPatch,
    bench: str,
):
    _patch_adapter_ok(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda _cmd: None)  # pyrefly: ignore

    section = build_bench_section(bench=bench, adapter="metric-lines")

    assert not any(c.name == "executable" for c in section.checks)
