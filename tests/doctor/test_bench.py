"""Tests for the doctor bench smoke-run section.

Only the two system boundaries are mocked — ``exec`` (the subprocess layer) and
``get_adapter`` (the adapter registry) — exactly as the upstream suite does. Every
other behavior is exercised through the real ``build_bench_section`` so the
status/detail/hint output reflects genuine control flow.
"""

import asyncio

import pytest

from gymrat_py.adapters.types import AdapterError, MetricDefaults, WarnSink, warn_to_stderr
from gymrat_py.config import KindEntry, MetricEntry
from gymrat_py.doctor.bench import BenchSectionInput, build_bench_section
from gymrat_py.errors import GymratError
from gymrat_py.exec import ExecResult, ExecTimeoutError

# ---------------------------------------------------------------------------
# Fakes and builders
# ---------------------------------------------------------------------------


class FakeAdapter:
    """A stand-in adapter whose parse output, warnings, and kind the test controls."""

    name = "metric-lines"

    def __init__(
        self,
        *,
        parsed: dict[str, float] | None = None,
        warnings: tuple[str, ...] = (),
        kind: str = "other",
        parse_error: Exception | None = None,
    ):
        self._parsed = {"latency": 42.0} if parsed is None else parsed
        self._warnings = warnings
        self._kind = kind
        self._parse_error = parse_error

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        for message in self._warnings:
            warn(message)
        if self._parse_error is not None:
            raise self._parse_error
        return self._parsed

    def defaults(self, metric_name: str) -> MetricDefaults:
        return MetricDefaults(direction="lower", kind=self._kind)


def make_input(**overrides: object) -> BenchSectionInput:
    base: dict[str, object] = {
        "bench": "node bench.js",
        "adapter": "metric-lines",
        "timeout_seconds": 30,
        "primary": "geomean",
        "repo_root": "/project",
        "abort": asyncio.Event(),
    }
    base.update(overrides)
    return BenchSectionInput(**base)  # pyrefly: ignore


def exec_result(*, stdout: str = "METRIC latency=42\n", stderr: str = "", exit_code: int = 0):
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def exec_timeout(timeout_ms: int = 30_000) -> ExecTimeoutError:
    return ExecTimeoutError(
        stdout="", stderr="", timeout_ms=timeout_ms, stdout_bytes=0, stderr_bytes=0
    )


def patch_adapter(monkeypatch: pytest.MonkeyPatch, adapter: FakeAdapter) -> None:
    def resolve(_name: str) -> FakeAdapter:
        return adapter

    monkeypatch.setattr("gymrat_py.doctor.bench.get_adapter", resolve)


def patch_adapter_raises(monkeypatch: pytest.MonkeyPatch, error: GymratError) -> None:
    def boom(_name: str) -> FakeAdapter:
        raise error

    monkeypatch.setattr("gymrat_py.doctor.bench.get_adapter", boom)


def patch_exec(
    monkeypatch: pytest.MonkeyPatch, result: ExecResult | ExecTimeoutError
) -> list[tuple[str, object]]:
    """Replace the exec seam with a fake that records its calls and hands back ``result``."""
    calls: list[tuple[str, object]] = []

    async def fake_exec(command: str, options: object) -> ExecResult | ExecTimeoutError:
        calls.append((command, options))
        return result

    monkeypatch.setattr("gymrat_py.doctor.bench.run_exec", fake_exec)
    return calls


def status_details(section: object, status: str) -> str:
    return "\n".join(check.detail for check in section.checks if check.status == status)  # pyrefly: ignore


def first_fail(section: object):
    fails = [check for check in section.checks if check.status == "fail"]  # pyrefly: ignore
    assert fails, "expected at least one fail check"
    return fails[0]


# ---------------------------------------------------------------------------
# section metadata
# ---------------------------------------------------------------------------


async def test_build_bench_section_has_title_bench():
    section = await build_bench_section(make_input(no_bench=True))

    assert section.title == "Bench"


# ---------------------------------------------------------------------------
# adapter resolution — runs before any skip
# ---------------------------------------------------------------------------


async def test_build_bench_section_when_adapter_unknown_does_fail_with_message_and_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    error = GymratError('Unknown adapter "bogus".', hint="valid adapters are: metric-lines, mitata")
    patch_adapter_raises(monkeypatch, error)
    calls = patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input(adapter="bogus"))

    assert len(section.checks) == 1
    check = section.checks[0]
    assert check.name == "adapter"
    assert check.status == "fail"
    assert "Unknown adapter" in check.detail
    assert check.hint == "valid adapters are: metric-lines, mitata"
    assert calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"no_bench": True}, id="no-bench"),
        pytest.param({"config_failed": True}, id="config-failed"),
    ],
)
async def test_build_bench_section_when_adapter_unknown_under_skip_still_fails(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
):
    error = GymratError('Unknown adapter "bogus".', hint="valid adapters are: metric-lines, mitata")
    patch_adapter_raises(monkeypatch, error)

    section = await build_bench_section(make_input(adapter="bogus", **overrides))

    assert len(section.checks) == 1
    check = section.checks[0]
    assert check.status == "fail"
    assert "Unknown adapter" in check.detail


# ---------------------------------------------------------------------------
# skips and unresolved bench
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        pytest.param({"no_bench": True}, "--no-bench", id="no-bench"),
        pytest.param({"config_failed": True}, "config", id="config-failed"),
    ],
)
async def test_build_bench_section_when_skipped_does_return_single_ok_without_running_exec(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], needle: str
):
    patch_adapter(monkeypatch, FakeAdapter())
    calls = patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input(**overrides))

    assert len(section.checks) == 1
    check = section.checks[0]
    assert check.status == "ok"
    assert needle in check.detail
    assert calls == []


async def test_build_bench_section_when_bench_unresolved_does_fail_naming_flag_and_config_key(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter())
    calls = patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input(bench=None))

    assert len(section.checks) == 1
    check = section.checks[0]
    assert check.status == "fail"
    assert check.hint is not None
    assert "--bench" in check.hint
    assert calls == []


# ---------------------------------------------------------------------------
# exec invocation
# ---------------------------------------------------------------------------


async def test_build_bench_section_when_running_does_exec_command_with_cwd_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter())
    calls = patch_exec(monkeypatch, exec_result())

    await build_bench_section(
        make_input(bench="node bench.js", repo_root="/my/repo", timeout_seconds=60)
    )

    assert len(calls) == 1
    command, options = calls[0]
    assert command == "node bench.js"
    assert options.cwd == "/my/repo"  # pyrefly: ignore
    assert options.timeout_ms == 60_000  # pyrefly: ignore


async def test_build_bench_section_when_running_does_forward_the_abort_event_to_exec(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter())
    calls = patch_exec(monkeypatch, exec_result())
    abort = asyncio.Event()

    await build_bench_section(make_input(abort=abort))

    _command, options = calls[0]
    assert options.abort is abort  # pyrefly: ignore


# ---------------------------------------------------------------------------
# exec failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        pytest.param(
            "Error: something broke",
            "Bench command exited with code 1: Error: something broke",
            id="stderr-content",
        ),
        pytest.param("", "Bench command exited with code 1", id="stderr-empty"),
        pytest.param("  \n  ", "Bench command exited with code 1", id="stderr-whitespace"),
    ],
)
async def test_build_bench_section_when_exit_nonzero_does_fail_naming_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, stderr: str, expected: str
):
    patch_adapter(monkeypatch, FakeAdapter())
    patch_exec(monkeypatch, exec_result(exit_code=1, stderr=stderr))

    section = await build_bench_section(make_input())

    assert first_fail(section).detail == expected


async def test_build_bench_section_when_timeout_does_fail_naming_limit_and_timeout_key(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter())
    patch_exec(monkeypatch, exec_timeout(timeout_ms=30_000))

    section = await build_bench_section(make_input(timeout_seconds=30))

    check = first_fail(section)
    assert "30" in check.detail
    assert "timeout" in (check.hint or "").lower()


async def test_build_bench_section_when_parse_raises_adapter_error_does_fail_with_message(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parse_error=AdapterError("No usable metrics found")))
    patch_exec(monkeypatch, exec_result(stdout="garbage"))

    section = await build_bench_section(make_input())

    assert "No usable metrics found" in first_fail(section).detail


async def test_build_bench_section_when_parse_raises_non_adapter_error_does_fail_naming_exception_type_and_message(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parse_error=ValueError("unexpected format")))
    patch_exec(monkeypatch, exec_result(stdout="garbage"))

    section = await build_bench_section(make_input())

    check = first_fail(section)
    assert check.name == "parse"
    assert "ValueError" in check.detail
    assert "unexpected format" in check.detail


async def test_build_bench_section_when_adapter_warns_then_parse_fails_does_report_warnings_alongside_parse_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(
        monkeypatch,
        FakeAdapter(
            warnings=("Skipped line 3: unrecognized format",),
            parse_error=AdapterError("No usable metrics found"),
        ),
    )
    patch_exec(monkeypatch, exec_result(stdout="garbage"))

    section = await build_bench_section(make_input())

    assert "No usable metrics found" in first_fail(section).detail
    assert "Skipped line 3" in status_details(section, "warn")


# ---------------------------------------------------------------------------
# success and the parsed-metric summary
# ---------------------------------------------------------------------------


async def test_build_bench_section_when_parse_succeeds_does_report_metric_count_and_names(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parsed={"latency": 42.0, "throughput": 100.0}))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input())

    ok_detail = status_details(section, "ok")
    assert "2" in ok_detail
    assert "latency" in ok_detail
    assert "throughput" in ok_detail


async def test_build_bench_section_when_many_metrics_does_cap_the_name_list_at_five(
    monkeypatch: pytest.MonkeyPatch,
):
    parsed = dict.fromkeys(("m1", "m2", "m3", "m4", "m5", "m6", "m7"), 1.0)
    patch_adapter(monkeypatch, FakeAdapter(parsed=parsed))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input())

    ok_detail = status_details(section, "ok")
    assert "… (7 total)" in ok_detail
    assert "m6" not in ok_detail
    assert "m7" not in ok_detail


async def test_build_bench_section_when_adapter_warns_does_report_warnings_as_their_own_check(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(warnings=("Skipped line 3: unrecognized format",)))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input())

    assert "Skipped line 3" in status_details(section, "warn")
    assert "Skipped line 3" not in status_details(section, "ok")


# ---------------------------------------------------------------------------
# post-parse cross-checks
# ---------------------------------------------------------------------------


async def test_build_bench_section_when_primary_absent_and_not_geomean_does_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parsed={"latency": 42.0}))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input(primary="throughput"))

    check = next(c for c in section.checks if c.status == "fail" and "throughput" in c.detail)
    assert "primary" in check.detail.lower()


async def test_build_bench_section_when_primary_is_geomean_does_not_fail_on_absence(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parsed={"latency": 42.0}))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input(primary="geomean"))

    assert [c for c in section.checks if c.status == "fail"] == []


async def test_build_bench_section_when_config_metric_missing_does_warn_listing_the_name(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parsed={"latency": 42.0}))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(
        make_input(metrics={"missing_metric": MetricEntry(gating=True)})
    )

    assert "missing_metric" in status_details(section, "warn")


async def test_build_bench_section_when_config_kind_unmatched_does_warn_listing_the_name(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_adapter(monkeypatch, FakeAdapter(parsed={"latency": 42.0}, kind="time"))
    patch_exec(monkeypatch, exec_result())

    section = await build_bench_section(make_input(kinds={"memory": KindEntry(gating=True)}))

    assert "memory" in status_details(section, "warn")
