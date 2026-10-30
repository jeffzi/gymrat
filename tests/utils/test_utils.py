"""Behavioral tests for the generic helpers in :mod:`gymrat.utils`.

``write_text_atomic`` writes UTF-8 text to a temporary sibling, flushes it and
syncs it to disk, then renames it over the target, so a reader sees either the
previous content or the full new content — never a partial file. On failure
the target is untouched, the temporary file is removed, and the ``OSError``
propagates.
"""

import datetime as dt
import errno
import io
import math
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path, PureWindowsPath
from typing import override

import pytest
from pydantic import TypeAdapter

from gymrat.utils import (
    LINE_TERMINATORS,
    NonFiniteNumberError,
    Number,
    SamplingEta,
    abbreviate_home,
    balanced_counts_around,
    balancing_cycle,
    collapse_whitespace,
    color_from_env,
    describe_error,
    expected_got,
    fan_out,
    finite_or_none,
    first_line,
    format_clock,
    format_duration,
    fraction_of_median,
    is_balanced,
    is_broken_pipe,
    is_finite_or_warn,
    is_tty,
    limit_output,
    loads_strict_json,
    medians_by_name,
    named_subject,
    os_error_reason,
    otlp_endpoint,
    pluralize,
    point_stream_at_devnull,
    record_metric,
    stderr_text_of,
    stream_color_from_env,
    warn_to_stderr,
    write_text_atomic,
)
from tests._platform import needs_posix_file_modes
from tests._streams import FakeStream

OLD_TEXT = "previous content\n"
NEW_TEXT = "café ✓ new content\n"

LIMIT_BYTES = 8192


def _error_with(message: str, **streams: str | bytes) -> Exception:
    """Build a plain exception carrying the given ``stdout``/``stderr`` attributes."""
    error = Exception(message)
    for name, value in streams.items():
        setattr(error, name, value)
    return error


# ---------------------------------------------------------------------------
# finite_or_none
# ---------------------------------------------------------------------------


def test_finite_or_none_when_value_is_finite_does_return_it_unchanged():
    assert finite_or_none(0.0) == 0.0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(math.nan, id="not-a-number"),
        pytest.param(math.inf, id="positive-infinity"),
    ],
)
def test_finite_or_none_when_value_is_not_finite_does_return_none(value: float):
    assert finite_or_none(value) is None


# ---------------------------------------------------------------------------
# sample balance: balancing_cycle, is_balanced, balanced_counts_around
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        pytest.param(1, 1, id="one-target"),
        pytest.param(4, 4, id="even"),
        pytest.param(3, 6, id="odd"),
    ],
)
def test_balancing_cycle_when_given_a_target_count_does_return_its_cycle_length(
    targets: int, expected: int
):
    result = balancing_cycle(targets)

    assert result == expected


@pytest.mark.parametrize(
    ("samples", "targets", "expected"),
    [
        pytest.param(12, 3, True, id="multiple-of-the-cycle"),
        pytest.param(9, 3, False, id="multiple-of-n-but-not-2n"),
        pytest.param(0, 2, False, id="zero"),
    ],
)
def test_is_balanced_when_given_a_sample_count_does_require_a_positive_multiple_of_the_cycle(
    samples: int, targets: int, expected: bool
):
    result = is_balanced(samples, targets)

    assert result is expected


@pytest.mark.parametrize(
    ("samples", "targets", "expected"),
    [
        pytest.param(10, 3, (6, 12), id="both-sides"),
        pytest.param(4, 3, (6,), id="below-would-be-zero"),
    ],
)
def test_balanced_counts_around_when_count_is_unbalanced_does_return_the_nearest_valid_counts(
    samples: int, targets: int, expected: tuple[int, ...]
):
    result = balanced_counts_around(samples, targets)

    assert result == expected


# ---------------------------------------------------------------------------
# fraction_of_median
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("numerator", "median", "expected"),
    [
        pytest.param(5.0, -20.0, 25.0, id="scaled-by-the-median-magnitude"),
        pytest.param(1.0, 0.0, None, id="median-zero"),
        pytest.param(1.0, 5e-324, None, id="fraction-overflows"),
    ],
)
def test_fraction_of_median_when_given_a_median_does_scale_the_fraction_or_report_none(
    numerator: float, median: float, expected: float | None
):
    fraction = fraction_of_median(numerator, median, scale=100)

    assert fraction == expected


# ---------------------------------------------------------------------------
# expected_got
# ---------------------------------------------------------------------------

# An integer whose decimal form has more digits than the interpreter converts.
HUGE_INTEGER = 1 << 20_000


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        pytest.param(True, "true", id="boolean"),
    ],
)
def test_expected_got_when_value_is_json_encodable_does_render_it_as_json(
    value: object, rendered: str
):
    result = expected_got("a string", value)

    assert result.endswith(rendered)


def test_expected_got_when_value_is_not_json_encodable_does_render_its_repr():
    result = expected_got("an integer", dt.date(1979, 5, 27))

    assert "an integer" in result
    assert result.endswith("datetime.date(1979, 5, 27)")


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        pytest.param(HUGE_INTEGER, "a 20001-bit integer", id="integer"),
        pytest.param([HUGE_INTEGER], "a list too large to display", id="list-holding-it"),
    ],
)
def test_expected_got_when_integer_exceeds_digit_limit_does_render_a_bounded_description(
    value: object, rendered: str
):
    result = expected_got("a number at or below 5", value)

    assert result.endswith(rendered)


# ---------------------------------------------------------------------------
# abbreviate_home
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        pytest.param(".", "~", id="home-itself"),
        pytest.param("work/repo", "~/work/repo", id="under-home"),
    ],
)
def test_abbreviate_home_when_path_under_home_does_use_a_tilde(
    relative: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert abbreviate_home(str(tmp_path / relative)) == expected


def test_abbreviate_home_when_path_outside_home_does_return_it_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    outside = str(tmp_path / "elsewhere" / "repo")

    assert abbreviate_home(outside) == outside


def test_abbreviate_home_when_home_cannot_be_found_does_return_path_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    def no_home() -> Path:
        msg = "Could not determine home directory."
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "home", no_home)

    assert abbreviate_home("/srv/repo") == "/srv/repo"


class _WindowsPath(PureWindowsPath):
    """A Windows-flavoured path whose home is a fixed user directory."""

    @classmethod
    def home(cls) -> "_WindowsPath":
        """The fixed home directory every host sees."""
        return cls(r"C:\Users\ada")


def test_abbreviate_home_when_windows_path_under_home_does_use_forward_slashes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("gymrat.utils.Path", _WindowsPath)

    assert abbreviate_home(r"C:\Users\ada\work\repo") == "~/work/repo"


# ---------------------------------------------------------------------------
# pluralize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "noun", "expected"),
    [
        pytest.param(1, "metric", "1 metric", id="one"),
        pytest.param(0, "file", "0 files", id="zero"),
    ],
)
def test_pluralize_when_count_varies_does_add_s_unless_the_count_is_one(
    count: int, noun: str, expected: str
):
    assert pluralize(count, noun) == expected


# ---------------------------------------------------------------------------
# warn_to_stderr
# ---------------------------------------------------------------------------


def test_warn_to_stderr_when_called_does_write_message_with_newline_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    warn_to_stderr("hello")

    captured = capsys.readouterr()
    assert captured.err == "hello\n"


# ---------------------------------------------------------------------------
# os_error_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            OSError(errno.EACCES, "Permission denied", "/tmp/session.jsonl"),
            "Permission denied",
            id="os-error-with-strerror",
        ),
        pytest.param(OSError("disk full"), "disk full", id="os-error-with-bare-message"),
    ],
)
def test_os_error_reason_when_error_varies_does_prefer_strerror_over_the_message(
    error: OSError, expected: str
):
    assert os_error_reason(error) == expected


# ---------------------------------------------------------------------------
# fan_out
# ---------------------------------------------------------------------------


class _Event:
    """A distinct object whose identity the subscribers can check."""


def _raise_boom(_: _Event) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_fan_out_when_called_does_dispatch_to_subscribers_in_order():
    calls: list[tuple[str, _Event]] = []
    errors: list[Exception] = []

    def first(event: _Event) -> None:
        calls.append(("first", event))

    def second(event: _Event) -> None:
        calls.append(("second", event))

    dispatch = fan_out([first, second], errors.append)
    event = _Event()

    dispatch(event)

    assert [name for name, _ in calls] == ["first", "second"]


def test_fan_out_when_subscriber_raises_does_report_the_error():
    errors: list[Exception] = []
    dispatch = fan_out([_raise_boom], errors.append)

    dispatch(_Event())

    assert [type(error) for error in errors] == [RuntimeError]


def test_fan_out_when_subscriber_raises_does_still_call_later_subscribers():
    received: list[_Event] = []
    dispatch = fan_out([_raise_boom, received.append], lambda _error: None)
    event = _Event()

    dispatch(event)

    assert received == [event]


# ---------------------------------------------------------------------------
# write_text_atomic: successful write
# ---------------------------------------------------------------------------


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """A target file holding ``OLD_TEXT``."""
    path = tmp_path / "state.json"
    path.write_text(OLD_TEXT, encoding="utf-8")
    return path


def _entries(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


@pytest.mark.parametrize(
    "previous",
    [
        pytest.param(None, id="target-absent"),
        pytest.param(OLD_TEXT, id="target-exists"),
    ],
)
def test_write_text_atomic_when_called_does_write_utf8_text_without_leftover_temp(
    tmp_path: Path, previous: str | None
):
    path = tmp_path / "state.json"
    if previous is not None:
        path.write_text(previous, encoding="utf-8")

    write_text_atomic(path, NEW_TEXT)

    assert path.read_bytes() == NEW_TEXT.encode("utf-8")
    assert _entries(tmp_path) == ["state.json"]


# ---------------------------------------------------------------------------
# write_text_atomic: file mode
# ---------------------------------------------------------------------------


@pytest.fixture
def umask() -> Iterator[int]:
    """The process umask set to ``0o022``; the previous one is restored on teardown."""
    mask = 0o022
    previous = os.umask(mask)
    yield mask
    os.umask(previous)


@needs_posix_file_modes
def test_write_text_atomic_when_target_absent_does_create_it_with_the_umask_mode(
    tmp_path: Path, umask: int
):
    path = tmp_path / "state.json"

    write_text_atomic(path, NEW_TEXT)

    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


@pytest.fixture
def stripping_umask() -> Iterator[None]:
    """A umask that strips the group and other bits a rewritten file would otherwise lose."""
    previous = os.umask(0o007)
    yield
    os.umask(previous)


@needs_posix_file_modes
@pytest.mark.usefixtures("stripping_umask")
def test_write_text_atomic_when_target_exists_does_keep_its_mode(target: Path):
    target.chmod(0o755)

    write_text_atomic(target, NEW_TEXT)

    assert stat.S_IMODE(target.stat().st_mode) == 0o755


@needs_posix_file_modes
def test_write_text_atomic_when_target_is_owner_only_does_never_hold_new_content_in_a_wider_file(
    target: Path, monkeypatch: pytest.MonkeyPatch
):
    target.chmod(0o600)
    real_fsync = os.fsync
    wider_bits_when_synced: set[int] = set()

    def recording_fsync(fd: int) -> None:
        real_fsync(fd)
        synced = os.fstat(fd)
        if stat.S_ISREG(synced.st_mode):
            wider_bits_when_synced.add(stat.S_IMODE(synced.st_mode) & ~0o600)

    monkeypatch.setattr("os.fsync", recording_fsync)

    write_text_atomic(target, NEW_TEXT)

    assert wider_bits_when_synced == {0}


# ---------------------------------------------------------------------------
# write_text_atomic: failed write
# ---------------------------------------------------------------------------


def _fail_fsync(_directory: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    def exploding_fsync(fd: int) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("os.fsync", exploding_fsync)
    return "disk full"


def _fail_replace(_directory: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    def exploding_replace(src: object, dst: object) -> None:
        msg = "Read-only file system"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)
    return "Read-only file system"


@pytest.mark.parametrize(
    "break_write",
    [
        pytest.param(_fail_fsync, id="fsync-fails"),
        pytest.param(_fail_replace, id="rename-fails"),
    ],
)
def test_write_text_atomic_when_write_fails_does_raise_the_os_error(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    break_write: Callable[[Path, pytest.MonkeyPatch], str],
):
    reason = break_write(target.parent, monkeypatch)

    with pytest.raises(OSError, match=reason):
        write_text_atomic(target, NEW_TEXT)


@pytest.mark.parametrize(
    "break_write",
    [
        pytest.param(_fail_fsync, id="fsync-fails"),
        pytest.param(_fail_replace, id="rename-fails"),
    ],
)
def test_write_text_atomic_when_write_fails_does_leave_target_and_directory_untouched(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    break_write: Callable[[Path, pytest.MonkeyPatch], str],
):
    reason = break_write(target.parent, monkeypatch)

    with pytest.raises(OSError, match=reason):
        write_text_atomic(target, NEW_TEXT)

    assert target.read_text(encoding="utf-8") == OLD_TEXT
    assert _entries(target.parent) == ["state.json"]


# ---------------------------------------------------------------------------
# SamplingEta
# ---------------------------------------------------------------------------


def test_sampling_eta_advanced_when_called_does_return_an_accumulated_copy() -> None:
    eta = SamplingEta(total=4)

    advanced = eta.advanced(100.0).advanced(300.0)

    assert (advanced.completed, advanced.total_time_ms, advanced.total) == (2, 400.0, 4)


@pytest.mark.parametrize(
    ("completed", "total_time_ms", "total", "expected"),
    [
        pytest.param(0, 0.0, 5, None, id="no-finished-pass"),
        pytest.param(2, 300.0, 5, 450.0, id="average-times-remaining"),
        pytest.param(5, 500.0, 5, None, id="nothing-remaining"),
        pytest.param(6, 600.0, 5, None, id="completed-past-total"),
    ],
)
def test_sampling_eta_eta_ms_when_given_state_does_return_expected_estimate(
    completed: int, total_time_ms: float, total: int, expected: float | None
) -> None:
    eta = SamplingEta(completed=completed, total_time_ms=total_time_ms, total=total)

    assert eta.eta_ms == expected


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (59_999, "59s"),
        (60_000, "1m 0s"),
        (3_900_000, "1h 05m"),
        pytest.param(-1, "0s", id="negative-renders-zero"),
    ],
)
def test_format_duration_when_given_milliseconds_does_render_expected_duration(
    ms: float, expected: str
) -> None:
    assert format_duration(ms) == expected


# ---------------------------------------------------------------------------
# format_clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        pytest.param(9_999, "00:09", id="floors-partial-second"),
        pytest.param(465_000, "07:45", id="minutes"),
        pytest.param(4_065_000, "1:07:45", id="hour-tier"),
    ],
)
def test_format_clock_when_given_milliseconds_does_render_expected_clock(
    ms: float, expected: str
) -> None:
    assert format_clock(ms) == expected


# ---------------------------------------------------------------------------
# limit_output
# ---------------------------------------------------------------------------


_HUNDRED_A_LINE = "a" * 100 + "\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "a short line\nand another\n", "a short line\nand another\n", id="within-budget"
        ),
        pytest.param(
            _HUNDRED_A_LINE * 200,
            (_HUNDRED_A_LINE * (LIMIT_BYTES // len(_HUNDRED_A_LINE))).removesuffix("\n"),
            id="multi-line-overrun-cuts-at-the-last-whole-line",
        ),
        pytest.param("a" * 9000, "a" * LIMIT_BYTES, id="single-long-line-cuts-at-a-char"),
        pytest.param(
            "\n" + "a" * 9000,
            "\n" + "a" * (LIMIT_BYTES - 1),
            id="only-newline-at-byte-zero-cuts-at-a-char",
        ),
    ],
)
def test_limit_output_when_given_text_does_keep_it_within_the_byte_budget(
    text: str, expected: str
) -> None:
    result = limit_output(text)

    assert result == expected


# ---------------------------------------------------------------------------
# stderr_text_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            subprocess.CalledProcessError(1, ["git"], stderr=b"fatal: bad thing\n"),
            "fatal: bad thing",
            id="stderr-bytes-over-message",
        ),
        pytest.param(ValueError("plain message"), "plain message", id="stderr-absent"),
        pytest.param(
            subprocess.CalledProcessError(
                1, ["git", "commit"], output="hook rejected the commit\n", stderr=""
            ),
            "hook rejected the commit",
            id="blank-stderr-falls-back-to-stdout-str",
        ),
        pytest.param(
            subprocess.CalledProcessError(
                1, ["git", "commit"], output="on stdout", stderr="on stderr\n"
            ),
            "on stderr",
            id="both-non-blank-prefers-stderr",
        ),
        pytest.param(
            _error_with("real message", stdout="  \n\t", stderr=""),
            "real message",
            id="both-blank-falls-back-to-message",
        ),
    ],
)
def test_stderr_text_of_when_streams_vary_does_prefer_stderr_then_stdout_then_message(
    error: Exception, expected: str
):
    assert stderr_text_of(error) == expected


# ---------------------------------------------------------------------------
# color_from_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("force_color", "no_color", "expected"),
    [
        pytest.param(None, None, None, id="neither-defers"),
        pytest.param(None, "", False, id="no-color-empty-still-present"),
        pytest.param("1", None, True, id="force-color"),
        pytest.param("1", "1", True, id="force-color-beats-no-color"),
        pytest.param("", None, False, id="force-color-empty"),
        pytest.param("FALSE", None, False, id="force-color-false-uppercase"),
    ],
)
def test_color_from_env_when_variables_vary_does_apply_the_shared_precedence(
    color_env: Callable[[str | None, str | None], None],
    force_color: str | None,
    no_color: str | None,
    expected: bool | None,
):
    color_env(force_color, no_color)

    assert color_from_env() is expected


# ---------------------------------------------------------------------------
# is_tty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        pytest.param(FakeStream(tty=True), True, id="tty"),
        pytest.param(FakeStream(tty=False), False, id="non-tty"),
        pytest.param(object(), False, id="no-isatty"),
    ],
)
def test_is_tty_when_called_does_reflect_the_streams_isatty(stream: object, expected: bool):
    assert is_tty(stream) is expected


# ---------------------------------------------------------------------------
# stream_color_from_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("force_color", "no_color", "tty", "expected"),
    [
        pytest.param(None, None, True, True, id="tty-alone"),
        pytest.param(None, None, False, False, id="redirect-alone"),
        pytest.param("1", None, False, True, id="force-color-on-a-redirect"),
        pytest.param("0", None, True, False, id="force-color-zero-on-a-tty"),
    ],
)
def test_stream_color_from_env_when_environment_and_tty_vary_does_let_the_environment_win(
    color_env: Callable[[str | None, str | None], None],
    force_color: str | None,
    no_color: str | None,
    tty: bool,
    expected: bool,
):
    color_env(force_color, no_color)

    assert stream_color_from_env(FakeStream(tty=tty)) is expected


# ---------------------------------------------------------------------------
# otlp_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("  http://localhost:4318\n", "http://localhost:4318", id="padded"),
        pytest.param("  \t", None, id="blank"),
        pytest.param(None, None, id="unset"),
    ],
)
def test_otlp_endpoint_when_given_a_raw_value_does_trim_it_or_report_none(
    value: str | None, expected: str | None
):
    assert otlp_endpoint(value) == expected


# ---------------------------------------------------------------------------
# loads_strict_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "literal"),
    [
        pytest.param('{"a":NaN}', "NaN", id="nan-literal"),
        pytest.param('{"a":1e999}', "1e999", id="overflowing-literal"),
    ],
)
def test_loads_strict_json_when_a_number_is_not_finite_does_raise_naming_the_literal(
    line: str, literal: str
):
    with pytest.raises(NonFiniteNumberError) as excinfo:
        loads_strict_json(line)

    assert literal in str(excinfo.value)


def test_loads_strict_json_when_numbers_finite_does_decode_them_unchanged():
    decoded = loads_strict_json('{"a":1.5}')

    assert decoded == {"a": 1.5}


# ---------------------------------------------------------------------------
# medians_by_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [
        pytest.param(
            ({"total_ms": 100, "alloc_bytes": 40}, {"total_ms": 300}),
            {"total_ms": 200, "alloc_bytes": 40},
            id="even-count",
        ),
        pytest.param(
            ({"total_ms": 300, "alloc_bytes": 40}, {"total_ms": 100}, {"total_ms": 200}),
            {"total_ms": 200, "alloc_bytes": 40},
            id="odd-count",
        ),
    ],
)
def test_medians_by_name_when_a_round_omits_a_metric_does_median_only_the_rounds_that_report_it(
    rounds: tuple[dict[str, float], ...], expected: dict[str, float]
):
    medians = medians_by_name(rounds)

    assert medians == expected


# ---------------------------------------------------------------------------
# first_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("first\nsecond\nthird", "first", id="line-feed-ends-the-line"),
        pytest.param("first\u2028second", "first\u2028second", id="line-separator-does-not"),
    ],
)
def test_first_line_when_text_has_several_lines_does_return_the_text_before_the_first_newline(
    text: str, expected: str
):
    assert first_line(text) == expected


# ---------------------------------------------------------------------------
# collapse_whitespace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("fatal:\tnot\r\na\n\nrepo", "fatal: not a repo", id="tab-crlf-and-blank-line"),
    ],
)
def test_collapse_whitespace_when_text_has_whitespace_runs_does_join_words_with_one_space(
    text: str, expected: str
):
    collapsed = collapse_whitespace(text)

    assert collapsed == expected


# ---------------------------------------------------------------------------
# describe_error
# ---------------------------------------------------------------------------


def test_describe_error_when_string_holds_a_line_terminator_does_escape_only_the_terminator():
    described = describe_error('naïve "x"\nboom')

    assert described == r'naïve "x"\nboom'


def test_describe_error_when_value_is_not_a_string_does_render_it_as_json():
    described = describe_error({"code": 7})

    assert described == '{"code": 7}'


# ---------------------------------------------------------------------------
# record_metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metrics", "warned"),
    [
        pytest.param({"total_ms": 1.0}, True, id="name-already-recorded"),
        pytest.param({"alloc_bytes": 1.0}, False, id="name-new"),
    ],
)
def test_record_metric_when_storing_a_reading_does_keep_it_warning_only_on_a_collision(
    metrics: dict[str, float], warned: bool
):
    warnings: list[str] = []

    record_metric(metrics, "total_ms", 2.0, warnings.append, "rename the runs")

    assert metrics["total_ms"] == 2.0
    assert [("total_ms" in w, "rename the runs" in w) for w in warnings] == (
        [(True, True)] if warned else []
    )


# ---------------------------------------------------------------------------
# Number
# ---------------------------------------------------------------------------


def test_number_when_validating_an_integer_does_keep_it_an_integer():
    validated = TypeAdapter(Number).validate_python(42)

    assert (validated, type(validated)) == (42, int)


# ---------------------------------------------------------------------------
# named_subject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param({"alias": "line\u2028break"}, 'benchmark "line\\u2028break"', id="string"),
        pytest.param({"alias": 3}, "benchmark", id="not-a-string"),
        pytest.param(None, "benchmark", id="not-an-object"),
    ],
)
def test_named_subject_when_entry_varies_does_name_it_only_by_a_string_field(
    entry: object, expected: str
):
    assert named_subject("benchmark", entry, "alias") == expected


# ---------------------------------------------------------------------------
# is_finite_or_warn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "finite"),
    [
        pytest.param(1.5, True, id="finite"),
        pytest.param(math.inf, False, id="infinity"),
        pytest.param(10**400, False, id="integer-overflows-float"),
    ],
)
def test_is_finite_or_warn_when_checking_a_reading_does_warn_only_when_not_finite(
    value: float, finite: bool
):
    warnings: list[str] = []

    result = is_finite_or_warn(value, warnings.append, "run", "stats.p50")

    assert result is finite
    assert ["stats.p50" in w for w in warnings] == ([] if finite else [True])


# ---------------------------------------------------------------------------
# LINE_TERMINATORS
# ---------------------------------------------------------------------------


def test_line_terminators_when_scanning_every_code_point_does_match_exactly_splitlines_breaks():
    code_points = [chr(code) for code in range(sys.maxunicode + 1)]

    matched = {char for char in code_points if LINE_TERMINATORS.search(char)}

    assert matched == {char for char in code_points if len(f"a{char}b".splitlines()) == 2}


# ---------------------------------------------------------------------------
# is_broken_pipe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "platform", "expected"),
    [
        pytest.param(BrokenPipeError(), "linux", True, id="posix-broken-pipe"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument"), "win32", True, id="windows-einval"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument"), "linux", False, id="posix-einval"),
        pytest.param(OSError(errno.EACCES, "Access denied"), "win32", False, id="windows-eacces"),
        pytest.param(ValueError("banana"), "win32", False, id="not-an-os-error"),
    ],
)
def test_is_broken_pipe_when_platform_and_error_vary_does_recognize_the_closed_pipe_error(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, platform: str, expected: bool
):
    monkeypatch.setattr("sys.platform", platform)

    assert is_broken_pipe(error) is expected


# ---------------------------------------------------------------------------
# point_stream_at_devnull
# ---------------------------------------------------------------------------


class _BadDescriptorStream(io.StringIO):
    """A stream whose descriptor is invalid, so redirecting it fails."""

    @override
    def fileno(self) -> int:
        return -1


def _open_descriptors() -> set[str]:
    """Snapshot this process's open descriptors from ``/dev/fd``, to compare for leaks."""
    return {entry.name for entry in Path("/dev/fd").iterdir()}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirected_does_point_descriptor_at_devnull_leaking_none(
    tmp_path: Path,
):
    with (tmp_path / "out.txt").open("w") as stream:
        before = _open_descriptors()

        point_stream_at_devnull(stream)

        assert os.path.samestat(os.fstat(stream.fileno()), Path(os.devnull).stat())
        assert _open_descriptors() == before


def test_point_stream_at_devnull_when_stream_has_no_descriptor_does_return_without_raising():
    stream = io.StringIO()

    point_stream_at_devnull(stream)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirect_fails_does_raise_leaking_no_descriptor():
    before = _open_descriptors()

    with pytest.raises(OSError, match="Bad file descriptor"):
        point_stream_at_devnull(_BadDescriptorStream())

    assert _open_descriptors() == before
