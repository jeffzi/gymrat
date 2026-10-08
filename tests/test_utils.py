"""Behavioral tests for the generic helpers in :mod:`gymrat.utils`.

``write_text_atomic`` writes UTF-8 text to a temporary sibling, flushes it and
syncs it to disk, then renames it over the target, so a reader sees either the
previous content or the full new content — never a partial file. On failure
the target is untouched, the temporary file is removed, and the ``OSError``
propagates.
"""

import datetime
import math
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path, PureWindowsPath

import pytest

from gymrat.utils import (
    SamplingEta,
    abbreviate_home,
    color_from_env,
    expected_got,
    fan_out,
    finite_or_none,
    format_clock,
    format_duration,
    format_eta,
    format_timestamp,
    is_tty,
    limit_output,
    otlp_endpoint,
    pluralize,
    stderr_text_of,
    stream_color_from_env,
    warn_to_stderr,
    write_text_atomic,
)
from tests._mode_bits import needs_mode_bits
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


@pytest.mark.parametrize("value", [0.0, -0.5, 12.25, 1e300, -1e300])
def test_finite_or_none_when_value_is_finite_does_return_it_unchanged(value: float):
    assert finite_or_none(value) == value


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(math.nan, id="not-a-number"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
    ],
)
def test_finite_or_none_when_value_is_not_finite_does_return_none(value: float):
    assert finite_or_none(value) is None


# ---------------------------------------------------------------------------
# expected_got
# ---------------------------------------------------------------------------

# An integer whose decimal form has more digits than the interpreter converts.
HUGE_INTEGER = 1 << 20_000


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        pytest.param("ten", '"ten"', id="string"),
        pytest.param('say "hi"\n', '"say \\"hi\\"\\n"', id="string-needing-escapes"),
        pytest.param(7, "7", id="integer"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param(math.nan, "NaN", id="not-a-number"),
        pytest.param(-math.inf, "-Infinity", id="negative-infinity"),
        pytest.param(True, "true", id="boolean"),
        pytest.param(None, "null", id="none"),
        pytest.param(["a", 1], '["a", 1]', id="list"),
        pytest.param({"cmd": "x"}, '{"cmd": "x"}', id="mapping"),
    ],
)
def test_expected_got_when_value_is_json_encodable_does_render_it_as_json(
    value: object, rendered: str
):
    assert expected_got("a string", value) == f"expected a string, got {rendered}"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        pytest.param(datetime.date(1979, 5, 27), "datetime.date(1979, 5, 27)", id="date"),
        pytest.param({1}, "{1}", id="set"),
        pytest.param([b"raw"], "[b'raw']", id="list-holding-bytes"),
    ],
)
def test_expected_got_when_value_is_not_json_encodable_does_render_its_repr(
    value: object, rendered: str
):
    assert expected_got("an integer", value) == f"expected an integer, got {rendered}"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        pytest.param(HUGE_INTEGER, "a 20001-bit integer", id="integer"),
        pytest.param(-HUGE_INTEGER, "a 20001-bit integer", id="negative-integer"),
        pytest.param([HUGE_INTEGER], "a list too large to display", id="list-holding-it"),
        pytest.param(
            {"samples": HUGE_INTEGER}, "a dict too large to display", id="mapping-holding-it"
        ),
    ],
)
def test_expected_got_when_integer_exceeds_digit_limit_does_render_a_bounded_description(
    value: object, rendered: str
):
    assert expected_got("a number at or below 5", value) == (
        f"expected a number at or below 5, got {rendered}"
    )


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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("work/repo", "work/repo", id="relative"),
        pytest.param("{home}/../elsewhere", "~/../elsewhere", id="dot-dot-kept-lexically"),
    ],
)
def test_abbreviate_home_when_path_is_not_normalized_does_compare_it_as_written(
    path: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    assert abbreviate_home(path.format(home=tmp_path)) == expected


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
        pytest.param(2, "kept iteration", "2 kept iterations", id="plural"),
        pytest.param(1, "metric", "1 metric", id="one"),
        pytest.param(0, "file", "0 files", id="zero"),
        pytest.param(-1, "file", "-1 files", id="negative"),
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
    assert captured.out == ""


# ---------------------------------------------------------------------------
# fan_out
# ---------------------------------------------------------------------------


class _Event:
    """A distinct object whose identity the subscribers can check."""


def _raise_boom(_: _Event) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_fan_out_when_called_does_dispatch_identical_event_to_subscribers_in_order():
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
    assert all(received is event for _, received in calls)
    assert errors == []


def test_fan_out_when_subscriber_raises_does_report_error_and_call_remaining():
    received: list[_Event] = []
    errors: list[Exception] = []
    dispatch = fan_out([_raise_boom, received.append], errors.append)
    event = _Event()

    dispatch(event)

    assert received == [event]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "boom"


# ---------------------------------------------------------------------------
# write_text_atomic: successful write
# ---------------------------------------------------------------------------


@pytest.fixture
def target(tmp_path: Path) -> Iterator[Path]:
    """A target file holding ``OLD_TEXT``; its directory is made writable again on teardown."""
    path = tmp_path / "state.json"
    path.write_text(OLD_TEXT, encoding="utf-8")
    yield path
    tmp_path.chmod(0o755)


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


def test_write_text_atomic_when_renaming_does_swap_in_a_fully_synced_file_over_untouched_target(
    target: Path, monkeypatch: pytest.MonkeyPatch
):
    real_fsync = os.fsync
    real_replace = os.replace
    synced_sizes: dict[int, int] = {}
    at_rename: dict[str, object] = {}

    def recording_fsync(fd: int) -> None:
        real_fsync(fd)
        stat = os.fstat(fd)
        synced_sizes[stat.st_ino] = stat.st_size

    def observing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        at_rename["size_when_synced"] = synced_sizes.get(Path(src).stat().st_ino)
        at_rename["source_text"] = Path(src).read_text(encoding="utf-8")
        at_rename["target_text"] = Path(dst).read_text(encoding="utf-8")
        real_replace(src, dst)

    monkeypatch.setattr("os.fsync", recording_fsync)
    monkeypatch.setattr("os.replace", observing_replace)

    write_text_atomic(target, NEW_TEXT)

    assert at_rename == {
        "size_when_synced": len(NEW_TEXT.encode("utf-8")),
        "source_text": NEW_TEXT,
        "target_text": OLD_TEXT,
    }


# ---------------------------------------------------------------------------
# write_text_atomic: file mode
# ---------------------------------------------------------------------------

_posix_modes = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


@pytest.fixture(params=[pytest.param(0o022, id="umask-022"), pytest.param(0o007, id="umask-007")])
def umask(request: pytest.FixtureRequest) -> Iterator[int]:
    """The process umask set to each mask under test; the previous one is restored on teardown."""
    mask: int = request.param
    previous = os.umask(mask)
    yield mask
    os.umask(previous)


@_posix_modes
def test_write_text_atomic_when_target_absent_does_create_it_with_the_umask_mode(
    tmp_path: Path, umask: int
):
    path = tmp_path / "state.json"

    write_text_atomic(path, NEW_TEXT)

    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


@_posix_modes
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(0o644, id="world-readable"),
        pytest.param(0o755, id="executable"),
        pytest.param(0o600, id="owner-only"),
    ],
)
@pytest.mark.usefixtures("umask")
def test_write_text_atomic_when_target_exists_does_keep_its_mode(target: Path, mode: int):
    target.chmod(mode)

    write_text_atomic(target, NEW_TEXT)

    assert stat.S_IMODE(target.stat().st_mode) == mode


@_posix_modes
@pytest.mark.usefixtures("umask")
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


def _make_directory_read_only(directory: Path, _monkeypatch: pytest.MonkeyPatch) -> str:
    directory.chmod(0o555)
    return "Permission denied"


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
        pytest.param(
            _make_directory_read_only,
            id="read-only-directory",
            marks=needs_mode_bits,
        ),
        pytest.param(_fail_fsync, id="fsync-fails"),
        pytest.param(_fail_replace, id="rename-fails"),
    ],
)
def test_write_text_atomic_when_write_fails_does_raise_and_leave_target_untouched(
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
    assert (eta.completed, eta.total_time_ms) == (0, 0.0)


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
        (0, "0s"),
        (999, "0s"),
        (1000, "1s"),
        (45_000, "45s"),
        (59_999, "59s"),
        (60_000, "1m 0s"),
        (90_000, "1m 30s"),
        (723_000, "12m 3s"),
        (3_599_999, "59m 59s"),
        (3_600_000, "1h 00m"),
        (3_900_000, "1h 05m"),
        (5_400_000, "1h 30m"),
        (7_200_000, "2h 00m"),
        pytest.param(36_000_000, "10h 00m", id="multi-digit-hours"),
        pytest.param(-1, "0s", id="negative-renders-zero"),
        pytest.param(-1000, "0s", id="negative-second-renders-zero"),
        pytest.param(-999_999, "0s", id="large-negative-renders-zero"),
    ],
)
def test_format_duration_when_given_milliseconds_does_render_expected_duration(
    ms: float, expected: str
) -> None:
    assert format_duration(ms) == expected


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("at_ms", "run_start_ms", "expected"),
    [
        pytest.param(1_000, 1_000, "[00:00:00]", id="zero-elapsed"),
        pytest.param(999, 1_000, "[00:00:00]", id="negative-elapsed-clamps-to-zero"),
        pytest.param(0, 90_000, "[00:00:00]", id="large-negative-elapsed-clamps-to-zero"),
        pytest.param(90_000, None, "[00:00:00]", id="unanchored-run"),
        pytest.param(91_000, 1_000, "[00:01:30]", id="ninety-seconds-elapsed"),
        pytest.param(3_601_000, 1_000, "[01:00:00]", id="one-hour-elapsed"),
        pytest.param(36_001_000, 1_000, "[10:00:00]", id="multi-digit-hours-elapsed"),
    ],
)
def test_format_timestamp_when_given_a_moment_does_render_the_elapsed_clock(
    at_ms: float, run_start_ms: float | None, expected: str
) -> None:
    assert format_timestamp(at_ms, run_start_ms) == expected


# ---------------------------------------------------------------------------
# format_clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        pytest.param(0, "00:00", id="zero"),
        pytest.param(9_000, "00:09", id="sub-minute"),
        pytest.param(9_999, "00:09", id="floors-partial-second"),
        pytest.param(59_999, "00:59", id="last-second-before-minute-tier"),
        pytest.param(465_000, "07:45", id="minutes"),
        pytest.param(3_599_000, "59:59", id="last-second-before-hour-tier"),
        pytest.param(3_600_000, "1:00:00", id="hour-tier-starts"),
        pytest.param(4_065_000, "1:07:45", id="hour-tier"),
        pytest.param(36_000_000, "10:00:00", id="multi-digit-hours"),
        pytest.param(-1, "00:00", id="negative-clamps-to-zero"),
        pytest.param(-90_000, "00:00", id="large-negative-clamps-to-zero"),
    ],
)
def test_format_clock_when_given_milliseconds_does_render_expected_clock(
    ms: float, expected: str
) -> None:
    assert format_clock(ms) == expected


# ---------------------------------------------------------------------------
# format_eta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (0, "~1s left"),
        (500, "~1s left"),
        (999, "~1s left"),
        (1000, "~1s left"),
        (48200, "~48s left"),
        (59999, "~1m left"),
        (60000, "~1m left"),
        (130000, "~2m 10s left"),
        (120000, "~2m left"),
        (3599999, "~1h left"),
        (3600000, "~1h left"),
        (3900000, "~1h 05m left"),
        (7200000, "~2h left"),
        (7260000, "~2h 01m left"),
    ],
)
def test_format_eta_when_given_milliseconds_does_render_expected_eta(
    ms: float, expected: str
) -> None:
    assert format_eta(ms) == expected


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
            "café résumé naïve €42",  # cspell:disable-line
            "café résumé naïve €42",  # cspell:disable-line
            id="multi-byte-within-budget",
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
    assert len(result.encode("utf-8")) <= LIMIT_BYTES


@pytest.mark.parametrize(
    ("char", "prefix"),
    [
        pytest.param("é", "", id="2-byte-accent"),
        pytest.param("€", "", id="3-byte-euro"),
        pytest.param("\U0001f389", "a", id="4-byte-emoji-with-ascii-prefix"),
    ],
)
def test_limit_output_when_cut_splits_wide_char_does_drop_partial_bytes(
    char: str,
    prefix: str,
) -> None:
    char_size = len(char.encode("utf-8"))
    prefix_size = len(prefix.encode("utf-8"))
    remaining = LIMIT_BYTES - prefix_size
    full_chars = remaining // char_size
    text = prefix + char * 9000

    result = limit_output(text)

    assert result == prefix + char * full_chars
    assert len(result.encode("utf-8")) <= LIMIT_BYTES
    assert "\N{REPLACEMENT CHARACTER}" not in result


# ---------------------------------------------------------------------------
# stderr_text_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            subprocess.CalledProcessError(1, ["git"], stderr="fatal: bad thing\n"),
            "fatal: bad thing",
            id="stderr-str-over-message",
        ),
        pytest.param(
            subprocess.CalledProcessError(1, ["git"], stderr=b"fatal: bad thing\n"),
            "fatal: bad thing",
            id="stderr-bytes-over-message",
        ),
        pytest.param(_error_with("real message", stderr=""), "real message", id="stderr-empty"),
        pytest.param(
            _error_with("real message", stderr="   \n\t"), "real message", id="stderr-blank"
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
                1, ["git", "commit"], output=b"hook rejected the commit\n", stderr=""
            ),
            "hook rejected the commit",
            id="blank-stderr-falls-back-to-stdout-bytes",
        ),
        pytest.param(
            subprocess.CalledProcessError(
                1, ["git", "commit"], output="on stdout", stderr="on stderr"
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
        pytest.param(None, "1", False, id="no-color"),
        pytest.param(None, "", False, id="no-color-empty-still-present"),
        pytest.param("1", None, True, id="force-color"),
        pytest.param("1", "1", True, id="force-color-beats-no-color"),
        pytest.param("0", None, False, id="force-color-zero"),
        pytest.param("", None, False, id="force-color-empty"),
        pytest.param("false", None, False, id="force-color-false"),
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
        pytest.param(None, "1", True, False, id="no-color-on-a-tty"),
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
        pytest.param("http://localhost:4318", "http://localhost:4318", id="plain"),
        pytest.param("  http://localhost:4318\n", "http://localhost:4318", id="padded"),
        pytest.param("", None, id="empty"),
        pytest.param("  \t", None, id="blank"),
        pytest.param(None, None, id="unset"),
    ],
)
def test_otlp_endpoint_when_given_a_raw_value_does_trim_it_or_report_none(
    value: str | None, expected: str | None
):
    assert otlp_endpoint(value) == expected
