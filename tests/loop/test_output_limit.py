"""Behavioral tests for the loop output byte-budget limiter."""

import pytest

from gymrat.loop.output_limit import limit_output

LIMIT_BYTES = 8192


# ---------------------------------------------------------------------------
# limit_output
# ---------------------------------------------------------------------------


def test_limit_output_when_within_budget_does_return_text_unchanged() -> None:
    text = "a short line\nand another\n"

    result = limit_output(text)

    assert result == text


def test_limit_output_when_multi_line_overrun_does_cut_to_last_whole_line() -> None:
    line = "a" * 100
    text = f"{line}\n" * 200
    whole_lines = LIMIT_BYTES // len(f"{line}\n".encode())

    result = limit_output(text)

    expected = (f"{line}\n" * whole_lines).removesuffix("\n")
    assert result == expected
    assert len(result.encode("utf-8")) <= LIMIT_BYTES


def test_limit_output_when_single_long_line_does_cut_to_last_whole_char() -> None:
    text = "a" * 9000

    result = limit_output(text)

    assert result == "a" * LIMIT_BYTES


def test_limit_output_when_only_newline_at_byte_zero_does_char_cut_not_empty() -> None:
    text = "\n" + "a" * 9000

    result = limit_output(text)

    assert result == "\n" + "a" * (LIMIT_BYTES - 1)
    assert result != ""
    assert len(result.encode("utf-8")) <= LIMIT_BYTES


def test_limit_output_when_cut_splits_multi_byte_char_does_not_emit_replacement() -> None:
    text = "é" * 9000  # each "é" is 2 bytes in UTF-8

    result = limit_output(text)

    assert len(result.encode("utf-8")) <= LIMIT_BYTES
    assert "�" not in result
    assert result == "é" * (LIMIT_BYTES // 2)


def test_limit_output_when_valid_utf8_within_budget_does_return_text_unchanged() -> None:
    text = "café résumé naïve €42"  # cspell:disable-line

    result = limit_output(text)

    assert result == text


@pytest.mark.parametrize(
    ("char", "prefix"),
    [
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
    assert "�" not in result
