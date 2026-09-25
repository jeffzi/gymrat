"""Cap loop output to a fixed byte budget without splitting characters."""

# The budget is measured in bytes, not characters: downstream consumers size
# their buffers in bytes, so a multi-byte-heavy string that "looks short" can
# still blow past the limit. Keep this module-private: the byte budget is an
# internal knob, not something a caller should tune.
_OUTPUT_LIMIT_BYTES = 8192


def limit_output(text: str) -> str:
    """Return at most ``_OUTPUT_LIMIT_BYTES`` bytes of ``text`` (UTF-8).

    Decoding the cut with ``errors="ignore"`` drops the trailing bytes of a
    character the cut split, so a multi-byte character is never severed and no
    U+FFFD replacement character is emitted.

    Args:
        text: The text to cap.

    Returns:
        The original text when it fits the budget. Otherwise the prefix up to
        the last newline inside the first ``_OUTPUT_LIMIT_BYTES`` bytes, with
        that newline dropped; when no usable newline exists (a single long
        line, or the only newline at byte 0), the prefix up to the last whole
        character.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= _OUTPUT_LIMIT_BYTES:
        return text

    head = encoded[:_OUTPUT_LIMIT_BYTES]
    last_newline = head.rfind(b"\n")
    # Require the newline past byte 0 so a leading-newline single line still
    # relays its content instead of collapsing to an empty string.
    if last_newline > 0:
        return head[:last_newline].decode("utf-8")

    return head.decode("utf-8", errors="ignore")
