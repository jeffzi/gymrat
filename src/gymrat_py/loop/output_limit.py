"""Cap loop output to a fixed byte budget without splitting characters."""

import codecs

# The budget is measured in bytes, not characters: downstream consumers size
# their buffers in bytes, so a multi-byte-heavy string that "looks short" can
# still blow past the limit. Keep this module-private, as in the reference.
_OUTPUT_LIMIT_BYTES = 8192

_NEWLINE_BYTE = 0x0A


def limit_output(text: str) -> str:
    """Return at most ``_OUTPUT_LIMIT_BYTES`` bytes of ``text`` (UTF-8).

    When ``text`` fits the budget it is returned unchanged. When it overruns,
    the cut prefers a whole-line boundary: the last newline inside the first
    ``_OUTPUT_LIMIT_BYTES`` bytes, with its trailing newline dropped.

    When no usable newline exists (a single long line, or the only newline at
    byte 0), the cut falls back to the last whole character. An incremental
    UTF-8 decoder with ``final=False`` holds back the bytes of a character the
    cut split, so a multi-byte character is never severed and no U+FFFD
    replacement character is emitted.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= _OUTPUT_LIMIT_BYTES:
        return text

    head = encoded[:_OUTPUT_LIMIT_BYTES]
    last_newline = head.rfind(_NEWLINE_BYTE)
    # Require the newline past byte 0 so a leading-newline single line still
    # relays its content instead of collapsing to an empty string.
    if last_newline > 0:
        return head[:last_newline].decode("utf-8")

    decoder = codecs.getincrementaldecoder("utf-8")()
    return decoder.decode(head, final=False)
