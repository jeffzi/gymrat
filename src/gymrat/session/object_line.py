"""Decode one raw log line into a JSON object, or nothing."""

from gymrat.session.records import decode_log_line


def decode_object_line(raw: bytes) -> dict[str, object] | None:
    """Decode one raw UTF-8 log line as a JSON object.

    Args:
        raw: The line's bytes, without its trailing newline.

    Returns:
        The decoded object, or ``None`` when the line is not valid UTF-8, is not
        strict JSON (non-finite numbers included), or decodes to something other
        than an object.
    """
    try:
        # UnicodeDecodeError is a ValueError, so a non-UTF-8 line lands here too.
        parsed = decode_log_line(raw.decode("utf-8"))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
