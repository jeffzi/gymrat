"""Lock subpackage: single-flight repository lock via hard-link atomicity."""

from gymrat.session.lock.acquire import acquire_lock

__all__ = [
    "acquire_lock",
]
