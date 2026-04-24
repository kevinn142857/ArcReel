"""Error formatting helpers shared across async workers and persistence layers."""

from __future__ import annotations


def format_exception_message(exc: BaseException) -> str:
    """Return a readable error message even when ``str(exc)`` is empty."""
    message = str(exc).strip()
    if message:
        return message

    fallback = repr(exc).strip()
    if fallback:
        return fallback

    return exc.__class__.__name__
