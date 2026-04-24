from __future__ import annotations

import httpx

from lib.error_utils import format_exception_message


def test_format_exception_message_uses_repr_when_str_is_empty():
    assert format_exception_message(httpx.ReadTimeout("")) == "ReadTimeout('')"


def test_format_exception_message_prefers_normal_error_text():
    assert format_exception_message(RuntimeError("boom")) == "boom"
