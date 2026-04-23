"""OpenAITextBackend — OpenAI 文本生成后端。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from lib.openai_shared import OPENAI_RETRYABLE_ERRORS, create_openai_client
from lib.providers import PROVIDER_OPENAI
from lib.retry import with_retry_async
from lib.text_backends.base import (
    TextCapability,
    TextGenerationRequest,
    TextGenerationResult,
    resolve_schema,
    warn_if_truncated,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAITextBackend:
    """OpenAI 文本生成后端，支持 Chat Completions API。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        # 禁用 SDK 内置重试，由本层 generate() 统一管理重试策略
        self._client = create_openai_client(api_key=api_key, base_url=base_url, max_retries=0)
        self._model = model or DEFAULT_MODEL
        self._capabilities: set[TextCapability] = {
            TextCapability.TEXT_GENERATION,
            TextCapability.STRUCTURED_OUTPUT,
            TextCapability.VISION,
        }

    @property
    def name(self) -> str:
        return PROVIDER_OPENAI

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[TextCapability]:
        return self._capabilities

    @with_retry_async(max_attempts=4, backoff_seconds=(2, 4, 8), retryable_errors=OPENAI_RETRYABLE_ERRORS)
    async def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        """生成文本回复。

        单一重试循环包裹整个流程：
        1. 尝试原生 response_format 调用
        2. 若遇 schema 不兼容错误 → 本次 attempt 内降级到 Instructor
        3. 若遇瞬态错误（429/500/503/网络）→ 由装饰器自动重试整个流程

        这样无论是原生调用还是降级路径遇到瞬态错误，都统一由外层重试处理。
        """
        messages = _build_messages(request)
        kwargs: dict = {"model": self._model, "messages": messages}
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens

        if request.response_schema:
            schema = resolve_schema(request.response_schema)
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            }

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if request.response_schema and _is_schema_error(exc):
                logger.warning(
                    "原生 response_format 失败 (%s)，降级到 Instructor 路径",
                    exc,
                )
                return await _instructor_fallback(self._client, self._model, request, messages)
            raise

        if request.response_schema and isinstance(response, str):
            error_message = _extract_sse_error_message(response)
            if error_message:
                raise RuntimeError(error_message)

        text, usage, finish_reason = _extract_text_usage_and_finish_reason(response)
        if request.response_schema:
            json_text = _coerce_json_text(text)
            if json_text is not None:
                text = json_text
            elif isinstance(response, str):
                logger.warning("原生 response_format 返回非 JSON 文本，降级到 prompt JSON 路径")
                return await _prompt_json_fallback(self._client, self._model, request, messages)
        output_tokens = _usage_value(usage, "completion_tokens")
        warn_if_truncated(
            finish_reason,
            provider=PROVIDER_OPENAI,
            model=self._model,
            output_tokens=output_tokens,
        )
        return TextGenerationResult(
            text=text,
            provider=PROVIDER_OPENAI,
            model=self._model,
            input_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=output_tokens,
        )


def _build_messages(request: TextGenerationRequest) -> list[dict]:
    """将 TextGenerationRequest 转为 OpenAI messages 格式。"""
    messages: list[dict] = []

    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})

    # 构建 user message
    if request.images:
        from lib.image_backends.base import image_to_base64_data_uri

        content: list[dict] = []
        for img in request.images:
            if img.path:
                data_uri = image_to_base64_data_uri(img.path)
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif img.url:
                content.append({"type": "image_url", "image_url": {"url": img.url}})
        content.append({"type": "text", "text": request.prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": request.prompt})

    return messages


_SCHEMA_ERROR_KEYWORDS = (
    "response_schema",
    "json_schema",
    "Unknown name",
    "Cannot find field",
    "Invalid JSON payload",
)


def _is_schema_error(exc: BaseException) -> bool:
    """判断异常是否为 JSON Schema 不兼容导致的错误。

    除了标准的 400 BadRequestError，一些 OpenAI 兼容代理（如 Gemini
    兼容端点）会将上游 schema 错误包装成其他状态码（如 429），
    因此也检查错误信息中是否包含 schema 相关关键字。
    """
    if isinstance(exc, BadRequestError):
        return True
    # 代理可能把上游 schema 错误包装成非 400 状态码
    error_str = str(exc)
    return any(kw in error_str for kw in _SCHEMA_ERROR_KEYWORDS)


def _extract_text_usage_and_finish_reason(response: Any) -> tuple[str, Any | None, str | None]:
    """兼容标准 ChatCompletion、字典响应、以及错误网关返回的 SSE 字符串。"""
    if isinstance(response, str):
        return _extract_from_string_response(response)

    if isinstance(response, dict):
        return _extract_from_mapping_response(response)

    choices = getattr(response, "choices", None) or []
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        text = _normalize_text_content(getattr(message, "content", ""))
        finish_reason = getattr(choice, "finish_reason", None)
        return text, getattr(response, "usage", None), finish_reason

    return str(response), getattr(response, "usage", None), None


def _extract_from_mapping_response(response: dict[str, Any]) -> tuple[str, Any | None, str | None]:
    choices = response.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        text = _normalize_text_content(message.get("content", ""))
        return text, response.get("usage"), choice.get("finish_reason")
    return str(response), response.get("usage"), None


def _extract_from_string_response(response: str) -> tuple[str, Any | None, str | None]:
    raw = response.strip()
    if not raw.startswith("data:"):
        return raw, None, None

    text_parts: list[str] = []
    usage: Any | None = None
    finish_reason: str | None = None

    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            text_parts.append(payload)
            continue

        if not isinstance(data, dict):
            continue

        if data.get("usage") is not None:
            usage = data.get("usage")

        choices = data.get("choices") or []
        if not choices:
            continue

        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        content = delta.get("content")
        if content in (None, ""):
            content = message.get("content", "")
        text_parts.append(_normalize_text_content(content))

    text = "".join(text_parts).strip()
    return text or raw, usage, finish_reason


def _extract_sse_error_message(response: str) -> str | None:
    event_name: str | None = None
    for line in response.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]" or event_name != "error":
            continue

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return payload

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = data.get("message")
            if isinstance(message, str) and message:
                return message
        return payload

    return None


_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _coerce_json_text(text: str) -> str | None:
    """尽量从普通文本中提取可供上层 schema 校验的 JSON 字符串。"""
    candidate = text.strip()
    if not candidate:
        return None

    match = _FENCED_JSON_RE.match(candidate)
    if match:
        candidate = match.group(1).strip()

    for probe in (candidate, _extract_json_substring(candidate)):
        if not probe:
            continue
        try:
            json.loads(probe)
        except json.JSONDecodeError:
            continue
        return probe
    return None


def _extract_json_substring(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _normalize_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _usage_value(usage: Any | None, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return value if isinstance(value, int) else None


async def _instructor_fallback(
    client: AsyncOpenAI,
    model: str,
    request: TextGenerationRequest,
    messages: list[dict],
) -> TextGenerationResult:
    """Instructor 降级：当原生 response_format 不可用时的备选路径。"""
    from lib.text_backends.instructor_support import instructor_fallback_async

    return await instructor_fallback_async(
        client=client,
        model=model,
        messages=messages,
        response_schema=request.response_schema,
        provider=PROVIDER_OPENAI,
        max_tokens=request.max_output_tokens,
    )


async def _prompt_json_fallback(
    client: AsyncOpenAI,
    model: str,
    request: TextGenerationRequest,
    messages: list[dict],
) -> TextGenerationResult:
    """兼容忽略 response_format 的网关：通过 prompt 注入强制 JSON。"""
    from lib.text_backends.base import TextGenerationResult
    from lib.text_backends.instructor_support import inject_json_instruction

    schema = resolve_schema(request.response_schema)
    schema_hint = (
        "Return ONLY valid JSON that matches this schema exactly. "
        "Do not add markdown fences or explanatory text.\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
    fb_messages = inject_json_instruction(messages)
    system_idx = next((i for i, m in enumerate(fb_messages) if m.get("role") == "system"), None)
    if system_idx is None:
        fb_messages.insert(0, {"role": "system", "content": schema_hint})
    else:
        fb_messages[system_idx] = {
            **fb_messages[system_idx],
            "content": ((fb_messages[system_idx].get("content") or "") + "\n" + schema_hint).strip(),
        }

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": fb_messages,
        "response_format": {"type": "json_object"},
    }
    if request.max_output_tokens is not None:
        create_kwargs["max_tokens"] = request.max_output_tokens

    response = await client.chat.completions.create(**create_kwargs)
    if isinstance(response, str):
        error_message = _extract_sse_error_message(response)
        if error_message:
            raise RuntimeError(error_message)

    text, usage, finish_reason = _extract_text_usage_and_finish_reason(response)
    json_text = _coerce_json_text(text)
    if json_text is None:
        raise ValueError("OpenAI-compatible endpoint returned non-JSON text during structured fallback")

    output_tokens = _usage_value(usage, "completion_tokens")
    warn_if_truncated(
        finish_reason,
        provider=PROVIDER_OPENAI,
        model=model,
        output_tokens=output_tokens,
    )
    return TextGenerationResult(
        text=json_text,
        provider=PROVIDER_OPENAI,
        model=model,
        input_tokens=_usage_value(usage, "prompt_tokens"),
        output_tokens=output_tokens,
    )
