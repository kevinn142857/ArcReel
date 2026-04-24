"""自定义供应商模型发现。

提供模型列表查询与 media_type 推断功能。
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx
from google import genai
from openai import OpenAI

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r"image|dall|img", re.IGNORECASE)
_VIDEO_PATTERN = re.compile(
    r"video|sora|kling|wan|seedance|cog|mochi|veo|pika|minimax|hailuo|seedream|jimeng|runway",
    re.IGNORECASE,
)

# Google generation method → media_type 映射
_GENERATION_METHOD_MAP: dict[str, str] = {
    "generateVideo": "video",
    "generateVideos": "video",
    "generateImages": "image",
    "generateImage": "image",
}

_GROK_FALLBACK_VIDEO_MODELS = ("grok-imagine-video",)


def infer_media_type(model_id: str) -> str:
    """根据模型 ID 关键字推断 media_type。

    Returns:
        "image" | "video" | "text"
    """
    if _IMAGE_PATTERN.search(model_id):
        return "image"
    if _VIDEO_PATTERN.search(model_id):
        return "video"
    return "text"


async def discover_models(api_format: str, base_url: str | None, api_key: str) -> list[dict]:
    """查询供应商的可用模型列表。

    Args:
        api_format: API 格式 ("openai" | "google" | "grok" | "grok2api" | "newapi")
        base_url: 供应商 API 基础 URL
        api_key: API 密钥

    Returns:
        模型列表，每项包含: model_id, display_name, media_type, is_default, is_enabled

    Raises:
        ValueError: api_format 不支持
    """
    if api_format in {"openai", "newapi"}:
        return await _discover_openai(base_url, api_key)
    elif api_format == "google":
        return await _discover_google(base_url, api_key)
    elif api_format == "grok":
        return await _discover_grok_rest(base_url, api_key)
    elif api_format == "grok2api":
        return await _discover_grok2api(base_url, api_key)
    else:
        raise ValueError(f"不支持的 api_format: {api_format!r}，支持: 'openai', 'google', 'grok', 'grok2api', 'newapi'")


async def _discover_openai(base_url: str | None, api_key: str) -> list[dict]:
    """通过 OpenAI 兼容 API 发现模型。"""

    def _sync():
        from lib.config.url_utils import ensure_openai_base_url

        client = OpenAI(api_key=api_key, base_url=ensure_openai_base_url(base_url))
        raw_models = client.models.list()
        models = sorted(raw_models, key=lambda m: m.id)
        return _build_result_list([(m.id, infer_media_type(m.id)) for m in models])

    return await asyncio.to_thread(_sync)


async def _discover_google(base_url: str | None, api_key: str) -> list[dict]:
    """通过 Google genai SDK 发现模型。"""

    def _sync():
        from lib.config.url_utils import ensure_google_base_url

        kwargs: dict = {"api_key": api_key}
        effective_url = ensure_google_base_url(base_url) if base_url else None
        if effective_url:
            kwargs["http_options"] = {"base_url": effective_url}
        client = genai.Client(**kwargs)

        raw_models = client.models.list()

        entries: list[tuple[str, str]] = []
        for m in raw_models:
            model_id = m.name
            if model_id.startswith("models/"):
                model_id = model_id[len("models/") :]
            media_type = _infer_from_generation_methods(m) or infer_media_type(model_id)
            entries.append((model_id, media_type))

        entries.sort(key=lambda e: e[0])
        return _build_result_list(entries)

    return await asyncio.to_thread(_sync)


async def _discover_grok_rest(base_url: str | None, api_key: str) -> list[dict]:
    """通过 xAI REST API 发现 Grok 模型。

    优先使用 xAI 的细分模型端点（language/image/video-generation-models），
    若自定义网关仅实现 OpenAI 兼容的 ``/v1/models``，则回退到 OpenAI 兼容发现。
    """

    from lib.config.url_utils import ensure_openai_base_url

    effective_base_url = ensure_openai_base_url(base_url) or "https://api.x.ai/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = 30.0

    async def _fetch_xai_models(endpoint: str, media_type: str) -> list[tuple[str, str]]:
        url = f"{effective_base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                raise httpx.HTTPStatusError("xAI endpoint not implemented", request=resp.request, response=resp)
            resp.raise_for_status()
            body = resp.json()

        models = body.get("models")
        if not isinstance(models, list):
            raise RuntimeError(f"xAI 模型发现返回缺少 models 数组: {body}")

        entries: list[tuple[str, str]] = []
        for model in models:
            model_id = model.get("id")
            if isinstance(model_id, str) and model_id:
                entries.append((model_id, media_type))
            aliases = model.get("aliases") or []
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias:
                        entries.append((alias, media_type))
        return entries

    try:
        language_entries, image_entries, video_entries = await asyncio.gather(
            _fetch_xai_models("language-models", "text"),
            _fetch_xai_models("image-generation-models", "image"),
            _fetch_xai_models("video-generation-models", "video"),
        )
        entries = language_entries + image_entries + video_entries
        if not entries:
            entries = [(name, "video") for name in _GROK_FALLBACK_VIDEO_MODELS]
        unique_entries = sorted({(model_id, media_type) for model_id, media_type in entries}, key=lambda e: e[0])
        return _build_result_list(unique_entries)
    except httpx.HTTPStatusError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
        logger.info("Grok 自定义网关未实现 xAI 专用模型端点，回退到 /v1/models 发现")
        return await _discover_openai(effective_base_url, api_key)


async def _discover_grok2api(base_url: str | None, api_key: str) -> list[dict]:
    """通过 OpenAI 兼容的 `/v1/models` 发现 Grok2API 网关模型。"""
    if not base_url or not base_url.strip():
        raise ValueError("grok2api 需要 base_url")
    return await _discover_openai(base_url, api_key)


def _infer_from_generation_methods(model) -> str | None:
    """从 Google model 的 supported_generation_methods 推断 media_type。

    Returns:
        推断出的 media_type，无法推断时返回 None
    """
    methods = getattr(model, "supported_generation_methods", None)
    if not methods:
        return None

    for method in methods:
        if method in _GENERATION_METHOD_MAP:
            return _GENERATION_METHOD_MAP[method]

    return None


def _build_result_list(entries: list[tuple[str, str]]) -> list[dict]:
    """将 (model_id, media_type) 列表转为结果字典列表，标记每种 media_type 的第一个为 default。"""
    seen_types: set[str] = set()
    result: list[dict] = []

    for model_id, media_type in entries:
        is_default = media_type not in seen_types
        seen_types.add(media_type)
        result.append(
            {
                "model_id": model_id,
                "display_name": model_id,
                "media_type": media_type,
                "is_default": is_default,
                "is_enabled": True,
            }
        )

    return result
