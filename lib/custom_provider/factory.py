"""自定义供应商 Backend 工厂。

根据 CustomProvider 配置创建包装后的 Backend 实例。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lib.config.url_utils import ensure_google_base_url, ensure_openai_base_url
from lib.custom_provider.backends import CustomImageBackend, CustomTextBackend, CustomVideoBackend
from lib.image_backends.gemini import GeminiImageBackend
from lib.image_backends.jimeng import JimengImageBackend
from lib.image_backends.openai import OpenAIImageBackend
from lib.text_backends.gemini import GeminiTextBackend
from lib.text_backends.openai import OpenAITextBackend
from lib.video_backends.gemini import GeminiVideoBackend
from lib.video_backends.grok_rest import GrokRestVideoBackend
from lib.video_backends.jimeng import JimengVideoBackend
from lib.video_backends.newapi import NewAPIVideoBackend
from lib.video_backends.openai import OpenAIVideoBackend

if TYPE_CHECKING:
    from lib.db.models.custom_provider import CustomProvider

_VALID_MEDIA_TYPES = {"text", "image", "video"}
_VALID_API_FORMATS = {"openai", "google", "grok", "grok2api", "newapi"}
_GROK_VIDEO_MODEL_PREFIXES = ("grok-imagine-video",)
_JIMENG_IMAGE_MODEL_PREFIX = "jimeng-"
_JIMENG_VIDEO_MODEL_PREFIXES = ("jimeng-video-", "seedance-")


def create_custom_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    media_type: str,
) -> CustomTextBackend | CustomImageBackend | CustomVideoBackend:
    """根据自定义供应商配置创建包装后的 Backend 实例。

    Args:
        provider: 自定义供应商 ORM 对象
        model_id: 要使用的模型 ID
        media_type: 媒体类型 ("text" | "image" | "video")

    Returns:
        包装后的 Custom*Backend 实例

    Raises:
        ValueError: api_format 或 media_type 不合法
    """
    api_format = provider.api_format
    if api_format not in _VALID_API_FORMATS:
        raise ValueError(f"不支持的 api_format: {api_format!r}，支持: {_VALID_API_FORMATS}")
    if media_type not in _VALID_MEDIA_TYPES:
        raise ValueError(f"不支持的 media_type: {media_type!r}，支持: {_VALID_MEDIA_TYPES}")

    if api_format == "openai":
        return _create_openai_backend(provider=provider, model_id=model_id, media_type=media_type)
    elif api_format == "google":
        return _create_google_backend(provider=provider, model_id=model_id, media_type=media_type)
    elif api_format == "grok":
        return _create_grok_backend(provider=provider, model_id=model_id, media_type=media_type)
    elif api_format == "grok2api":
        return _create_grok2api_backend(provider=provider, model_id=model_id, media_type=media_type)
    else:  # newapi
        return _create_newapi_backend(provider=provider, model_id=model_id, media_type=media_type)


def _create_openai_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    media_type: str,
) -> CustomTextBackend | CustomImageBackend | CustomVideoBackend:
    """创建 OpenAI 格式的后端。"""
    pid = provider.provider_id
    base_url = ensure_openai_base_url(provider.base_url)
    if media_type == "text":
        delegate = OpenAITextBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomTextBackend(provider_id=pid, delegate=delegate, model=model_id)
    elif media_type == "image":
        if _should_use_jimeng_image_backend(model_id):
            return _create_jimeng_image_backend(
                provider=provider, model_id=model_id, provider_id=pid, base_url=base_url
            )
        delegate = OpenAIImageBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomImageBackend(provider_id=pid, delegate=delegate, model=model_id)
    else:  # video
        if _should_use_jimeng_video_backend(model_id):
            return _create_jimeng_video_backend(
                provider=provider, model_id=model_id, provider_id=pid, base_url=base_url
            )
        if _should_use_grok_video_backend(model_id):
            delegate = GrokRestVideoBackend(api_key=provider.api_key, base_url=provider.base_url, model=model_id)
            return CustomVideoBackend(provider_id=pid, delegate=delegate, model=model_id)
        delegate = OpenAIVideoBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomVideoBackend(provider_id=pid, delegate=delegate, model=model_id)


def _should_use_grok_video_backend(model_id: str) -> bool:
    """识别需要走 xAI 原生视频后端的 Grok 视频模型。"""
    normalized = model_id.strip().lower()
    return normalized.startswith(_GROK_VIDEO_MODEL_PREFIXES)


def _should_use_jimeng_image_backend(model_id: str) -> bool:
    """识别需要走 Jimeng 图片后端的模型。"""
    normalized = model_id.strip().lower()
    return normalized.startswith(_JIMENG_IMAGE_MODEL_PREFIX) and not normalized.startswith("jimeng-video-")


def _should_use_jimeng_video_backend(model_id: str) -> bool:
    """识别需要走 Jimeng 视频后端的模型。"""
    normalized = model_id.strip().lower()
    return normalized.startswith(_JIMENG_VIDEO_MODEL_PREFIXES)


def _create_jimeng_image_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    provider_id: str,
    base_url: str | None,
) -> CustomImageBackend:
    delegate = JimengImageBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
    return CustomImageBackend(provider_id=provider_id, delegate=delegate, model=model_id)


def _create_jimeng_video_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    provider_id: str,
    base_url: str | None,
) -> CustomVideoBackend:
    delegate = JimengVideoBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
    return CustomVideoBackend(provider_id=provider_id, delegate=delegate, model=model_id)


def _create_grok_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    media_type: str,
) -> CustomTextBackend | CustomImageBackend | CustomVideoBackend:
    """创建 Grok 格式的后端（自定义供应商统一走 xAI REST/OpenAI 兼容接口）。"""
    pid = provider.provider_id
    configured_base_url = ensure_openai_base_url(provider.base_url)
    base_url = configured_base_url or "https://api.x.ai/v1"
    if media_type == "text":
        delegate = OpenAITextBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomTextBackend(provider_id=pid, delegate=delegate, model=model_id)
    elif media_type == "image":
        if _should_use_jimeng_image_backend(model_id):
            return _create_jimeng_image_backend(
                provider=provider,
                model_id=model_id,
                provider_id=pid,
                base_url=configured_base_url,
            )
        delegate = OpenAIImageBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomImageBackend(provider_id=pid, delegate=delegate, model=model_id)
    else:  # video
        if _should_use_jimeng_video_backend(model_id):
            return _create_jimeng_video_backend(
                provider=provider,
                model_id=model_id,
                provider_id=pid,
                base_url=configured_base_url,
            )
        delegate = GrokRestVideoBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomVideoBackend(provider_id=pid, delegate=delegate, model=model_id)


def _create_google_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    media_type: str,
) -> CustomTextBackend | CustomImageBackend | CustomVideoBackend:
    """创建 Google 格式的后端。"""
    base_url = ensure_google_base_url(provider.base_url) or None
    pid = provider.provider_id
    if media_type == "text":
        delegate = GeminiTextBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomTextBackend(provider_id=pid, delegate=delegate, model=model_id)
    elif media_type == "image":
        delegate = GeminiImageBackend(api_key=provider.api_key, base_url=base_url, image_model=model_id)
        return CustomImageBackend(provider_id=pid, delegate=delegate, model=model_id)
    else:  # video
        delegate = GeminiVideoBackend(api_key=provider.api_key, base_url=base_url, video_model=model_id)
        return CustomVideoBackend(provider_id=pid, delegate=delegate, model=model_id)


def _create_grok2api_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    media_type: str,
) -> CustomTextBackend | CustomImageBackend | CustomVideoBackend:
    """创建 Grok2API 格式的后端。

    该模式面向 `grok2api` 一类 OpenAI 兼容网关：
    - 文本/图片沿用 OpenAI 兼容接口
    - 视频优先走 `/v1/videos`
    """
    pid = provider.provider_id
    base_url = ensure_openai_base_url(provider.base_url)
    if not base_url:
        raise ValueError("grok2api 需要 base_url")

    if media_type == "text":
        delegate = OpenAITextBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomTextBackend(provider_id=pid, delegate=delegate, model=model_id)
    elif media_type == "image":
        delegate = OpenAIImageBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomImageBackend(provider_id=pid, delegate=delegate, model=model_id)
    else:  # video
        delegate = GrokRestVideoBackend(
            api_key=provider.api_key,
            base_url=base_url,
            model=model_id,
            prefer_proxy_endpoint=True,
        )
        return CustomVideoBackend(provider_id=pid, delegate=delegate, model=model_id)


def _create_newapi_backend(
    *,
    provider: CustomProvider,
    model_id: str,
    media_type: str,
) -> CustomTextBackend | CustomImageBackend | CustomVideoBackend:
    """创建 NewAPI 格式的后端：文本/图片复用 OpenAI delegate，视频走 NewAPIVideoBackend。"""
    pid = provider.provider_id
    base_url = ensure_openai_base_url(provider.base_url)
    if media_type == "text":
        delegate = OpenAITextBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomTextBackend(provider_id=pid, delegate=delegate, model=model_id)
    elif media_type == "image":
        if _should_use_jimeng_image_backend(model_id):
            return _create_jimeng_image_backend(
                provider=provider, model_id=model_id, provider_id=pid, base_url=base_url
            )
        delegate = OpenAIImageBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomImageBackend(provider_id=pid, delegate=delegate, model=model_id)
    else:  # video
        if _should_use_jimeng_video_backend(model_id):
            return _create_jimeng_video_backend(
                provider=provider, model_id=model_id, provider_id=pid, base_url=base_url
            )
        delegate = NewAPIVideoBackend(api_key=provider.api_key, base_url=base_url, model=model_id)
        return CustomVideoBackend(provider_id=pid, delegate=delegate, model=model_id)
