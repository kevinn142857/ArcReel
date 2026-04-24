"""JimengImageBackend — Jimeng 兼容图片生成后端。"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from lib.config.url_utils import ensure_openai_base_url
from lib.image_backends.base import (
    ImageCapability,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from lib.providers import PROVIDER_JIMENG
from lib.retry import (
    BASE_RETRYABLE_ERRORS,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DOWNLOAD_BACKOFF_SECONDS,
    DOWNLOAD_MAX_ATTEMPTS,
    with_retry_async,
)
from lib.video_backends.base import IMAGE_MIME_TYPES

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "jimeng-4.6"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RESPONSE_FORMAT = "url"
_MAX_REFERENCE_IMAGES = 10

_JIMENG_RETRYABLE_ERRORS = BASE_RETRYABLE_ERRORS + (httpx.RequestError, httpx.HTTPStatusError)

_RESOLUTION_MAP: dict[str, str] = {
    "512PX": "1k",
    "1K": "1k",
    "2K": "2k",
    "4K": "4k",
}

_SUPPORTED_RATIOS = frozenset({"1:1", "3:4", "4:3", "9:16", "16:9"})


def _resolve_base_url(base_url: str | None) -> str:
    return (
        ensure_openai_base_url(base_url or os.getenv("JIMENG_BASE_URL") or DEFAULT_BASE_URL) or f"{DEFAULT_BASE_URL}/v1"
    )


def _resolve_ratio(aspect_ratio: str) -> str:
    if aspect_ratio in _SUPPORTED_RATIOS:
        return aspect_ratio
    logger.warning("JimengImageBackend 未知 aspect_ratio=%s，回退到 9:16", aspect_ratio)
    return "9:16"


def _resolve_resolution(image_size: str) -> str:
    return _RESOLUTION_MAP.get((image_size or "").upper(), "1k")


@with_retry_async(
    max_attempts=DOWNLOAD_MAX_ATTEMPTS,
    backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
    retryable_errors=_JIMENG_RETRYABLE_ERRORS,
)
async def download_image(url: str, output_path: Path, *, timeout: int = 120) -> None:
    """下载 Jimeng 返回的图片 URL 到本地。"""
    await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()
        await asyncio.to_thread(output_path.write_bytes, response.content)


class JimengImageBackend:
    """Jimeng OpenAI 兼容图片后端，支持 T2I / I2I。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        http_timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("JimengImageBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = _resolve_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._capabilities: set[ImageCapability] = {
            ImageCapability.TEXT_TO_IMAGE,
            ImageCapability.IMAGE_TO_IMAGE,
        }

    @property
    def name(self) -> str:
        return PROVIDER_JIMENG

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[ImageCapability]:
        return self._capabilities

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if request.reference_images:
            refs = self._collect_reference_images(request)
            if refs:
                body = await self._post_multipart(request, refs)
                return await self._save_result(body, request)
            logger.warning("Jimeng 所有参考图均无效，回退为文生图")
        body = await self._post_json(request)
        return await self._save_result(body, request)

    def _build_payload(self, request: ImageGenerationRequest) -> dict[str, Any]:
        return {
            "model": self._model,
            "prompt": request.prompt,
            "ratio": _resolve_ratio(request.aspect_ratio),
            "resolution": _resolve_resolution(request.image_size),
            "response_format": DEFAULT_RESPONSE_FORMAT,
        }

    def _collect_reference_images(self, request: ImageGenerationRequest) -> list[Path]:
        refs: list[Path] = []
        for ref in request.reference_images:
            path = Path(ref.path)
            if not path.exists():
                logger.warning("Jimeng 参考图不存在，已忽略: %s", path)
                continue
            refs.append(path)
        if len(refs) > _MAX_REFERENCE_IMAGES:
            logger.warning("Jimeng 参考图数量 %d 超过上限 %d，已截断", len(refs), _MAX_REFERENCE_IMAGES)
            refs = refs[:_MAX_REFERENCE_IMAGES]
        return refs

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retryable_errors=_JIMENG_RETRYABLE_ERRORS,
    )
    async def _post_json(self, request: ImageGenerationRequest) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            response = await client.post(
                f"{self._base_url}/images/generations",
                json=self._build_payload(request),
                headers=self._json_headers(),
            )
            response.raise_for_status()
            return response.json()

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retryable_errors=_JIMENG_RETRYABLE_ERRORS,
    )
    async def _post_multipart(self, request: ImageGenerationRequest, refs: list[Path]) -> dict[str, Any]:
        files = [("images", _make_upload_tuple(path)) for path in refs]
        data = {key: str(value) for key, value in self._build_payload(request).items()}
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            response = await client.post(
                f"{self._base_url}/images/generations",
                data=data,
                files=files,
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def _save_result(self, body: dict[str, Any], request: ImageGenerationRequest) -> ImageGenerationResult:
        item = _extract_first_item(body)
        image_url = item.get("url")
        if isinstance(image_url, str) and image_url:
            await download_image(image_url, request.output_path)
            return ImageGenerationResult(
                image_path=request.output_path,
                provider=PROVIDER_JIMENG,
                model=self._model,
                image_uri=image_url,
                quality=_resolve_resolution(request.image_size),
            )

        b64_json = item.get("b64_json")
        if isinstance(b64_json, str) and b64_json:
            image_bytes = base64.b64decode(b64_json)
            await asyncio.to_thread(request.output_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(request.output_path.write_bytes, image_bytes)
            return ImageGenerationResult(
                image_path=request.output_path,
                provider=PROVIDER_JIMENG,
                model=self._model,
                quality=_resolve_resolution(request.image_size),
            )

        raise RuntimeError(f"Jimeng 图片生成返回体缺少 url/b64_json: {body}")

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _json_headers(self) -> dict[str, str]:
        return {**self._auth_headers(), "Content-Type": "application/json"}


def _make_upload_tuple(path: Path) -> tuple[str, bytes, str]:
    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return path.name, path.read_bytes(), mime_type


def _extract_first_item(body: dict[str, Any]) -> dict[str, Any]:
    data = body.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError(f"Jimeng 图片生成返回体格式不正确: {body}")
    return data[0]
