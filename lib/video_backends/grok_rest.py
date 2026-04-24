"""GrokRestVideoBackend — 基于 xAI REST API 的视频生成后端。

仅用于自定义供应商场景，支持通过 HTTP Base URL 访问官方 xAI REST API
或兼容其协议的自定义网关。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from lib.config.url_utils import ensure_openai_base_url
from lib.providers import PROVIDER_GROK
from lib.retry import (
    BASE_RETRYABLE_ERRORS,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DOWNLOAD_BACKOFF_SECONDS,
    DOWNLOAD_MAX_ATTEMPTS,
    with_retry_async,
)
from lib.video_backends.base import (
    IMAGE_MIME_TYPES,
    VideoCapabilities,
    VideoCapability,
    VideoGenerationRequest,
    VideoGenerationResult,
    download_video,
    poll_with_retry,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "grok-imagine-video"
DEFAULT_BASE_URL = "https://api.x.ai/v1"

_POLL_INTERVAL_SECONDS = 5.0
_MIN_POLL_TIMEOUT_SECONDS = 600
_POLL_TIMEOUT_PER_SECOND = 30
_GROK_REST_RETRYABLE_ERRORS = BASE_RETRYABLE_ERRORS + (httpx.RequestError, httpx.HTTPStatusError)
_CREATE_FALLBACK_STATUS_CODES = {404, 405, 415}
_CONTENT_FALLBACK_STATUS_CODES = {404, 405}
_SIZE_MAP: dict[tuple[str, str], str] = {
    ("720p", "9:16"): "720x1280",
    ("720p", "16:9"): "1280x720",
    ("1080p", "9:16"): "1080x1920",
    ("1080p", "16:9"): "1920x1080",
    ("1024p", "9:16"): "1024x1792",
    ("1024p", "16:9"): "1792x1024",
}
_DEFAULT_SIZE = "720x1280"


class GrokRestVideoBackend:
    """通过 xAI REST API 生成视频。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
        prefer_proxy_endpoint: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("GrokRestVideoBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = ensure_openai_base_url(base_url) or DEFAULT_BASE_URL
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._prefer_proxy_endpoint = prefer_proxy_endpoint
        self._capabilities: set[VideoCapability] = {
            VideoCapability.TEXT_TO_VIDEO,
            VideoCapability.IMAGE_TO_VIDEO,
        }

    @property
    def name(self) -> str:
        return PROVIDER_GROK

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[VideoCapability]:
        return self._capabilities

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return VideoCapabilities(reference_images=True, max_reference_images=7)

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        payload = self._build_payload(request)

        logger.info("Grok REST 视频生成开始: model=%s, duration=%s", self._model, request.duration_seconds)

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            created, used_proxy_endpoint = await self._create_task(client, request, payload)
            task_id = _extract_task_id(created)
            logger.info("Grok REST 任务创建: task_id=%s", task_id or "<sync>")

            final = created
            if _extract_video_url(created) is None:
                if not task_id:
                    raise RuntimeError(f"Grok REST 创建任务返回体缺少任务 ID: {created}")
                final = await poll_with_retry(
                    poll_fn=lambda: self._poll_once(client, task_id),
                    is_done=_is_done,
                    is_failed=_extract_failure,
                    poll_interval=_POLL_INTERVAL_SECONDS,
                    max_wait=self._max_wait(request.duration_seconds),
                    retryable_errors=_GROK_REST_RETRYABLE_ERRORS,
                    label="Grok REST",
                )

            video_url = _extract_video_url(final)
            download_uri = video_url
            if used_proxy_endpoint and task_id:
                try:
                    await self._download_proxy_content_with_retry(client, task_id, request.output_path)
                    download_uri = self._content_endpoint(task_id)
                except httpx.HTTPStatusError as exc:
                    if exc.response is None or exc.response.status_code not in _CONTENT_FALLBACK_STATUS_CODES:
                        raise
                    if not video_url:
                        raise RuntimeError(f"Grok REST 视频任务完成但缺少可下载 URL: {final}") from exc
                    await self._download_with_retry(video_url, request.output_path)
            else:
                if not video_url:
                    raise RuntimeError(f"Grok REST 视频任务完成但缺少可下载 URL: {final}")
                await self._download_with_retry(video_url, request.output_path)

        raw_duration = _extract_first(final, ("duration",), ("video", "duration"), ("metadata", "duration"))
        duration_seconds = int(float(raw_duration)) if raw_duration is not None else request.duration_seconds

        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_GROK,
            model=self._model,
            duration_seconds=duration_seconds,
            task_id=task_id,
            video_uri=download_uri,
            generate_audio=True,
        )

    def _build_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": request.prompt,
            "duration": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
        }

        reference_images = [Path(p) for p in (request.reference_images or []) if Path(p).exists()]
        if reference_images:
            if request.start_image:
                logger.warning("Grok REST 不支持同时传 start_image 与 reference_images，已优先使用 reference_images")
            payload["reference_images"] = [{"url": _image_to_data_uri(p)} for p in reference_images]
            return payload

        if request.start_image and Path(request.start_image).exists():
            payload["image"] = {"url": _image_to_data_uri(request.start_image)}
        elif request.start_image:
            logger.warning("Grok REST start_image 文件不存在，已忽略: %s", request.start_image)

        return payload

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retryable_errors=_GROK_REST_RETRYABLE_ERRORS,
    )
    async def _create_task(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if self._prefer_proxy_endpoint:
            try:
                return await self._create_task_via_proxy(client, request), True
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code not in _CREATE_FALLBACK_STATUS_CODES:
                    raise
                logger.info(
                    "Grok 代理兼容接口返回 %s，回退到官方视频创建接口 /videos/generations",
                    exc.response.status_code,
                )

        resp = await client.post(f"{self._base_url}/videos/generations", json=payload, headers=self._json_headers())
        if resp.status_code in _CREATE_FALLBACK_STATUS_CODES:
            logger.info("Grok REST 官方视频创建接口返回 %s，回退到代理兼容接口 /videos", resp.status_code)
            return await self._create_task_via_proxy(client, request), True
        resp.raise_for_status()
        return resp.json(), False

    async def _create_task_via_proxy(
        self, client: httpx.AsyncClient, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        data = {
            "model": self._model,
            "prompt": request.prompt,
            "seconds": str(request.duration_seconds),
            "size": _resolve_size(request.resolution, request.aspect_ratio),
            "resolution_name": request.resolution,
            "preset": "normal",
        }
        files = _build_proxy_reference_files(request)
        resp = await client.post(
            f"{self._base_url}/videos",
            data=data,
            files=files or None,
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def _poll_once(self, client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
        resp = await client.get(f"{self._base_url}/videos/{task_id}", headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json()

    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retryable_errors=_GROK_REST_RETRYABLE_ERRORS,
    )
    async def _download_proxy_content_with_retry(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        output_path: Path,
    ) -> None:
        await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
        async with client.stream("GET", self._content_endpoint(task_id), headers=self._auth_headers()) as resp:
            if resp.status_code >= 400:
                await resp.aread()
            resp.raise_for_status()
            chunks: list[bytes] = []
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                chunks.append(chunk)

        def _write_all() -> None:
            with open(output_path, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)

        await asyncio.to_thread(_write_all)

    @staticmethod
    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retryable_errors=_GROK_REST_RETRYABLE_ERRORS,
    )
    async def _download_with_retry(video_url: str, output_path: Path) -> None:
        await download_video(video_url, output_path)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _json_headers(self) -> dict[str, str]:
        return {**self._auth_headers(), "Content-Type": "application/json"}

    def _content_endpoint(self, task_id: str) -> str:
        return f"{self._base_url}/videos/{task_id}/content"

    @staticmethod
    def _max_wait(duration_seconds: int) -> float:
        return max(_MIN_POLL_TIMEOUT_SECONDS, duration_seconds * _POLL_TIMEOUT_PER_SECOND)


def _is_done(state: dict[str, Any]) -> bool:
    status = str(state.get("status", "")).lower()
    return status in {"completed", "done", "succeeded", "success"}


def _extract_failure(state: dict[str, Any]) -> str | None:
    status = str(state.get("status", "")).lower()
    if status not in {"failed", "error", "cancelled", "canceled"}:
        return None
    error = _extract_first(
        state,
        ("error", "message"),
        ("error",),
        ("message",),
        ("detail",),
    )
    return f"Grok REST 视频生成失败: {error or 'unknown'}"


def _extract_video_url(state: dict[str, Any]) -> str | None:
    candidates = (
        ("url",),
        ("video_url",),
        ("video", "url"),
        ("result", "url"),
        ("data", 0, "url"),
        ("results", 0, "url"),
        ("output", 0, "url"),
    )
    value = _extract_first(state, *candidates)
    return value if isinstance(value, str) and value else None


def _extract_task_id(state: dict[str, Any]) -> str | None:
    value = _extract_first(state, ("id",), ("request_id",), ("task_id",), ("video_id",))
    return value if isinstance(value, str) and value else None


def _extract_first(state: dict[str, Any], *paths: tuple[Any, ...]) -> Any | None:
    for path in paths:
        current: Any = state
        for part in path:
            if isinstance(part, int):
                if not isinstance(current, list) or part >= len(current):
                    current = None
                    break
                current = current[part]
                continue
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def _image_to_data_uri(path: str | Path) -> str:
    from lib.image_backends.base import image_to_base64_data_uri

    return image_to_base64_data_uri(Path(path))


def _resolve_size(resolution: str, aspect_ratio: str) -> str:
    return _SIZE_MAP.get((resolution, aspect_ratio), _DEFAULT_SIZE)


def _build_proxy_reference_files(
    request: VideoGenerationRequest,
) -> list[tuple[str, tuple[str, bytes, str]]]:
    reference_paths = [Path(p) for p in (request.reference_images or []) if Path(p).exists()]
    if reference_paths:
        if request.start_image:
            logger.warning("Grok 代理接口不支持同时传 start_image 与 reference_images，已优先使用 reference_images")
        return [("input_reference[]", _make_upload_tuple(path)) for path in reference_paths]

    if request.start_image and Path(request.start_image).exists():
        return [("input_reference[]", _make_upload_tuple(request.start_image))]

    if request.start_image:
        logger.warning("Grok 代理接口 start_image 文件不存在，已忽略: %s", request.start_image)

    return []


def _make_upload_tuple(path: str | Path) -> tuple[str, bytes, str]:
    image_path = Path(path)
    mime_type = IMAGE_MIME_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
    return image_path.name, image_path.read_bytes(), mime_type
