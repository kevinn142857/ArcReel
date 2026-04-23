"""GrokRestVideoBackend — 基于 xAI REST API 的视频生成后端。

仅用于自定义供应商场景，支持通过 HTTP Base URL 访问官方 xAI REST API
或兼容其协议的自定义网关。
"""

from __future__ import annotations

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


class GrokRestVideoBackend:
    """通过 xAI REST API 生成视频。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("GrokRestVideoBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = ensure_openai_base_url(base_url) or DEFAULT_BASE_URL
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
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
            task_id = await self._create_task(client, payload)
            logger.info("Grok REST 任务创建: task_id=%s", task_id)

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
            video_uri=video_url,
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
    async def _create_task(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        resp = await client.post(f"{self._base_url}/videos/generations", json=payload, headers=self._headers())
        resp.raise_for_status()
        body = resp.json()
        task_id = _extract_first(body, ("id",), ("request_id",), ("task_id",))
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"Grok REST 创建任务返回体缺少任务 ID: {body}")
        return task_id

    async def _poll_once(self, client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
        resp = await client.get(f"{self._base_url}/videos/{task_id}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retryable_errors=_GROK_REST_RETRYABLE_ERRORS,
    )
    async def _download_with_retry(video_url: str, output_path: Path) -> None:
        await download_video(video_url, output_path)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

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
