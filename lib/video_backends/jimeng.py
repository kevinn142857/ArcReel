"""JimengVideoBackend — Jimeng 兼容视频生成后端。"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

from lib.config.url_utils import ensure_openai_base_url
from lib.providers import PROVIDER_JIMENG
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

DEFAULT_MODEL = "jimeng-video-3.5-pro"
DEFAULT_BASE_URL = "http://localhost:8000"

_POLL_INTERVAL_SECONDS = 5.0
_MIN_POLL_TIMEOUT_SECONDS = 600
_POLL_TIMEOUT_PER_SECOND = 30
_MAX_STANDARD_INPUTS = 2
_MAX_SEEDANCE_INPUTS = 10

_JIMENG_RETRYABLE_ERRORS = BASE_RETRYABLE_ERRORS + (httpx.RequestError, httpx.HTTPStatusError)
_SUPPORTED_RATIOS = frozenset({"1:1", "3:4", "4:3", "9:16", "16:9"})


def _resolve_base_url(base_url: str | None) -> str:
    return (
        ensure_openai_base_url(base_url or os.getenv("JIMENG_BASE_URL") or DEFAULT_BASE_URL) or f"{DEFAULT_BASE_URL}/v1"
    )


def _resolve_ratio(aspect_ratio: str) -> str:
    if aspect_ratio in _SUPPORTED_RATIOS:
        return aspect_ratio
    logger.warning("JimengVideoBackend 未知 aspect_ratio=%s，回退到 9:16", aspect_ratio)
    return "9:16"


def _is_seedance_model(model: str) -> bool:
    return model.startswith("jimeng-video-seedance-") or model.startswith("seedance-")


def _is_international_token(api_key: str) -> bool:
    return bool(re.match(r"^[a-z]{2}-", api_key.strip().lower()))


class JimengVideoBackend:
    """Jimeng 兼容视频后端，统一走异步提交 + 轮询接口。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("JimengVideoBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = _resolve_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._capabilities: set[VideoCapability] = {
            VideoCapability.TEXT_TO_VIDEO,
            VideoCapability.IMAGE_TO_VIDEO,
        }

    @property
    def name(self) -> str:
        return PROVIDER_JIMENG

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[VideoCapability]:
        return self._capabilities

    @property
    def video_capabilities(self) -> VideoCapabilities:
        if _is_seedance_model(self._model):
            return VideoCapabilities(first_frame=True, last_frame=False, reference_images=True, max_reference_images=10)
        return VideoCapabilities(first_frame=True, last_frame=True, reference_images=False, max_reference_images=0)

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        endpoint = self._async_endpoint()
        submission = self._build_submission(request)

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            task_id = await self._create_task(client, endpoint, submission)
            logger.info("Jimeng 视频任务创建: model=%s task_id=%s", self._model, task_id)

            final = await poll_with_retry(
                poll_fn=lambda: self._poll_once(client, endpoint, task_id),
                is_done=_is_done,
                is_failed=_extract_failure,
                poll_interval=_POLL_INTERVAL_SECONDS,
                max_wait=self._max_wait(request.duration_seconds),
                retryable_errors=_JIMENG_RETRYABLE_ERRORS,
                label="Jimeng",
            )

        video_url = _extract_video_url(final)
        if not video_url:
            raise RuntimeError(f"Jimeng 视频任务完成但缺少可下载 URL: {final}")

        await self._download_with_retry(video_url, request.output_path)
        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_JIMENG,
            model=self._model,
            duration_seconds=request.duration_seconds,
            task_id=task_id,
            video_uri=video_url,
        )

    def _build_submission(self, request: VideoGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": request.prompt,
            "ratio": _resolve_ratio(request.aspect_ratio),
            "resolution": request.resolution,
            "duration": request.duration_seconds,
        }

        if _is_seedance_model(self._model):
            return self._build_seedance_submission(request, payload)
        return self._build_standard_submission(request, payload)

    def _build_standard_submission(self, request: VideoGenerationRequest, payload: dict[str, Any]) -> dict[str, Any]:
        files: list[tuple[str, tuple[str, bytes, str]]] = []

        if request.start_image:
            start_image = Path(request.start_image)
            if start_image.exists():
                files.append(("files", _make_upload_tuple(start_image)))
            else:
                logger.warning("Jimeng start_image 文件不存在，已忽略: %s", start_image)

        if request.end_image:
            end_image = Path(request.end_image)
            if end_image.exists():
                files.append(("files", _make_upload_tuple(end_image)))
            else:
                logger.warning("Jimeng end_image 文件不存在，已忽略: %s", end_image)

        if request.reference_images:
            logger.warning(
                "Jimeng 非 Seedance 模型不支持 reference_images，已忽略 %d 张", len(request.reference_images)
            )

        files = files[:_MAX_STANDARD_INPUTS]
        if files:
            return {
                "data": {key: str(value) for key, value in payload.items()},
                "files": files,
            }
        return {"json": payload}

    def _build_seedance_submission(self, request: VideoGenerationRequest, payload: dict[str, Any]) -> dict[str, Any]:
        if request.end_image:
            logger.warning("Jimeng Seedance 不支持尾帧 end_image，已忽略: %s", request.end_image)

        material_paths: list[Path] = []
        if request.start_image:
            start_image = Path(request.start_image)
            if start_image.exists():
                material_paths.append(start_image)
            else:
                logger.warning("Jimeng Seedance start_image 文件不存在，已忽略: %s", start_image)

        for ref in request.reference_images or []:
            ref_path = Path(ref)
            if not ref_path.exists():
                logger.warning("Jimeng Seedance 参考图不存在，已忽略: %s", ref_path)
                continue
            material_paths.append(ref_path)

        if len(material_paths) > _MAX_SEEDANCE_INPUTS:
            logger.warning("Jimeng Seedance 素材数量 %d 超过上限 %d，已截断", len(material_paths), _MAX_SEEDANCE_INPUTS)
            material_paths = material_paths[:_MAX_SEEDANCE_INPUTS]

        if self._uses_international_seedance():
            if not material_paths:
                raise ValueError("Jimeng 国际版 Seedance 兼容接口至少需要一张参考图或首帧图")
            files = []
            for idx, path in enumerate(material_paths):
                field_name = "image_file" if idx == 0 else f"image_file_{idx}"
                files.append((field_name, _make_upload_tuple(path)))
            return {
                "data": {key: str(value) for key, value in payload.items()},
                "files": files,
            }

        if material_paths:
            files = [("files", _make_upload_tuple(path)) for path in material_paths]
            return {
                "data": {key: str(value) for key, value in payload.items()},
                "files": files,
            }
        return {"json": payload}

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retryable_errors=_JIMENG_RETRYABLE_ERRORS,
    )
    async def _create_task(self, client: httpx.AsyncClient, endpoint: str, submission: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {"headers": self._auth_headers()}
        if "json" in submission:
            kwargs["json"] = submission["json"]
            kwargs["headers"] = self._json_headers()
        else:
            kwargs["data"] = submission["data"]
            kwargs["files"] = submission["files"]

        response = await client.post(endpoint, **kwargs)
        response.raise_for_status()
        body = response.json()
        task_id = body.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"Jimeng 创建任务返回体缺少 task_id: {body}")
        return task_id

    async def _poll_once(self, client: httpx.AsyncClient, endpoint: str, task_id: str) -> dict[str, Any]:
        response = await client.get(f"{endpoint}/{task_id}", headers=self._auth_headers())
        response.raise_for_status()
        return response.json()

    @staticmethod
    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retryable_errors=_JIMENG_RETRYABLE_ERRORS,
    )
    async def _download_with_retry(video_url: str, output_path: Path) -> None:
        await download_video(video_url, output_path)

    def _async_endpoint(self) -> str:
        if self._uses_international_seedance():
            return f"{self._base_url}/videos/international/generations/async"
        return f"{self._base_url}/videos/generations/async"

    def _uses_international_seedance(self) -> bool:
        return _is_seedance_model(self._model) and _is_international_token(self._api_key)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _json_headers(self) -> dict[str, str]:
        return {**self._auth_headers(), "Content-Type": "application/json"}

    @staticmethod
    def _max_wait(duration_seconds: int) -> float:
        return max(_MIN_POLL_TIMEOUT_SECONDS, duration_seconds * _POLL_TIMEOUT_PER_SECOND)


def _make_upload_tuple(path: Path) -> tuple[str, bytes, str]:
    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return path.name, path.read_bytes(), mime_type


def _is_done(state: dict[str, Any]) -> bool:
    status = str(state.get("status", "")).lower()
    return status in {"completed", "done", "succeeded", "success"}


def _stringify_failure_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail.strip()
    return str(detail).strip()


def _summarize_failure_detail(detail: str) -> str:
    first_line = next((line.strip() for line in detail.splitlines() if line.strip()), detail)
    lower_detail = detail.lower()

    if "browsercontext.newpage" in lower_detail or "target page, context or browser has been closed" in lower_detail:
        cause = "检测到上游页面静态资源加载失败" if "loading css chunk" in lower_detail else "Playwright/Chromium 会话提前关闭"
        logger.warning("Jimeng Seedance 浏览器代理异常详情: %s", detail[:1000])
        return (
            "Seedance 浏览器代理异常"
            f"（{cause}，远端 jimeng-free-api-all 服务当前不可用）。"
            "可先切换到非 Seedance Jimeng 视频模型（如 jimeng-video-3.5-pro），"
            "或升级/重启 Jimeng 服务后重试。"
            f" 原始错误: {first_line}"
        )

    return detail


def _extract_failure(state: dict[str, Any]) -> str | None:
    status = str(state.get("status", "")).lower()
    if status not in {"failed", "error", "cancelled", "canceled"}:
        return None
    error = state.get("error")
    if isinstance(error, dict):
        detail = error.get("message") or error.get("detail") or error
    else:
        detail = error or state.get("message") or "unknown"
    normalized_detail = _summarize_failure_detail(_stringify_failure_detail(detail))
    return f"Jimeng 视频生成失败: {normalized_detail}"


def _extract_video_url(state: dict[str, Any]) -> str | None:
    data = state.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        url = data[0].get("url")
        if isinstance(url, str) and url:
            return url
    url = state.get("url")
    return url if isinstance(url, str) and url else None
