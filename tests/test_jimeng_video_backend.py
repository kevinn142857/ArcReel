"""JimengVideoBackend 单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.providers import PROVIDER_JIMENG
from lib.video_backends.base import VideoCapability, VideoGenerationRequest


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.raise_for_status = MagicMock()
    return response


def _fake_download_factory(payload: bytes = b"mp4-bytes"):
    async def _fake(url: str, output_path: Path, *, timeout: int = 120) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)

    return _fake


class TestJimengVideoBackend:
    def test_standard_model_capabilities(self):
        from lib.video_backends.jimeng import JimengVideoBackend

        backend = JimengVideoBackend(api_key="jm-token", model="jimeng-video-3.5-pro", base_url="http://localhost:8000")
        assert backend.name == PROVIDER_JIMENG
        assert backend.model == "jimeng-video-3.5-pro"
        assert VideoCapability.TEXT_TO_VIDEO in backend.capabilities
        assert VideoCapability.IMAGE_TO_VIDEO in backend.capabilities
        assert backend.video_capabilities.first_frame is True
        assert backend.video_capabilities.last_frame is True
        assert backend.video_capabilities.reference_images is False

    def test_seedance_capabilities(self):
        from lib.video_backends.jimeng import JimengVideoBackend

        backend = JimengVideoBackend(
            api_key="jm-token", model="jimeng-video-seedance-2.0", base_url="http://localhost:8000"
        )
        assert backend.video_capabilities.first_frame is True
        assert backend.video_capabilities.last_frame is False
        assert backend.video_capabilities.reference_images is True
        assert backend.video_capabilities.max_reference_images == 10

    async def test_text_to_video_happy_path(self, tmp_path: Path):
        create_response = _make_response(200, {"task_id": "task-42", "status": "processing"})
        poll_response = _make_response(
            200,
            {"task_id": "task-42", "status": "succeeded", "data": [{"url": "https://cdn.example.com/out.mp4"}]},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=poll_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = AsyncMock(side_effect=_fake_download_factory())

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.jimeng._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.jimeng.download_video", fake_download),
        ):
            from lib.video_backends.jimeng import JimengVideoBackend

            backend = JimengVideoBackend(
                api_key="jm-token", base_url="http://localhost:8000", model="jimeng-video-3.5-pro"
            )
            result = await backend.generate(
                VideoGenerationRequest(
                    prompt="a runner in neon rain",
                    output_path=tmp_path / "out.mp4",
                    aspect_ratio="9:16",
                    resolution="720p",
                    duration_seconds=5,
                )
            )

        assert result.provider == PROVIDER_JIMENG
        assert result.model == "jimeng-video-3.5-pro"
        assert result.task_id == "task-42"
        assert result.video_uri == "https://cdn.example.com/out.mp4"
        assert result.video_path.read_bytes() == b"mp4-bytes"

        post_call = mock_client.post.call_args
        assert post_call.args[0].endswith("/videos/generations/async")
        assert post_call.kwargs["json"] == {
            "model": "jimeng-video-3.5-pro",
            "prompt": "a runner in neon rain",
            "ratio": "9:16",
            "resolution": "720p",
            "duration": 5,
        }
        assert post_call.kwargs["headers"]["Authorization"] == "Bearer jm-token"

    async def test_standard_image_to_video_uses_first_and_last_frame(self, tmp_path: Path):
        start_image = tmp_path / "start.png"
        end_image = tmp_path / "end.png"
        start_image.write_bytes(b"start")
        end_image.write_bytes(b"end")

        create_response = _make_response(200, {"task_id": "task-1", "status": "processing"})
        poll_response = _make_response(
            200,
            {"task_id": "task-1", "status": "succeeded", "data": [{"url": "https://cdn.example.com/out.mp4"}]},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=poll_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = AsyncMock(side_effect=_fake_download_factory())

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.jimeng._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.jimeng.download_video", fake_download),
        ):
            from lib.video_backends.jimeng import JimengVideoBackend

            backend = JimengVideoBackend(api_key="jm-token", base_url="http://localhost:8000", model="jimeng-video-3.0")
            await backend.generate(
                VideoGenerationRequest(
                    prompt="animate these frames",
                    output_path=tmp_path / "out.mp4",
                    start_image=start_image,
                    end_image=end_image,
                    resolution="720p",
                    aspect_ratio="16:9",
                    duration_seconds=10,
                )
            )

        post_call = mock_client.post.call_args
        assert post_call.args[0].endswith("/videos/generations/async")
        assert post_call.kwargs["data"] == {
            "model": "jimeng-video-3.0",
            "prompt": "animate these frames",
            "ratio": "16:9",
            "resolution": "720p",
            "duration": "10",
        }
        assert [field for field, _ in post_call.kwargs["files"]] == ["files", "files"]

    async def test_international_seedance_uses_international_async_endpoint(self, tmp_path: Path):
        start_image = tmp_path / "start.png"
        ref1 = tmp_path / "ref1.png"
        ref2 = tmp_path / "ref2.png"
        start_image.write_bytes(b"start")
        ref1.write_bytes(b"ref-1")
        ref2.write_bytes(b"ref-2")

        create_response = _make_response(200, {"task_id": "seedance-1", "status": "processing"})
        poll_response = _make_response(
            200,
            {"task_id": "seedance-1", "status": "succeeded", "data": [{"url": "https://cdn.example.com/seedance.mp4"}]},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=poll_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = AsyncMock(side_effect=_fake_download_factory(b"seedance"))

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.jimeng._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.jimeng.download_video", fake_download),
        ):
            from lib.video_backends.jimeng import JimengVideoBackend

            backend = JimengVideoBackend(
                api_key="jp-token-123",
                base_url="http://localhost:8000",
                model="jimeng-video-seedance-2.0-fast",
            )
            await backend.generate(
                VideoGenerationRequest(
                    prompt="use all references",
                    output_path=tmp_path / "seedance.mp4",
                    start_image=start_image,
                    reference_images=[ref1, ref2],
                    aspect_ratio="4:3",
                    resolution="720p",
                    duration_seconds=6,
                )
            )

        post_call = mock_client.post.call_args
        assert post_call.args[0].endswith("/videos/international/generations/async")
        assert post_call.kwargs["data"] == {
            "model": "jimeng-video-seedance-2.0-fast",
            "prompt": "use all references",
            "ratio": "4:3",
            "resolution": "720p",
            "duration": "6",
        }
        assert [field for field, _ in post_call.kwargs["files"]] == ["image_file", "image_file_1", "image_file_2"]

    async def test_international_seedance_requires_materials(self, tmp_path: Path):
        from lib.video_backends.jimeng import JimengVideoBackend

        backend = JimengVideoBackend(
            api_key="us-token-123",
            base_url="http://localhost:8000",
            model="jimeng-video-seedance-2.0",
        )

        with pytest.raises(ValueError, match="至少需要一张参考图或首帧图"):
            await backend.generate(
                VideoGenerationRequest(
                    prompt="text only",
                    output_path=tmp_path / "out.mp4",
                    aspect_ratio="4:3",
                    resolution="720p",
                    duration_seconds=4,
                )
            )

    def test_extract_failure_summarizes_seedance_browser_proxy_crash(self):
        from lib.video_backends.jimeng import _extract_failure

        message = _extract_failure(
            {
                "status": "failed",
                "error": {
                    "message": "\n".join(
                        [
                            "browserContext.newPage: Target page, context or browser has been closed",
                            "Browser logs:",
                            "[pid=32][err] Uncaught (in promise) Error: Loading CSS chunk 9133 failed.",
                        ]
                    )
                },
            }
        )

        assert message is not None
        assert "Seedance 浏览器代理异常" in message
        assert "上游页面静态资源加载失败" in message
        assert "jimeng-video-3.5-pro" in message
        assert "browserContext.newPage: Target page, context or browser has been closed" in message

    def test_extract_failure_preserves_regular_provider_message(self):
        from lib.video_backends.jimeng import _extract_failure

        message = _extract_failure({"status": "failed", "error": {"message": "生成失败，错误码: 2038"}})

        assert message == "Jimeng 视频生成失败: 生成失败，错误码: 2038"
