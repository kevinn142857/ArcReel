"""GrokRestVideoBackend 单元测试（mock httpx）。"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.providers import PROVIDER_GROK
from lib.video_backends.base import VideoCapability, VideoGenerationRequest


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


def _fake_download_factory(payload: bytes = b"mp4-bytes"):
    async def _fake(url: str, output_path: Path, *, timeout: int = 120) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)

    return _fake


class TestGrokRestVideoBackend:
    def test_name_and_model(self):
        from lib.video_backends.grok_rest import GrokRestVideoBackend

        backend = GrokRestVideoBackend(
            api_key="xai-test", base_url="https://proxy.example.com", model="grok-imagine-video"
        )
        assert backend.name == PROVIDER_GROK
        assert backend.model == "grok-imagine-video"

    def test_capabilities(self):
        from lib.video_backends.grok_rest import GrokRestVideoBackend

        backend = GrokRestVideoBackend(api_key="xai-test", base_url="https://proxy.example.com")
        assert VideoCapability.TEXT_TO_VIDEO in backend.capabilities
        assert VideoCapability.IMAGE_TO_VIDEO in backend.capabilities
        assert backend.video_capabilities.reference_images is True
        assert backend.video_capabilities.max_reference_images == 7

    async def test_text_to_video_happy_path(self, tmp_path: Path):
        create_resp = _make_response(200, {"id": "vid-42", "status": "pending"})
        poll_resp = _make_response(
            200,
            {
                "id": "vid-42",
                "status": "completed",
                "video": {"url": "https://cdn.example.com/grok.mp4", "duration": 5},
            },
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        fake_download = AsyncMock(side_effect=_fake_download_factory())

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.grok_rest._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.grok_rest.download_video", fake_download),
        ):
            from lib.video_backends.grok_rest import GrokRestVideoBackend

            backend = GrokRestVideoBackend(
                api_key="xai-test", base_url="https://proxy.example.com", model="grok-imagine-video"
            )
            result = await backend.generate(
                VideoGenerationRequest(
                    prompt="A cat running",
                    output_path=tmp_path / "out.mp4",
                    aspect_ratio="9:16",
                    resolution="720p",
                    duration_seconds=5,
                )
            )

        assert result.provider == PROVIDER_GROK
        assert result.model == "grok-imagine-video"
        assert result.duration_seconds == 5
        assert result.task_id == "vid-42"
        assert result.video_uri == "https://cdn.example.com/grok.mp4"
        assert result.generate_audio is True

        post_call = mock_client.post.call_args
        assert post_call.args[0] == "https://proxy.example.com/v1/videos/generations"
        assert post_call.kwargs["json"]["prompt"] == "A cat running"
        assert post_call.kwargs["json"]["model"] == "grok-imagine-video"
        assert post_call.kwargs["headers"]["Authorization"] == "Bearer xai-test"

        get_call = mock_client.get.call_args
        assert get_call.args[0] == "https://proxy.example.com/v1/videos/vid-42"

    async def test_image_to_video_uses_image_object(self, tmp_path: Path):
        img_bytes = b"\x89PNG\r\nfake"
        img_path = tmp_path / "start.png"
        img_path.write_bytes(img_bytes)

        create_resp = _make_response(200, {"id": "vid-1", "status": "pending"})
        poll_resp = _make_response(200, {"id": "vid-1", "status": "done", "url": "https://cdn.example.com/v.mp4"})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        fake_download = AsyncMock(side_effect=_fake_download_factory(b"v"))

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.grok_rest._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.grok_rest.download_video", fake_download),
        ):
            from lib.video_backends.grok_rest import GrokRestVideoBackend

            backend = GrokRestVideoBackend(api_key="xai-test", base_url="https://proxy.example.com")
            await backend.generate(
                VideoGenerationRequest(
                    prompt="Bring this scene to life",
                    output_path=tmp_path / "out.mp4",
                    start_image=img_path,
                    aspect_ratio="16:9",
                    resolution="720p",
                    duration_seconds=5,
                )
            )

        sent_image = mock_client.post.call_args.kwargs["json"]["image"]["url"]
        expected = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        assert sent_image == expected

    async def test_reference_images_take_priority_when_both_inputs_provided(self, tmp_path: Path, caplog):
        start_path = tmp_path / "start.png"
        ref_path = tmp_path / "ref.png"
        start_path.write_bytes(b"\x89PNG\r\nstart")
        ref_path.write_bytes(b"\x89PNG\r\nref")

        create_resp = _make_response(200, {"id": "vid-2", "status": "pending"})
        poll_resp = _make_response(200, {"id": "vid-2", "status": "done", "url": "https://cdn.example.com/v.mp4"})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        fake_download = AsyncMock(side_effect=_fake_download_factory(b"v"))

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.grok_rest._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.grok_rest.download_video", fake_download),
            caplog.at_level("WARNING", logger="lib.video_backends.grok_rest"),
        ):
            from lib.video_backends.grok_rest import GrokRestVideoBackend

            backend = GrokRestVideoBackend(api_key="xai-test", base_url="https://proxy.example.com")
            await backend.generate(
                VideoGenerationRequest(
                    prompt="Cinematic scene",
                    output_path=tmp_path / "out.mp4",
                    start_image=start_path,
                    reference_images=[ref_path],
                    aspect_ratio="16:9",
                    resolution="720p",
                    duration_seconds=5,
                )
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert "reference_images" in payload
        assert "image" not in payload
        assert any("已优先使用 reference_images" in rec.message for rec in caplog.records)

    async def test_failed_status_raises(self, tmp_path: Path):
        create_resp = _make_response(200, {"id": "vid-err", "status": "pending"})
        poll_resp = _make_response(200, {"id": "vid-err", "status": "failed", "error": {"message": "upstream down"}})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.grok_rest._POLL_INTERVAL_SECONDS", 0.0),
        ):
            from lib.video_backends.grok_rest import GrokRestVideoBackend

            backend = GrokRestVideoBackend(api_key="xai-test", base_url="https://proxy.example.com")
            with pytest.raises(RuntimeError, match="upstream down"):
                await backend.generate(
                    VideoGenerationRequest(
                        prompt="A cat running",
                        output_path=tmp_path / "out.mp4",
                        aspect_ratio="9:16",
                        resolution="720p",
                        duration_seconds=5,
                    )
                )
