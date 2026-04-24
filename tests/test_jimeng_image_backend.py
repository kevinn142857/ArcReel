"""JimengImageBackend 单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from lib.image_backends.base import ImageCapability, ImageGenerationRequest, ReferenceImage
from lib.providers import PROVIDER_JIMENG


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.raise_for_status = MagicMock()
    return response


def _fake_download_factory(payload: bytes = b"png-bytes"):
    async def _fake(url: str, output_path: Path, *, timeout: int = 120) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)

    return _fake


class TestJimengImageBackend:
    def test_name_model_and_capabilities(self):
        from lib.image_backends.jimeng import JimengImageBackend

        backend = JimengImageBackend(api_key="jm-token", model="jimeng-5.0", base_url="http://localhost:8000")
        assert backend.name == PROVIDER_JIMENG
        assert backend.model == "jimeng-5.0"
        assert backend.capabilities == {
            ImageCapability.TEXT_TO_IMAGE,
            ImageCapability.IMAGE_TO_IMAGE,
        }

    async def test_text_to_image_happy_path(self, tmp_path: Path):
        post_response = _make_response(200, {"data": [{"url": "https://cdn.example.com/out.png"}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = AsyncMock(side_effect=_fake_download_factory())

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.image_backends.jimeng.download_image", fake_download),
        ):
            from lib.image_backends.jimeng import JimengImageBackend

            backend = JimengImageBackend(api_key="jm-token", base_url="http://localhost:8000", model="jimeng-4.6")
            result = await backend.generate(
                ImageGenerationRequest(
                    prompt="a cinematic portrait",
                    output_path=tmp_path / "out.png",
                    aspect_ratio="9:16",
                    image_size="2K",
                )
            )

        assert result.provider == PROVIDER_JIMENG
        assert result.model == "jimeng-4.6"
        assert result.image_path == tmp_path / "out.png"
        assert result.image_uri == "https://cdn.example.com/out.png"
        assert result.image_path.read_bytes() == b"png-bytes"

        post_call = mock_client.post.call_args
        assert post_call.args[0].endswith("/images/generations")
        assert post_call.kwargs["json"] == {
            "model": "jimeng-4.6",
            "prompt": "a cinematic portrait",
            "ratio": "9:16",
            "resolution": "2k",
            "response_format": "url",
        }
        assert post_call.kwargs["headers"]["Authorization"] == "Bearer jm-token"

    async def test_image_to_image_uses_multipart_images(self, tmp_path: Path):
        ref1 = tmp_path / "ref1.png"
        ref2 = tmp_path / "ref2.jpg"
        ref1.write_bytes(b"img-1")
        ref2.write_bytes(b"img-2")

        post_response = _make_response(200, {"data": [{"url": "https://cdn.example.com/out.png"}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = AsyncMock(side_effect=_fake_download_factory(b"merged"))

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.image_backends.jimeng.download_image", fake_download),
        ):
            from lib.image_backends.jimeng import JimengImageBackend

            backend = JimengImageBackend(api_key="jm-token", base_url="http://localhost:8000", model="jimeng-5.0")
            await backend.generate(
                ImageGenerationRequest(
                    prompt="blend these references",
                    output_path=tmp_path / "merged.png",
                    aspect_ratio="16:9",
                    image_size="4K",
                    reference_images=[
                        ReferenceImage(path=str(ref1)),
                        ReferenceImage(path=str(ref2)),
                    ],
                )
            )

        post_call = mock_client.post.call_args
        assert post_call.args[0].endswith("/images/generations")
        assert post_call.kwargs["data"] == {
            "model": "jimeng-5.0",
            "prompt": "blend these references",
            "ratio": "16:9",
            "resolution": "4k",
            "response_format": "url",
        }
        assert [field for field, _ in post_call.kwargs["files"]] == ["images", "images"]
        assert post_call.kwargs["headers"] == {"Authorization": "Bearer jm-token"}

    async def test_missing_reference_images_falls_back_to_text_to_image(self, tmp_path: Path):
        post_response = _make_response(200, {"data": [{"url": "https://cdn.example.com/out.png"}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = AsyncMock(side_effect=_fake_download_factory())

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.image_backends.jimeng.download_image", fake_download),
        ):
            from lib.image_backends.jimeng import JimengImageBackend

            backend = JimengImageBackend(api_key="jm-token", base_url="http://localhost:8000")
            await backend.generate(
                ImageGenerationRequest(
                    prompt="fallback mode",
                    output_path=tmp_path / "out.png",
                    reference_images=[ReferenceImage(path=str(tmp_path / "missing.png"))],
                )
            )

        assert "json" in mock_client.post.call_args.kwargs
        assert "files" not in mock_client.post.call_args.kwargs
