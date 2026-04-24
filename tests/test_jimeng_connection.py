"""Jimeng 连接测试 (_test_jimeng) 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from server.routers.providers import _test_jimeng
from tests.conftest import make_translator

_t = make_translator()


def _make_client_with_models(models: list[dict]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"data": models}
    response.raise_for_status = MagicMock()

    client = MagicMock()
    client.get.return_value = response
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    return client


class TestJimengConnection:
    def test_filters_relevant_models(self):
        mock_client = _make_client_with_models(
            [
                {"id": "jimeng"},
                {"id": "jimeng-4.6"},
                {"id": "jimeng-video-3.5-pro"},
                {"id": "seedance-2.0"},
                {"id": "unrelated-model"},
            ]
        )

        with patch("httpx.Client", return_value=mock_client):
            result = _test_jimeng({"api_key": "jm-token", "base_url": "http://localhost:8000"}, _t)

        assert result.success is True
        assert result.message == "连接成功"
        assert result.available_models == ["jimeng-4.6", "jimeng-video-3.5-pro", "seedance-2.0"]
        mock_client.get.assert_called_once_with(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer jm-token"},
        )

    def test_invalid_body_raises_runtime_error(self):
        response = MagicMock()
        response.json.return_value = {"oops": "bad-format"}
        response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="/models 返回格式不正确"):
                _test_jimeng({"api_key": "jm-token", "base_url": "http://localhost:8000"}, _t)
