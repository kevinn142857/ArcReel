"""URL 归一化工具函数。"""

from __future__ import annotations

import re
from urllib.parse import urlparse


def ensure_openai_base_url(url: str | None) -> str | None:
    """自动补全 OpenAI 兼容 API 的 /v1 路径后缀。

    用户可能只填了 ``https://api.example.com``，但 OpenAI SDK 期望
    ``https://api.example.com/v1``。本函数在缺少版本路径时自动追加。
    """
    if not url:
        return url
    stripped = url.strip().rstrip("/")
    if not re.search(r"/v\d+$", stripped):
        stripped += "/v1"
    return stripped


def normalize_base_url(url: str | None) -> str | None:
    """确保 base_url 以 / 结尾。

    Google genai SDK 的 http_options.base_url 要求尾部带 /，
    否则请求路径拼接会失败。预置 Gemini 后端使用此函数。
    """
    if not url:
        return None
    url = url.strip()
    if not url:
        return None
    if not url.endswith("/"):
        url += "/"
    return url


def ensure_google_base_url(url: str | None) -> str | None:
    """规范化 Google genai SDK 的 base_url。

    Google genai SDK 会自动在 base_url 后拼接 ``api_version``（默认 ``v1beta``）。
    如果用户误填了 ``https://example.com/v1beta``，SDK 会拼出
    ``https://example.com/v1beta/v1beta/models``，导致请求失败。

    本函数剥离末尾的版本路径（如 ``/v1beta``、``/v1``），并确保尾部带 ``/``。
    """
    if not url:
        return None
    url = url.strip()
    if not url:
        return None
    url = url.rstrip("/")
    # 剥离末尾的版本路径（/v1, /v1beta, /v1alpha 等）
    url = re.sub(r"/v\d+\w*$", "", url)
    if not url.endswith("/"):
        url += "/"
    return url


def resolve_grok_api_host(url: str | None) -> tuple[str | None, bool]:
    """将 Grok 自定义 base_url 解析为 xAI SDK 可接受的 api_host。

    xAI SDK 使用 gRPC，接收的是 ``api_host`` 而不是 HTTP 风格的 ``base_url``。
    这里兼容用户常见输入：

    - ``https://api.x.ai`` -> (``api.x.ai``, False)
    - ``http://localhost:50051`` -> (``localhost:50051``, True)
    - ``https://proxy.example.com/v1`` -> (``proxy.example.com``, False)
    - ``proxy.example.com:443`` -> (``proxy.example.com:443``, False)

    Returns:
        (api_host, use_insecure_channel)
    """
    if not url:
        return None, False

    raw = url.strip()
    if not raw:
        return None, False

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if not parsed.hostname:
        raise ValueError("Grok Base URL 必须包含主机名")

    api_host = parsed.netloc
    if not api_host:
        raise ValueError("Grok Base URL 必须包含主机名")

    if "://" in raw:
        use_insecure_channel = parsed.scheme == "http"
    else:
        use_insecure_channel = parsed.hostname in {"localhost", "127.0.0.1", "::1"}

    return api_host, use_insecure_channel
