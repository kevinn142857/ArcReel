"""自定义供应商模型能力推断。"""

from __future__ import annotations

_GROK2API_VIDEO_DURATIONS: dict[str, list[int]] = {
    "grok-imagine-video": [6, 10, 12, 16, 20],
}


def infer_supported_durations(
    *,
    api_format: str | None,
    model_id: str | None,
    media_type: str | None = None,
) -> list[int] | None:
    """根据自定义供应商模式与模型 ID 推断 supported_durations。"""
    normalized_api_format = (api_format or "").strip().lower()
    normalized_model_id = (model_id or "").strip().lower()
    normalized_media_type = (media_type or "video").strip().lower()

    if normalized_media_type != "video" or not normalized_model_id:
        return None

    if normalized_api_format == "grok2api":
        durations = _GROK2API_VIDEO_DURATIONS.get(normalized_model_id)
        if durations:
            return list(durations)

    return None


def coalesce_supported_durations(
    explicit: list[int] | None,
    *,
    api_format: str | None,
    model_id: str | None,
    media_type: str | None = None,
) -> list[int] | None:
    """优先使用显式配置；缺失时回退到可推断的默认时长。"""
    if explicit:
        return [int(d) for d in explicit]
    return infer_supported_durations(api_format=api_format, model_id=model_id, media_type=media_type)
