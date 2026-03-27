"""
视觉理解服务 - 智谱 GLM-4V-Flash 多模态理解
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image
from zai import ZhipuAiClient

from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("vision_service")

_settings = get_settings()

# 图片压缩参数：长边上限、JPEG 质量
_MAX_IMAGE_LONG_EDGE = 2048
_JPEG_QUALITY = 85


def get_vision_client() -> ZhipuAiClient:
    return ZhipuAiClient(api_key=_settings.zhipuai_api_key)


def compress_image_base64(
    image_base64: str,
    max_long_edge: int = _MAX_IMAGE_LONG_EDGE,
    quality: int = _JPEG_QUALITY,
) -> tuple[str, str]:
    """
    压缩 base64 图片，防止超过智谱 API 大小限制（约 4MB）。

    策略：
    1. 如果 base64 已经 < 3MB，直接返回（安全余量）
    2. 否则用 Pillow 缩放长边到 max_long_edge，转 JPEG 压缩

    Args:
        image_base64: 原始 base64 编码（不含 data:image/xxx;base64, 前缀）
        max_long_edge: 长边最大像素
        quality: JPEG 压缩质量（1-100）

    Returns:
        (base64_data, mime_type) 元组
        - base64_data: 压缩后的 base64 编码（或原图）
        - mime_type: 图片 MIME 子类型，如 "jpeg", "png", "webp"
    """
    raw_size = len(image_base64)

    # 尝试检测原始图片格式（用于跳过压缩时返回正确的 MIME）
    original_mime = "jpeg"  # 默认
    try:
        img_bytes = base64.b64decode(image_base64)
        img_probe = Image.open(io.BytesIO(img_bytes))
        fmt = img_probe.format or "JPEG"
        original_mime = fmt.lower()
        if original_mime == "jpg":
            original_mime = "jpeg"
    except Exception:
        pass

    # 3MB base64 ≈ 2.25MB 原始文件，安全范围内直接跳过
    if raw_size < 3 * 1024 * 1024:
        return image_base64, original_mime

    try:
        img = Image.open(io.BytesIO(img_bytes))

        # RGBA → RGB（JPEG 不支持 alpha 通道）
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")

        w, h = img.size
        # 长边超过阈值才缩放
        if max(w, h) > max_long_edge:
            scale = max_long_edge / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(
                "图片缩放: %dx%d → %dx%d (base64 %.1fMB → 压缩中)",
                w, h, new_w, new_h, raw_size / 1024 / 1024,
            )

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        compressed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info(
            "图片压缩完成: base64 %.1fMB → %.1fMB",
            raw_size / 1024 / 1024,
            len(compressed_b64) / 1024 / 1024,
        )
        return compressed_b64, "jpeg"

    except Exception as e:
        logger.warning("图片压缩失败，使用原图: %s", e)
        return image_base64, original_mime


def vision_chat(
    prompt: str,
    image_url: str | None = None,
    image_base64: str | None = None,
    image_path: str | Path | None = None,
    model: str | None = None,
) -> str:
    """
    调用视觉模型，支持 URL / base64 / 本地文件三种图片输入。

    Args:
        prompt: 文本提示
        image_url: 图片 URL（公开可访问）
        image_base64: base64 编码的图片（不含 data:image/xxx;base64, 前缀）
        image_path: 本地图片文件路径
        model: 模型名，默认从配置读取
    Returns:
        模型回复的文本
    """
    client = get_vision_client()
    model = model or _settings.vision_model_name

    # 构造 content 消息体
    content_parts: list[dict] = [{"type": "text", "text": prompt}]

    if image_path:
        # 从本地文件读取 base64
        p = Path(image_path)
        ext = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        b64, compressed_mime = compress_image_base64(b64)
        image_url = f"data:image/{compressed_mime};base64,{b64}"

    elif image_base64:
        b64, compressed_mime = compress_image_base64(image_base64)
        image_url = f"data:image/{compressed_mime};base64,{b64}"

    if image_url:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })

    messages = [{"role": "user", "content": content_parts}]

    logger.debug("调用视觉模型: model=%s", model)
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    logger.debug("视觉模型响应长度: %d 字符", len(content) if content else 0)
    return content or ""


# ---------- 图片理解专用 prompt ----------

IMAGE_UNDERSTANDING_PROMPT = """请分析这张图片，按以下结构输出：

## 摘要
（简要描述这张图片的整体内容和类型）

## 关键元素
（列出图片中的主要对象、场景、人物等）

## 文字识别
（如果图片中包含文字，请完整列出；没有则写"无"）

## 表格/图表信息
（如果图片中包含表格或图表，请提取其关键数据；没有则写"无"）

## 关键词
（给出 3-5 个检索关键词，用逗号分隔）

## 文件来源
（如果图片是截图或文档页面，请推测其来源类型，如"电子表格截图""PPT 页面""网页截图"等；无法判断则写"未知"）"""


def understand_image(
    image_url: str | None = None,
    image_base64: str | None = None,
    image_path: str | Path | None = None,
) -> str:
    """
    图片深度理解 - 生成可检索的文本化知识单元。
    返回结构化分析结果，适合直接入库做向量检索。
    """
    return vision_chat(
        prompt=IMAGE_UNDERSTANDING_PROMPT,
        image_url=image_url,
        image_base64=image_base64,
        image_path=image_path,
    )
