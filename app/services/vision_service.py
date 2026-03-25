"""
视觉理解服务 - 智谱 GLM-4V-Flash 多模态理解
"""
from __future__ import annotations

import base64
from pathlib import Path

from zai import ZhipuAiClient

from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("vision_service")

_settings = get_settings()


def get_vision_client() -> ZhipuAiClient:
    return ZhipuAiClient(api_key=_settings.zhipuai_api_key)


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
        image_url = f"data:image/{mime};base64,{b64}"

    elif image_base64:
        image_url = f"data:image/jpeg;base64,{image_base64}"

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
