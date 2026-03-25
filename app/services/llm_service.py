"""
LLM 服务 - 智谱 GLM 文本生成
"""
from __future__ import annotations

from zai import ZhipuAiClient

from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("llm_service")

_settings = get_settings()


def get_llm_client() -> ZhipuAiClient:
    """获取智谱 AI 客户端（单例风格，每次调用轻量创建）"""
    return ZhipuAiClient(api_key=_settings.zhipuai_api_key)


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    调用智谱 ChatCompletion API，返回文本内容。

    Args:
        messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
    Returns:
        模型回复的文本
    """
    client = get_llm_client()
    model = model or _settings.llm_model_name

    logger.debug("调用 LLM: model=%s, messages=%d条", model, len(messages))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    logger.debug("LLM 响应长度: %d 字符", len(content) if content else 0)
    return content or ""
