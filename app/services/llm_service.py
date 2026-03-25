"""
LLM 服务 - 智谱 GLM 文本生成
"""
from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import AsyncGenerator

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


def chat_completion_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """
    流式调用智谱 ChatCompletion API，yield 每个 delta 文本片段。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
    Yields:
        每个 chunk 的增量文本内容（可能为空字符串）
    """
    client = get_llm_client()
    model = model or _settings.llm_model_name

    logger.debug("流式调用 LLM: model=%s, messages=%d条", model, len(messages))

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            yield delta


async def chat_completion_async(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    异步调用智谱 ChatCompletion API，返回文本内容。
    注意：zai-sdk 暂未提供原生异步客户端，此处用同步包装。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
    Returns:
        模型回复的文本
    """
    # zai-sdk 暂无 async client，直接复用同步实现
    return chat_completion(messages, model, temperature, max_tokens)


async def chat_completion_stream_async(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    异步流式调用智谱 ChatCompletion API。

    由于 zai-sdk 暂未提供原生异步客户端，此处用同步生成器包装为异步迭代器。
    如果后续 SDK 支持，可替换为原生异步流。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
    Yields:
        每个 chunk 的增量文本内容
    """
    for delta in chat_completion_stream(messages, model, temperature, max_tokens):
        yield delta
        await asyncio.sleep(0)  # 让出事件循环控制权，使 StreamingResponse 能逐 chunk 发送
