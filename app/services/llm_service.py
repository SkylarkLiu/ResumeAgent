"""
LLM 服务 - 智谱 GLM 文本生成
"""
from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any, AsyncGenerator

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
    *,
    thinking: dict | None = None,
) -> str:
    """
    调用智谱 ChatCompletion API，返回文本内容。

    兼容 glm-5 深度思考模式：
    - 如果 content 为空但 reasoning_content 有值，自动取 reasoning_content
    - 通过 thinking 参数控制是否开启深度思考

    Args:
        messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        thinking: 深度思考配置，如 {"type": "disabled"} 关闭思考模式
    Returns:
        模型回复的文本
    """
    client = get_llm_client()
    model = model or _settings.llm_model_name

    logger.debug("调用 LLM: model=%s, messages=%d条, thinking=%s", model, len(messages), thinking)

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking is not None:
        kwargs["thinking"] = thinking

    response = client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    content = msg.content

    # glm-5 深度思考模式：content 可能为空，推理内容在 reasoning_content
    if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
        content = msg.reasoning_content
        logger.debug("glm-5 深度思考模式: 从 reasoning_content 取得 %d 字符", len(content))

    logger.debug("LLM 响应长度: %d 字符", len(content) if content else 0)
    return content or ""


def chat_completion_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    *,
    thinking: dict | None = None,
) -> Generator[str, None, None]:
    """
    流式调用智谱 ChatCompletion API，yield 每个 delta 文本片段。
    兼容 glm-5 深度思考模式：当 content 为空时也输出 reasoning_content。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        thinking: 深度思考配置
    Yields:
        每个 chunk 的增量文本内容
    """
    client = get_llm_client()
    model = model or _settings.llm_model_name

    logger.debug("流式调用 LLM: model=%s, messages=%d条, thinking=%s", model, len(messages), thinking)

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if thinking is not None:
        kwargs["thinking"] = thinking

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 优先取 content；glm-5 思考模式下 content 可能为空但 reasoning_content 有值
        if delta.content:
            yield delta.content
        elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
            yield delta.reasoning_content


def chat_completion_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    tool_choice: str = "auto",
    thinking: dict | None = None,
) -> dict[str, Any]:
    """
    调用智谱 ChatCompletion API（带 tools），返回文本内容和工具调用。

    Returns:
        {
            "content": str,
            "tool_calls": [
                {
                    "id": str,
                    "name": str,
                    "arguments": str,
                }
            ],
            "finish_reason": str | None,
        }
    """
    client = get_llm_client()
    model = model or _settings.llm_model_name

    logger.debug(
        "调用 LLM tools: model=%s, messages=%d条, tools=%d, tool_choice=%s",
        model,
        len(messages),
        len(tools),
        tool_choice,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking is not None:
        kwargs["thinking"] = thinking

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message

    content = getattr(message, "content", None) or ""
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", []) or []:
        function = getattr(tool_call, "function", None)
        tool_calls.append(
            {
                "id": getattr(tool_call, "id", "") or "",
                "name": getattr(function, "name", "") or "",
                "arguments": getattr(function, "arguments", "") or "",
            }
        )

    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": getattr(choice, "finish_reason", None),
    }


async def chat_completion_async(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    *,
    thinking: dict | None = None,
) -> str:
    """
    异步调用智谱 ChatCompletion API，返回文本内容。
    注意：zai-sdk 暂未提供原生异步客户端，此处用同步包装。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认从配置读取
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        thinking: 深度思考配置
    Returns:
        模型回复的文本
    """
    # zai-sdk 暂无 async client，直接复用同步实现
    return chat_completion(messages, model, temperature, max_tokens, thinking=thinking)


async def chat_completion_stream_async(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    *,
    thinking: dict | None = None,
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
        thinking: 深度思考配置
    Yields:
        每个 chunk 的增量文本内容
    """
    for delta in chat_completion_stream(messages, model, temperature, max_tokens, thinking=thinking):
        yield delta
        await asyncio.sleep(0)  # 让出事件循环控制权，使 StreamingResponse 能逐 chunk 发送
