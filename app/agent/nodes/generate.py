"""
生成节点 - 拼装上下文 + 历史裁剪 + 调 LLM

阶段 2 改造：
- 历史 messages 由 LangGraph checkpointer 自动管理（add_messages reducer）
- 本节点仍保留 _max_history 裁剪，但裁剪仅影响发送给 LLM 的上下文窗口，
  不影响 checkpointer 中存储的完整历史

阶段 4 改造（流式输出）：
- 新增 generate_stream 异步生成器，供 SSE API 直接调用
- 保持原有 generate 函数用于非流式调用
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    WEB_AGENT_SYSTEM_PROMPT,
)
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion, chat_completion_stream_async

logger = setup_logger("agent.generate")

# 由 graph.py 在构建时设置
_max_history: int = 20


def set_max_history(max_history: int) -> None:
    """注入最大历史轮数（由 graph.py 调用）"""
    global _max_history
    _max_history = max_history


def generate(state: AgentState) -> dict:
    """
    生成节点：拼装上下文 + 历史裁剪 + 调 LLM 生成回答。

    消息来源：state["messages"] 已由 checkpointer 管理完整历史，
    本节点只做 LLM 上下文窗口裁剪（不影响持久化存储）。

    Returns:
        {"final_answer": str, "messages": [AIMessage]}
    """
    llm_messages = _build_llm_messages(state)
    route_type = state.get("route_type", "direct")
    route_type_str = route_type.value if hasattr(route_type, "value") else str(route_type)
    working_context = state.get("working_context", "")

    logger.info(
        "生成请求: route=%s, 消息=%d条, 上下文=%d字符",
        route_type_str,
        len(llm_messages),
        len(working_context),
    )

    # 调用 LLM
    answer = chat_completion(llm_messages)
    logger.info("生成完成: 回答 %d 字符", len(answer))

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def _build_llm_messages(state: AgentState) -> list[dict]:
    """
    从 AgentState 构造 LLM 请求消息列表（供 generate 和 generate_stream 复用）。

    Returns:
        OpenAI 格式的消息列表
    """
    messages = state.get("messages", [])
    working_context = state.get("working_context", "")
    route_type = state.get("route_type", "direct")

    # 统一转为字符串
    route_type_str = route_type.value if hasattr(route_type, "value") else str(route_type)

    # 裁剪发送给 LLM 的上下文（不影响 checkpointer 存储）
    trimmed_messages = messages[-_max_history:] if len(messages) > _max_history else messages

    # 选择系统 prompt
    if route_type_str == "web":
        system_prompt = WEB_AGENT_SYSTEM_PROMPT
    elif route_type_str == "retrieve":
        system_prompt = AGENT_SYSTEM_PROMPT
    else:
        system_prompt = DIRECT_SYSTEM_PROMPT

    # 构造 LLM 请求
    llm_messages = [{"role": "system", "content": system_prompt}]

    # 添加历史对话（排除路由决策消息）
    for msg in trimmed_messages:
        if isinstance(msg, HumanMessage):
            llm_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and not msg.content.startswith("[路由决策]"):
            llm_messages.append({"role": "assistant", "content": msg.content})

    # 如果有上下文，在最后一条用户消息中注入
    if working_context:
        last_user_msg = ""
        for msg in reversed(trimmed_messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        # 替换最后一条用户消息为带上下文的版本
        context_block = f"\n\n参考内容：\n{working_context}\n\n---\n\n用户问题：{last_user_msg}"
        if llm_messages and llm_messages[-1]["role"] == "user":
            llm_messages[-1] = {"role": "user", "content": context_block}
        else:
            llm_messages.append({"role": "user", "content": context_block})

    return llm_messages


async def generate_stream(state: AgentState) -> AsyncGenerator[dict[str, Any], None]:
    """
    流式生成节点：拼装上下文 + 历史裁剪 + 流式调用 LLM。

    Yields:
        每个 SSE 事件 payload：
        - {"type": "token", "content": "..."}  - 增量文本
        - {"type": "done", "final_answer": "...", "messages": [...]}  - 完整结果
    """
    llm_messages = _build_llm_messages(state)
    route_type = state.get("route_type", "direct")
    route_type_str = route_type.value if hasattr(route_type, "value") else str(route_type)

    logger.info(
        "流式生成请求: route=%s, 消息=%d条",
        route_type_str,
        len(llm_messages),
    )

    full_answer = ""
    async for delta in chat_completion_stream_async(llm_messages):
        if delta:
            full_answer += delta
            yield {"type": "token", "content": delta}

    logger.info("流式生成完成: 回答 %d 字符", len(full_answer))

    # 最终返回完整结果（用于更新 state）
    yield {
        "type": "done",
        "final_answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
    }
