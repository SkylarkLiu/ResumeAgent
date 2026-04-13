"""
生成节点 - 拼装上下文 + 分层裁剪 + 调 LLM

阶段 2 改造：
- 历史 messages 由 LangGraph checkpointer 自动管理（add_messages reducer）
- 本节点保留裁剪逻辑，但裁剪仅影响发送给 LLM 的上下文窗口，
  不影响 checkpointer 中存储的完整历史

阶段 4 改造（流式输出）：
- 新增 generate_stream 异步生成器，供 SSE API 直接调用
- 保持原有 generate 函数用于非流式调用

分层裁剪改造（方案 A）：
- System prompt 始终保留
- 早期消息压缩为摘要文本（conversation_summary），注入 system prompt
- 最近 N 条消息完整保留
- 裁剪策略由 config.py 中的 agent_recent_message_count / agent_summary_token_budget 控制
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    WEB_AGENT_SYSTEM_PROMPT,
)
from app.agent.state import AgentState
from app.agent.utils.history_utils import build_layered_messages, estimate_tokens
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion, chat_completion_stream_async

logger = setup_logger("agent.generate")

# 由 graph.py 在构建时设置
_max_history: int = 20  # 兼容旧逻辑的 fallback

# 分层裁剪参数（由 graph.py 注入）
_recent_message_count: int = 10
_summary_max_chars: int = 120
_summary_token_budget: int = 800


def set_max_history(max_history: int) -> None:
    """注入最大历史轮数（由 graph.py 调用，兼容旧逻辑）"""
    global _max_history
    _max_history = max_history


def set_layered_config(
    recent_count: int,
    summary_max_chars: int,
    summary_token_budget: int,
) -> None:
    """注入分层裁剪配置（由 graph.py 调用）"""
    global _recent_message_count, _summary_max_chars, _summary_token_budget
    _recent_message_count = recent_count
    _summary_max_chars = summary_max_chars
    _summary_token_budget = summary_token_budget


def generate(state: AgentState) -> dict:
    """
    生成节点：拼装上下文 + 分层裁剪 + 调 LLM 生成回答。

    消息来源：state["messages"] 已由 checkpointer 管理完整历史，
    本节点只做 LLM 上下文窗口裁剪（不影响持久化存储）。

    Returns:
        {"final_answer": str, "messages": [AIMessage], "conversation_summary": str}
    """
    llm_messages, merged_summary = _build_llm_messages(state)
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
    answer = chat_completion(
        llm_messages,
        thinking={"type": "disabled"},
    )
    logger.info("生成完成: 回答 %d 字符", len(answer))

    result = {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }
    if merged_summary:
        result["conversation_summary"] = merged_summary
    return result


def _emit_custom_event(payload: dict[str, Any]) -> None:
    """在 graph stream_mode=custom 下发送生成阶段事件。"""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _build_llm_messages(state: AgentState) -> list[dict]:
    """
    从 AgentState 构造 LLM 请求消息列表（供 generate 和 generate_stream 复用）。

    分层裁剪策略（方案 A）：
    1. System prompt 始终保留
    2. 早期消息 → 压缩摘要，注入 system prompt 尾部
    3. 最近 N 条消息 → 完整保留
    4. 如有 working_context，注入到最后一条用户消息

    Returns:
        OpenAI 格式的消息列表
    """
    messages = state.get("messages", [])
    working_context = state.get("working_context", "")
    route_type = state.get("route_type", "direct")
    conversation_summary = state.get("conversation_summary", "")

    # 统一转为字符串
    route_type_str = route_type.value if hasattr(route_type, "value") else str(route_type)

    # ---- 分层裁剪 ----
    complete_msgs, new_summary = build_layered_messages(
        messages,
        recent_count=_recent_message_count,
        summary_max_chars=_summary_max_chars,
        token_budget=_summary_token_budget,
    )

    # 合并已有摘要和新摘要（跨轮次累积）
    # 如果已有 conversation_summary（来自上一轮），将其与新摘要合并
    # 新摘要优先（更贴近当前对话），旧摘要追加在后面作为补充
    merged_summary = new_summary
    if conversation_summary and new_summary:
        # 旧摘要可能和新的早期消息重叠，简单合并：新摘要 + 旧摘要截断保留
        old_trimmed = conversation_summary
        if len(old_trimmed) > 400:
            old_trimmed = "…" + old_trimmed[-400:]
        merged_summary = f"{new_summary}\n[更早]{old_trimmed}"
    elif conversation_summary and not new_summary:
        # 没有新的早期消息（消息总数 < recent_count），保留旧摘要
        merged_summary = conversation_summary

    # 选择系统 prompt
    if route_type_str == "web":
        system_prompt = WEB_AGENT_SYSTEM_PROMPT
    elif route_type_str == "retrieve":
        system_prompt = AGENT_SYSTEM_PROMPT
    else:
        system_prompt = DIRECT_SYSTEM_PROMPT

    # 如果有摘要，追加到 system prompt
    if merged_summary:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"--- 早期对话摘要 ---\n"
            f"{merged_summary}\n"
            f"--- 摘要结束 ---"
        )

    # 构造 LLM 请求
    llm_messages = [{"role": "system", "content": system_prompt}]

    # 添加完整层消息
    llm_messages.extend(complete_msgs)

    # 如果有上下文，在最后一条用户消息中注入
    if working_context:
        last_user_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        # 替换最后一条用户消息为带上下文的版本
        context_block = f"\n\n参考内容：\n{working_context}\n\n---\n\n用户问题：{last_user_msg}"
        if llm_messages and llm_messages[-1]["role"] == "user":
            llm_messages[-1] = {"role": "user", "content": context_block}
        else:
            llm_messages.append({"role": "user", "content": context_block})

    # 日志：记录分层裁剪效果
    total_messages = len(messages)
    early_count = max(0, total_messages - _recent_message_count)
    summary_tokens = estimate_tokens(merged_summary) if merged_summary else 0
    logger.info(
        "分层裁剪: 总消息=%d, 早期=%d→摘要(%d token), 最近=%d完整, system+msg=%d条",
        total_messages,
        early_count,
        summary_tokens,
        len(complete_msgs),
        len(llm_messages),
    )

    return llm_messages, merged_summary


async def generate_stream(state: AgentState) -> AsyncGenerator[dict[str, Any], None]:
    """
    流式生成节点：拼装上下文 + 分层裁剪 + 流式调用 LLM。

    Yields:
        每个 SSE 事件 payload：
        - {"type": "token", "content": "..."}  - 增量文本
        - {"type": "done", "final_answer": "...", "messages": [...], "conversation_summary": "..."}  - 完整结果
    """
    llm_messages, merged_summary = _build_llm_messages(state)
    route_type = state.get("route_type", "direct")
    route_type_str = route_type.value if hasattr(route_type, "value") else str(route_type)

    logger.info(
        "流式生成请求: route=%s, 消息=%d条",
        route_type_str,
        len(llm_messages),
    )

    full_answer = ""
    async for delta in chat_completion_stream_async(
        llm_messages,
        thinking={"type": "disabled"},
    ):
        if delta:
            full_answer += delta
            yield {"type": "token", "content": delta}

    logger.info("流式生成完成: 回答 %d 字符", len(full_answer))

    # 最终返回完整结果（用于更新 state）
    done_event = {
        "type": "done",
        "final_answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
    }
    if merged_summary:
        done_event["conversation_summary"] = merged_summary
    yield done_event


async def generate_streaming_node(state: AgentState) -> dict:
    """
    图内流式桥接节点：复用 generate_stream，并通过自定义事件向外发 token。

    Returns:
        {"final_answer": str, "messages": [AIMessage], "conversation_summary": str}
    """
    final_result: dict[str, Any] | None = None

    async for event in generate_stream(state):
        event_type = event.get("type")
        if event_type == "token":
            _emit_custom_event({"type": "token", "content": event.get("content", "")})
        elif event_type == "done":
            final_result = {
                "final_answer": event.get("final_answer", ""),
                "messages": event.get("messages", [AIMessage(content=event.get("final_answer", ""))]),
            }
            if event.get("conversation_summary"):
                final_result["conversation_summary"] = event["conversation_summary"]
        elif event_type == "error":
            message = event.get("message", "生成失败")
            _emit_custom_event({"type": "error", "message": message})
            raise RuntimeError(message)

    if final_result is None:
        final_result = {
            "final_answer": "",
            "messages": [AIMessage(content="")],
        }
    return final_result
