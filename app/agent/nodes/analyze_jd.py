"""
JD 分析节点 - 基于 JD 结构化数据生成技术分析 + 简历写作建议

输入：state["jd_data"]（结构化 JD 信息）
输出：state["final_answer"]（Markdown 分析报告）+ state["messages"]

阶段 5 改造：新增流式生成器 analyze_jd_stream
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.prompts import JD_ANALYSIS_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion, chat_completion_stream_async

logger = setup_logger("agent.analyze_jd")


def _build_jd_analysis_messages(state: AgentState) -> list[dict] | None:
    """
    构造 JD 分析的 LLM 消息列表（供 analyze_jd 和 analyze_jd_stream 复用）。

    Returns:
        消息列表，如果 JD 提取有错误则返回 None（调用方需处理错误）。
    """
    jd_data = state.get("jd_data") or {}
    messages = state.get("messages", [])

    # 检查 JD 提取是否出错
    if jd_data.get("extract_error"):
        return None

    # 提取用户最后一条消息
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # 序列化 jd_data（去掉 raw_text 避免超长）
    jd_for_prompt = {k: v for k, v in jd_data.items() if k not in ("raw_text", "extract_error")}
    jd_json = json.dumps(jd_for_prompt, ensure_ascii=False, indent=2)

    # 构造 prompt
    prompt = JD_ANALYSIS_PROMPT.format(jd_data=jd_json)

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_question or "请分析这个岗位的核心要求并给出简历写作建议"},
    ]


def analyze_jd(state: AgentState) -> dict:
    """
    JD 分析节点（非流式）：基于结构化 JD 数据，生成岗位技术分析和简历写作建议。

    Returns:
        {"final_answer": str, "messages": [AIMessage]}
    """
    jd_data = state.get("jd_data") or {}
    messages = state.get("messages", [])

    # 提取用户最后一条消息
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # 检查 JD 提取是否出错
    if jd_data.get("extract_error"):
        error_msg = jd_data["extract_error"]
        logger.warning("JD 提取有错误，直接返回错误信息")
        return {
            "final_answer": f"❌ JD 解析失败：{error_msg}\n\n请尝试：\n1. 重新粘贴 JD 内容\n2. 确保包含完整的岗位描述信息",
            "messages": [AIMessage(content=f"JD 解析失败：{error_msg}")],
        }

    llm_messages = _build_jd_analysis_messages(state)

    logger.info(
        "生成 JD 分析报告: position=%s, skills_must=%d项",
        jd_data.get("position", "未知"),
        len(jd_data.get("skills_must", [])),
    )

    try:
        answer = chat_completion(
            llm_messages,
            temperature=0.5,
            max_tokens=4096,
            thinking={"type": "disabled"},
        )
        logger.info("JD 分析报告生成完成: %d 字符", len(answer))

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error("JD 分析报告生成失败: %s", e, exc_info=True)
        return {
            "final_answer": f"❌ JD 分析报告生成失败：{e}",
            "messages": [AIMessage(content=f"分析报告生成失败：{e}")],
        }


async def analyze_jd_stream(state: AgentState) -> AsyncGenerator[dict[str, Any], None]:
    """
    JD 分析流式生成节点：逐 token 生成岗位分析报告。

    Yields:
        - {"type": "token", "content": "..."}  - 增量文本
        - {"type": "done", "final_answer": "...", "messages": [...]}  - 完整结果
        - {"type": "error", "message": "..."}   - 错误信息
    """
    jd_data = state.get("jd_data") or {}

    # 检查 JD 提取是否出错
    if jd_data.get("extract_error"):
        error_msg = jd_data["extract_error"]
        error_answer = f"❌ JD 解析失败：{error_msg}\n\n请尝试：\n1. 重新粘贴 JD 内容\n2. 确保包含完整的岗位描述信息"
        yield {
            "type": "done",
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }
        return

    llm_messages = _build_jd_analysis_messages(state)
    if llm_messages is None:
        error_answer = "❌ JD 分析消息构造失败"
        yield {
            "type": "done",
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }
        return

    logger.info(
        "流式生成 JD 分析报告: position=%s",
        jd_data.get("position", "未知"),
    )

    full_answer = ""
    try:
        async for delta in chat_completion_stream_async(
            llm_messages,
            temperature=0.5,
            max_tokens=4096,
            thinking={"type": "disabled"},
        ):
            if delta:
                full_answer += delta
                yield {"type": "token", "content": delta}

        logger.info("JD 分析流式生成完成: %d 字符", len(full_answer))

        yield {
            "type": "done",
            "final_answer": full_answer,
            "messages": [AIMessage(content=full_answer)],
        }

    except Exception as e:
        logger.error("JD 分析流式生成失败: %s", e, exc_info=True)
        yield {"type": "error", "message": f"❌ JD 分析报告生成失败：{e}"}
