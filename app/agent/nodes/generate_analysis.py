"""
简历分析报告生成节点 - 基于 resume_data + JD 上下文生成结构化分析报告

输入：state["resume_data"] + state["working_context"]（JD 检索结果）
输出：state["final_answer"]（Markdown 分析报告）+ state["messages"]

阶段 5 改造：新增流式生成器 generate_analysis_stream
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.prompts import RESUME_ANALYSIS_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion, chat_completion_stream_async

logger = setup_logger("agent.generate_analysis")


def _build_analysis_messages(state: AgentState) -> list[dict] | None:
    """
    构造简历分析的 LLM 消息列表（供 generate_analysis 和 generate_analysis_stream 复用）。

    Returns:
        消息列表，如果简历提取有错误则返回 None（调用方需处理错误）。
    """
    resume_data = state.get("resume_data") or {}
    working_context = state.get("working_context", "")
    messages = state.get("messages", [])

    # 提取用户最后一条消息
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # 检查简历提取是否出错
    if resume_data.get("extract_error"):
        return None

    # 序列化 resume_data（去掉 raw_text 避免超长）
    resume_for_prompt = {k: v for k, v in resume_data.items() if k != "raw_text"}
    resume_json = json.dumps(resume_for_prompt, ensure_ascii=False, indent=2)

    # JD 上下文（优先使用真实 JD，无则用知识库检索结果）
    jd_data = state.get("jd_data")
    if jd_data and isinstance(jd_data, dict) and not jd_data.get("extract_error") and working_context:
        jd_context = working_context
        logger.info("使用真实 JD 数据进行简历分析")
    elif working_context:
        jd_context = working_context
    else:
        jd_context = "（知识库中暂无匹配的岗位要求标准，将基于通用后端岗位标准进行评估）"

    # 构造 prompt
    prompt = RESUME_ANALYSIS_PROMPT.format(
        resume_data=resume_json,
        jd_context=jd_context,
        user_question=user_question or "请对我的简历进行全面分析评估",
    )

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_question or "请对我的简历进行全面分析评估"},
    ]


def generate_analysis(state: AgentState) -> dict:
    """
    简历分析生成节点（非流式）：基于结构化简历 + JD 参考 + 用户问题，生成分析报告。

    Returns:
        {"final_answer": str, "messages": [AIMessage]}
    """
    resume_data = state.get("resume_data") or {}
    messages = state.get("messages", [])

    # 提取用户最后一条消息
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # 检查简历提取是否出错
    if resume_data.get("extract_error"):
        error_msg = resume_data["extract_error"]
        logger.warning("简历提取有错误，直接返回错误信息")
        return {
            "final_answer": f"❌ 简历解析失败：{error_msg}\n\n请尝试：\n1. 重新上传简历文件\n2. 将简历内容粘贴到输入框中",
            "messages": [AIMessage(content=f"简历解析失败：{error_msg}")],
        }

    llm_messages = _build_analysis_messages(state)

    logger.info(
        "生成简历分析报告: resume=%s, 用户问题='%s'",
        resume_data.get("name", "未知"),
        user_question[:50] if user_question else "全面分析",
    )

    try:
        answer = chat_completion(
            llm_messages,
            temperature=0.5,
            max_tokens=4096,
        )
        logger.info("简历分析报告生成完成: %d 字符", len(answer))

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error("简历分析报告生成失败: %s", e, exc_info=True)
        return {
            "final_answer": f"❌ 简历分析报告生成失败：{e}",
            "messages": [AIMessage(content=f"分析报告生成失败：{e}")],
        }


async def generate_analysis_stream(state: AgentState) -> AsyncGenerator[dict[str, Any], None]:
    """
    简历分析流式生成节点：逐 token 生成分析报告。

    Yields:
        - {"type": "token", "content": "..."}  - 增量文本
        - {"type": "done", "final_answer": "...", "messages": [...]}  - 完整结果
        - {"type": "error", "message": "..."}   - 错误信息
    """
    resume_data = state.get("resume_data") or {}

    # 检查简历提取是否出错
    if resume_data.get("extract_error"):
        error_msg = resume_data["extract_error"]
        error_answer = f"❌ 简历解析失败：{error_msg}\n\n请尝试：\n1. 重新上传简历文件\n2. 将简历内容粘贴到输入框中"
        yield {
            "type": "done",
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }
        return

    llm_messages = _build_analysis_messages(state)
    if llm_messages is None:
        error_answer = "❌ 简历分析消息构造失败"
        yield {
            "type": "done",
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }
        return

    logger.info(
        "流式生成简历分析报告: resume=%s",
        resume_data.get("name", "未知"),
    )

    full_answer = ""
    try:
        async for delta in chat_completion_stream_async(
            llm_messages,
            temperature=0.5,
            max_tokens=4096,
        ):
            if delta:
                full_answer += delta
                yield {"type": "token", "content": delta}

        logger.info("简历分析流式生成完成: %d 字符", len(full_answer))

        yield {
            "type": "done",
            "final_answer": full_answer,
            "messages": [AIMessage(content=full_answer)],
        }

    except Exception as e:
        logger.error("简历分析流式生成失败: %s", e, exc_info=True)
        yield {"type": "error", "message": f"❌ 简历分析报告生成失败：{e}"}
