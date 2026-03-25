"""
简历分析报告生成节点 - 基于 resume_data + JD 上下文生成结构化分析报告

输入：state["resume_data"] + state["working_context"]（JD 检索结果）
输出：state["final_answer"]（Markdown 分析报告）+ state["messages"]
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.prompts import RESUME_ANALYSIS_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.generate_analysis")


def generate_analysis(state: AgentState) -> dict:
    """
    简历分析生成节点：基于结构化简历 + JD 参考 + 用户问题，生成分析报告。

    Returns:
        {"final_answer": str, "messages": [AIMessage]}
    """
    resume_data = state.get("resume_data") or {}
    working_context = state.get("working_context", "")
    messages = state.get("messages", [])

    # 提取用户最后一条消息（用户可能提出了具体问题）
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

    # 序列化 resume_data（去掉 raw_text 避免超长）
    resume_for_prompt = {k: v for k, v in resume_data.items() if k != "raw_text"}
    resume_json = json.dumps(resume_for_prompt, ensure_ascii=False, indent=2)

    # JD 上下文
    jd_context = working_context if working_context else "（知识库中暂无匹配的岗位要求标准，将基于通用后端岗位标准进行评估）"

    # 构造 prompt
    prompt = RESUME_ANALYSIS_PROMPT.format(
        resume_data=resume_json,
        jd_context=jd_context,
        user_question=user_question or "请对我的简历进行全面分析评估",
    )

    llm_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_question or "请对我的简历进行全面分析评估"},
    ]

    logger.info(
        "生成简历分析报告: resume=%s, JD上下文=%d字符, 用户问题='%s'",
        resume_data.get("name", "未知"),
        len(jd_context),
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
