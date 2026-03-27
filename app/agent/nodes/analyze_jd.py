"""
JD 分析节点 - 基于 JD 结构化数据生成技术分析 + 简历写作建议

输入：state["jd_data"]（结构化 JD 信息）
输出：state["final_answer"]（Markdown 分析报告）+ state["messages"]
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.prompts import JD_ANALYSIS_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.analyze_jd")


def analyze_jd(state: AgentState) -> dict:
    """
    JD 分析节点：基于结构化 JD 数据，生成岗位技术分析和简历写作建议。

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

    # 序列化 jd_data（去掉 raw_text 避免超长）
    jd_for_prompt = {k: v for k, v in jd_data.items() if k not in ("raw_text", "extract_error")}
    jd_json = json.dumps(jd_for_prompt, ensure_ascii=False, indent=2)

    # 构造 prompt
    prompt = JD_ANALYSIS_PROMPT.format(jd_data=jd_json)

    llm_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_question or "请分析这个岗位的核心要求并给出简历写作建议"},
    ]

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
