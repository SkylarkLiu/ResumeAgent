"""
综合评估专家节点。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from app.agent.prompts import SUMMARY_EXPERT_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.summary")

_DIMENSION_LABELS = {
    "job_match": "岗位匹配度",
    "resume_expression": "简历表达力",
    "project_depth": "项目深度",
    "knowledge_mastery": "知识掌握度",
    "interview_performance": "面试表现",
}


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _latest_user_question(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
    return ""


def _parse_json_from_response(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    cleaned = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", '"').replace("\u2019", '"')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _average_interview_score(interview_data: dict[str, Any] | None) -> int:
    history = list((interview_data or {}).get("history") or [])
    scores = [int(item.get("score", 0) or 0) for item in history if item.get("score") is not None]
    if not scores:
        return 70
    return max(0, min(100, round(sum(scores) / len(scores))))


def _build_radar(dimension_scores: dict[str, Any]) -> list[dict[str, Any]]:
    radar = []
    for key, label in _DIMENSION_LABELS.items():
        score = int(dimension_scores.get(key, 70) or 70)
        radar.append({"key": key, "label": label, "score": max(0, min(100, score))})
    return radar


def _fallback_summary_data(state: AgentState) -> dict[str, Any]:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}
    interview_data = state.get("interview_data") or {}

    job_match = 80 if jd_data and resume_data else 68 if (jd_data or resume_data) else 60
    resume_expression = 76 if resume_data else 62
    project_depth = 78 if (resume_data.get("projects") or resume_data.get("experience")) else 64
    knowledge_mastery = _average_interview_score(interview_data) if interview_data.get("history") else 70
    interview_performance = _average_interview_score(interview_data) if interview_data.get("history") else 66
    overall_score = round((job_match + resume_expression + project_depth + knowledge_mastery + interview_performance) / 5)

    dimension_scores = {
        "job_match": job_match,
        "resume_expression": resume_expression,
        "project_depth": project_depth,
        "knowledge_mastery": knowledge_mastery,
        "interview_performance": interview_performance,
    }
    return {
        "overall_score": overall_score,
        "dimension_scores": dimension_scores,
        "radar": _build_radar(dimension_scores),
        "strengths": [
            "已具备较完整的岗位/简历上下文，可形成较稳定的综合评估",
            "项目经历与技术栈具备一定基础，适合继续强化表达与面试输出",
        ],
        "weaknesses": [
            "部分评分来自有限上下文，仍需结合更多实战问答验证",
            "知识掌握度与面试表现仍需通过针对性训练进一步拉开差距",
        ],
        "advice": [
            {"title": "优先补齐短板维度", "detail": "围绕得分最低的 1-2 个维度制定专项训练计划。", "priority": "high"},
            {"title": "强化项目表达", "detail": "用问题-方案-结果结构重写关键项目，补充指标与取舍。", "priority": "high"},
            {"title": "做针对性模拟面试", "detail": "围绕目标岗位继续做多轮问答，验证知识掌握与表达稳定性。", "priority": "medium"},
        ],
        "resources": [
            {"title": "项目复盘模板", "type": "practice", "reason": "帮助提升项目表达与面试复盘质量", "priority": "high"},
            {"title": "目标岗位知识清单", "type": "knowledge_base", "reason": "帮助补齐岗位匹配与知识短板", "priority": "high"},
            {"title": "模拟面试题单", "type": "practice", "reason": "用于持续验证回答质量和稳定性", "priority": "medium"},
        ],
        "radar_summary": "整体能力较均衡，但仍建议优先针对最低分维度做专项补强。",
        "final_verdict": "当前具备一定岗位竞争力，但仍需要通过系统化优化进一步提升通过率。",
        "next_steps": [
            "先补齐最低分维度的关键知识点与案例表达",
            "重写 1-2 个核心项目的 STAR/量化版本",
            "继续进行至少 1 轮针对性模拟面试验证改进效果",
        ],
    }


def _sanitize_summary_data(summary_data: dict[str, Any]) -> dict[str, Any]:
    data = dict(summary_data or {})
    dimension_scores = data.get("dimension_scores") or {}
    normalized_scores = {}
    for key in _DIMENSION_LABELS:
        normalized_scores[key] = max(0, min(100, int(dimension_scores.get(key, 70) or 70)))
    data["dimension_scores"] = normalized_scores
    data["overall_score"] = max(0, min(100, int(data.get("overall_score", round(sum(normalized_scores.values()) / len(normalized_scores))) or 0)))
    data["strengths"] = [str(item).strip() for item in (data.get("strengths") or []) if str(item).strip()][:4]
    data["weaknesses"] = [str(item).strip() for item in (data.get("weaknesses") or []) if str(item).strip()][:4]
    data["advice"] = [
        {
            "title": str(item.get("title", "")).strip() or "改进建议",
            "detail": str(item.get("detail", "")).strip(),
            "priority": str(item.get("priority", "medium")).strip() or "medium",
        }
        for item in (data.get("advice") or [])
        if isinstance(item, dict) and str(item.get("detail", "")).strip()
    ][:5]
    data["resources"] = [
        {
            "title": str(item.get("title", "")).strip() or "推荐资源",
            "type": str(item.get("type", "article")).strip() or "article",
            "reason": str(item.get("reason", "")).strip(),
            "priority": str(item.get("priority", "medium")).strip() or "medium",
        }
        for item in (data.get("resources") or [])
        if isinstance(item, dict) and str(item.get("title", "")).strip()
    ][:6]
    data["radar_summary"] = str(data.get("radar_summary", "")).strip()
    data["final_verdict"] = str(data.get("final_verdict", "")).strip()
    data["next_steps"] = [str(item).strip() for item in (data.get("next_steps") or []) if str(item).strip()][:5]
    data["radar"] = _build_radar(normalized_scores)
    return data


def _render_summary_answer(summary_data: dict[str, Any]) -> str:
    score = summary_data.get("overall_score", 0)
    radar_lines = "\n".join(
        f"- **{item['label']}**：{item['score']}/100"
        for item in summary_data.get("radar", [])
    )
    strengths = "\n".join(f"- {item}" for item in (summary_data.get("strengths") or [])) or "- 暂无"
    weaknesses = "\n".join(f"- {item}" for item in (summary_data.get("weaknesses") or [])) or "- 暂无"
    advice = "\n".join(
        f"- **{item.get('title', '建议')}**（{item.get('priority', 'medium')}）：{item.get('detail', '')}"
        for item in (summary_data.get("advice") or [])
    ) or "- 暂无"
    resources = "\n".join(
        f"- **{item.get('title', '资源')}**（{item.get('type', 'article')} / {item.get('priority', 'medium')}）：{item.get('reason', '')}"
        for item in (summary_data.get("resources") or [])
    ) or "- 暂无"
    next_steps = "\n".join(f"{idx}. {item}" for idx, item in enumerate(summary_data.get("next_steps") or [], 1)) or "1. 暂无"

    return (
        "## 综合评估报告\n\n"
        f"### 总体评分：**{score}/100**\n\n"
        f"**总评：** {summary_data.get('final_verdict', '当前整体表现中等偏上，建议继续针对性提升。')}\n\n"
        "### 能力雷达概览\n"
        f"{radar_lines}\n\n"
        f"**雷达图解读：** {summary_data.get('radar_summary', '整体能力存在一定短板，建议优先补齐最低分维度。')}\n\n"
        "### 核心优势\n"
        f"{strengths}\n\n"
        "### 关键短板\n"
        f"{weaknesses}\n\n"
        "### 改进建议\n"
        f"{advice}\n\n"
        "### 推荐资源\n"
        f"{resources}\n\n"
        "### 下一步行动清单\n"
        f"{next_steps}"
    )


async def _stream_text(text: str) -> None:
    chunk_size = 8
    for i in range(0, len(text), chunk_size):
        _emit_custom_event({"type": "token", "content": text[i:i + chunk_size]})
        await asyncio.sleep(0.008)


def _build_summary_messages(state: AgentState, question: str) -> list[dict[str, str]]:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}
    interview_data = state.get("interview_data") or {}
    agent_outputs = state.get("agent_outputs", {}) or {}
    summary_payload = {
        "jd_data": jd_data,
        "resume_data": resume_data,
        "interview_data": interview_data,
        "agent_outputs": {
            key: {
                "summary": value.get("summary", ""),
                "final_answer": value.get("final_answer", ""),
            }
            for key, value in agent_outputs.items()
            if key in {"jd_expert", "resume_expert", "interview_expert"}
        },
    }
    return [
        {"role": "system", "content": SUMMARY_EXPERT_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户要求：{question}\n\n"
                f"当前会话综合上下文：\n{json.dumps(summary_payload, ensure_ascii=False)}"
            ),
        },
    ]


async def summary_expert_node(state: AgentState) -> dict[str, Any]:
    question = _latest_user_question(state.get("messages", []))
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}
    interview_data = state.get("interview_data") or {}

    if not jd_data and not resume_data and not interview_data:
        answer = (
            "## 综合评估暂不可用\n\n"
            "当前会话里还没有足够的 JD、简历或模拟面试数据。"
            "建议先完成 **岗位分析**、**简历分析** 或 **模拟面试**，再生成综合评估报告。"
        )
        await _stream_text(answer)
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    _emit_custom_event({"type": "status", "content": "正在生成综合评估"})

    summary_data = None
    try:
        raw = chat_completion(
            _build_summary_messages(state, question or "请生成综合评估报告"),
            temperature=0.3,
            max_tokens=2200,
            thinking={"type": "disabled"},
        )
        parsed = _parse_json_from_response(raw)
        if parsed:
            summary_data = _sanitize_summary_data(parsed)
    except Exception as exc:
        logger.warning("总结专家结构化输出失败，使用 fallback: %s", exc)

    if not summary_data:
        summary_data = _sanitize_summary_data(_fallback_summary_data(state))

    _emit_custom_event({"type": "summary_data", "summary_data": summary_data})

    answer = _render_summary_answer(summary_data)
    await _stream_text(answer)
    return {
        "summary_data": summary_data,
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }
