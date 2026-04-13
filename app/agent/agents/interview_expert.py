"""
模拟面试专家节点。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from app.agent.prompts import INTERVIEW_EVALUATE_PROMPT, INTERVIEW_START_PROMPT, INTERVIEW_SUMMARY_PROMPT
from app.agent.state import AgentState
from app.agent.utils.history_utils import summarize_early_messages
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion, chat_completion_stream_async

logger = setup_logger("agent.interview")

_EXIT_INTERVIEW_KEYWORDS = (
    "结束模拟面试", "结束面试", "退出模拟面试", "停止面试", "先到这里",
    "结束", "不想面了", "不面了", "退出面试", "暂停面试",
)

_SUMMARY_KEYWORDS = ("总结", "复盘", "整体表现", "整体报告", "面试总结", "面试复盘", "总评")


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


def _build_dialogue_summary(messages: list[Any], limit: int = 6) -> str:
    """构建面试对话摘要，使用分层裁剪工具统一压缩逻辑。"""
    if not messages:
        return ""
    # 面试场景：只取最近 limit 条，压缩为摘要文本
    recent = messages[-limit:] if len(messages) > limit else messages
    return summarize_early_messages(recent, max_chars_per_msg=240)


def _infer_question_count(question: str) -> int:
    match = re.search(r"(\d+)\s*(?:题|道)", question)
    if match:
        try:
            return max(3, min(8, int(match.group(1))))
        except ValueError:
            return 5
    return 5


def _build_start_messages(state: AgentState, question: str) -> list[dict[str, str]]:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}
    dialogue = _build_dialogue_summary(state.get("messages", []))
    total_questions = _infer_question_count(question)
    user_prompt = (
        f"用户请求：{question}\n\n"
        f"目标题量：{total_questions}\n\n"
        f"JD 信息：\n{json.dumps(jd_data, ensure_ascii=False)}\n\n"
        f"简历信息：\n{json.dumps(resume_data, ensure_ascii=False)}\n\n"
        f"历史对话摘要：\n{dialogue or '（无）'}"
    )
    return [
        {"role": "system", "content": INTERVIEW_START_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_evaluate_messages(state: AgentState, question: str, interview_data: dict[str, Any]) -> list[dict[str, str]]:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}
    history = interview_data.get("history") or []
    current_question = interview_data.get("current_question", "")
    total_questions = int(interview_data.get("total_questions", 5) or 5)
    current_index = int(interview_data.get("question_index", 0) or 0)
    history_json = json.dumps(history[-4:], ensure_ascii=False, indent=2)
    user_prompt = (
        f"当前面试题：{current_question}\n\n"
        f"候选人本轮回答：{question}\n\n"
        f"当前题号：{current_index + 1}/{total_questions}\n\n"
        f"JD 信息：\n{json.dumps(jd_data, ensure_ascii=False)}\n\n"
        f"简历信息：\n{json.dumps(resume_data, ensure_ascii=False)}\n\n"
        f"已有面试记录：\n{history_json}"
    )
    return [
        {"role": "system", "content": INTERVIEW_EVALUATE_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_summary_messages(state: AgentState, interview_data: dict[str, Any]) -> list[dict[str, str]]:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}
    history = interview_data.get("history") or []

    history_lines = []
    for i, record in enumerate(history, 1):
        q = record.get("question", "")
        a = record.get("answer", "")
        s = record.get("score", "?")
        v = record.get("verdict", "")
        history_lines.append(f"第{i}题: {q}\n回答: {a[:200]}\n得分: {s}/100 | {v}")

    user_prompt = (
        f"面试历史记录：\n" + "\n\n".join(history_lines) + "\n\n"
        f"JD 信息：\n{json.dumps(jd_data, ensure_ascii=False)}\n\n"
        f"简历信息：\n{json.dumps(resume_data, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": INTERVIEW_SUMMARY_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _render_start_answer(opening: str, first_question: str, total_questions: int, focus: str) -> str:
    return (
        "## 模拟面试开始\n\n"
        f"{opening or '现在开始一轮模拟面试。'}\n\n"
        f"- 面试重点：**{focus or '岗位核心能力'}**\n"
        f"- 预计题量：**{total_questions} 题**\n\n"
        f"说「结束面试」即可停止。\n\n"
        f"### 第 1 题\n{first_question}"
    )


def _render_round_answer(score: int, verdict: str, strengths: list[str], gaps: list[str], advice: str, next_question: str, next_index: int, finished: bool) -> str:
    answer = (
        "## 本轮评分\n\n"
        f"- 分数：**{score}/100**\n"
        f"- 评价：{verdict}\n\n"
        "### 回答亮点\n"
        + ("\n".join(f"- {item}" for item in strengths) if strengths else "- 暂无明显亮点")
        + "\n\n### 主要不足\n"
        + ("\n".join(f"- {item}" for item in gaps) if gaps else "- 暂无明显不足")
        + f"\n\n### 改进建议\n{advice or '建议补充更具体的项目细节、指标和方案取舍。'}"
    )
    if finished or not next_question:
        return answer + "\n\n## 模拟面试结束\n本轮模拟面试已完成。说「总结面试」可以查看完整复盘报告，或说「重新面试」开始新一轮。"
    return answer + f"\n\n### 下一题（第 {next_index} 题）\n{next_question}"


async def _stream_text(text: str) -> None:
    """将完整文本以 token 事件逐段流式发送，模拟打字效果。"""
    chunk_size = 6
    for i in range(0, len(text), chunk_size):
        _emit_custom_event({"type": "token", "content": text[i:i + chunk_size]})
        await asyncio.sleep(0.01)


async def interview_expert_node(state: AgentState) -> dict[str, Any]:
    question = _latest_user_question(state.get("messages", []))
    interview_data = dict(state.get("interview_data") or {})

    if not question:
        answer = "❌ 未识别到有效问题，无法启动模拟面试。"
        _emit_custom_event({"type": "token", "content": answer})
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    # ---- 退出面试 ----
    if any(keyword in question for keyword in _EXIT_INTERVIEW_KEYWORDS):
        _emit_custom_event({"type": "status", "content": "模拟面试已结束"})
        history = list(interview_data.get("history") or [])

        if history:
            # 面试有记录，先退出，再流式生成结束语
            answer = "## 模拟面试已结束\n\n本轮模拟面试已手动结束。说「总结面试」可以查看完整复盘报告，或说「重新面试」开始新一轮。"
            await _stream_text(answer)
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
                "interview_data": {
                    **interview_data,
                    "active": False,
                },
            }
        else:
            answer = "## 模拟面试已结束\n\n尚未开始答题就退出了面试。需要时可以重新开始。"
            await _stream_text(answer)
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
                "interview_data": {
                    **interview_data,
                    "active": False,
                },
            }

    # ---- 面试总结（面试已结束，用户要求复盘） ----
    if not interview_data.get("active") and interview_data.get("history") and any(k in question for k in _SUMMARY_KEYWORDS):
        _emit_custom_event({"type": "status", "content": "正在生成面试复盘报告"})
        _emit_custom_event({"type": "interview_progress", "phase": "summary"})
        messages = _build_summary_messages(state, interview_data)

        # 流式生成总结
        collected = []
        async for delta in chat_completion_stream_async(
            messages,
            temperature=0.3,
            max_tokens=2000,
            thinking={"type": "disabled"},
        ):
            _emit_custom_event({"type": "token", "content": delta})
            collected.append(delta)

        answer = "".join(collected)
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    # ---- 面试结束后追问（如"那第3题怎么答"） ----
    if not interview_data.get("active") and interview_data.get("history"):
        _emit_custom_event({"type": "status", "content": "正在回答面试追问"})
        interview_history = list(interview_data.get("history") or [])
        history_text = "\n".join(
            f"第{i+1}题: {h.get('question', '')}\n候选人回答: {h.get('answer', '')[:200]}\n得分: {h.get('score', '?')}/100 | {h.get('verdict', '')}"
            for i, h in enumerate(interview_history)
        )
        followup_messages = [
            {"role": "system", "content": "你是一名技术面试复盘助手。面试已结束，用户正在追问面试相关话题。请基于面试历史记录简洁回答，默认不超过 300 字。不要暴露内部思考。"},
            {"role": "user", "content": f"面试历史：\n{history_text}\n\n用户追问：{question}"},
        ]

        collected = []
        async for delta in chat_completion_stream_async(
            followup_messages,
            temperature=0.3,
            max_tokens=800,
            thinking={"type": "disabled"},
        ):
            _emit_custom_event({"type": "token", "content": delta})
            collected.append(delta)

        answer = "".join(collected)
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    # ---- 开始新一轮模拟面试 ----
    if not interview_data.get("active"):
        _emit_custom_event({"type": "status", "content": "正在生成模拟面试题目"})
        _emit_custom_event({"type": "interview_progress", "phase": "start"})

        # 用 asyncio.to_thread 避免阻塞事件循环
        response = await asyncio.to_thread(
            chat_completion,
            _build_start_messages(state, question),
            temperature=0.4,
            max_tokens=1600,
            thinking={"type": "disabled"},
        )
        data = _parse_json_from_response(response) or {}
        if not data:
            logger.warning("面试出题 JSON 解析失败，原始响应: %s", response[:500])
        questions = [str(item).strip() for item in (data.get("questions") or []) if str(item).strip()]
        if not questions:
            questions = [
                "请你先介绍一个你最能体现 RAG 或 Agent 工程能力的项目，并说明你负责的关键模块。",
                "如果检索结果相关性不稳定，你会如何定位问题并改进召回效果？",
                "在多智能体或复杂工作流中，你如何避免循环调用、幻觉和状态污染？",
            ]
        total_questions = max(1, min(len(questions), int(data.get("total_questions") or len(questions))))
        questions = questions[:total_questions]
        first_question = questions[0]
        focus = str(data.get("focus") or "岗位核心能力").strip()
        opening = str(data.get("opening") or "现在开始模拟面试，请尽量结合真实项目经验作答。").strip()
        answer = _render_start_answer(opening, first_question, total_questions, focus)

        _emit_custom_event({
            "type": "interview_progress",
            "phase": "question",
            "question_index": 0,
            "total_questions": total_questions,
        })
        await _stream_text(answer)
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "interview_data": {
                "active": True,
                "focus": focus,
                "total_questions": total_questions,
                "question_index": 0,
                "current_question": first_question,
                "planned_questions": questions,
                "history": [],
            },
        }

    # ---- 面试追问 / 评分 ----
    _emit_custom_event({"type": "status", "content": "正在评估本轮回答"})

    response = await asyncio.to_thread(
        chat_completion,
        _build_evaluate_messages(state, question, interview_data),
        temperature=0.3,
        max_tokens=1400,
        thinking={"type": "disabled"},
    )
    data = _parse_json_from_response(response) or {}
    if not data:
        logger.warning("面试评分 JSON 解析失败，原始响应: %s", response[:500])
    score = max(0, min(100, int(data.get("score") or 0)))
    verdict = str(data.get("verdict") or "已完成本轮回答评估。").strip()
    strengths = [str(item).strip() for item in (data.get("strengths") or []) if str(item).strip()]
    gaps = [str(item).strip() for item in (data.get("gaps") or []) if str(item).strip()]
    advice = str(data.get("advice") or "").strip()
    current_index = int(interview_data.get("question_index", 0) or 0)
    planned_questions = list(interview_data.get("planned_questions") or [])
    total_questions = int(interview_data.get("total_questions", len(planned_questions) or 1) or 1)
    next_question = str(data.get("next_question") or "").strip()
    next_index = current_index + 1
    finished = bool(data.get("finished")) or next_index >= total_questions

    if not finished and not next_question and next_index < len(planned_questions):
        next_question = str(planned_questions[next_index]).strip()
    if not finished and not next_question:
        finished = True

    history = list(interview_data.get("history") or [])
    history.append(
        {
            "question": interview_data.get("current_question", ""),
            "answer": question,
            "score": score,
            "verdict": verdict,
        }
    )

    answer = _render_round_answer(score, verdict, strengths, gaps, advice, next_question, next_index + 1, finished)

    _emit_custom_event({
        "type": "interview_progress",
        "phase": "evaluated",
        "question_index": next_index if not finished else total_questions,
        "total_questions": total_questions,
        "current_score": score,
        "average_score": round(sum(h.get("score", 0) for h in history) / len(history), 1) if history else 0,
    })

    await _stream_text(answer)
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "interview_data": {
            **interview_data,
            "active": not finished,
            "question_index": next_index if not finished else total_questions,
            "current_question": next_question if not finished else "",
            "history": history,
        },
    }
