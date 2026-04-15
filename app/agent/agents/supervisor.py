"""
多 Agent V1 Supervisor：受控计划 + 最多 3 步执行。
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from app.agent.prompts import MULTI_AGENT_FINAL_PROMPT, ROUTER_SYSTEM_PROMPT
from app.agent.state import AgentState, RouteDecision, RouteType, TaskType
from app.core.logger import setup_logger
from app.core.observation import log_request_decision
from app.services.llm_service import chat_completion

logger = setup_logger("agent.supervisor")


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _normalize_agent_name(name: str | None) -> str | None:
    if name in {"qa_flow", "jd_expert", "resume_expert", "interview_expert", "summary_expert", "react_fallback", "respond"}:
        return name
    return None


AGENT_STATUS_TEXT: dict[str, str] = {
    "qa_flow": "正在检索知识库",
    "jd_expert": "正在分析岗位描述",
    "resume_expert": "正在评估简历",
    "interview_expert": "正在进行模拟面试",
    "summary_expert": "正在生成综合评估",
    "react_fallback": "正在组合工具处理非标准请求",
}

_MAX_ROUTER_RETRIES = 1


def _latest_user_question(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content.strip() if isinstance(msg.content, str) else ""
            if content:
                return content
    return ""


def _build_state_summary(state: AgentState) -> str:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}

    parts: list[str] = []
    parts.append(f"has_jd_data={bool(jd_data)}")
    parts.append(f"has_resume_data={bool(resume_data)}")

    if jd_data.get("position"):
        parts.append(f"jd_position={jd_data.get('position')}")
    if jd_data.get("summary"):
        parts.append(f"jd_summary={str(jd_data.get('summary'))[:120]}")

    if resume_data.get("target_position"):
        parts.append(f"resume_target_position={resume_data.get('target_position')}")
    if resume_data.get("summary"):
        parts.append(f"resume_summary={str(resume_data.get('summary'))[:120]}")

    return "\n".join(parts)


def _is_resume_like_text(question: str) -> bool:
    q = question.lower()
    if len(question) >= 250:
        resume_markers = (
            "教育经历",
            "工作经历",
            "项目经历",
            "专业技能",
            "教育背景",
            "自我评价",
            "实习经历",
            "校内经历",
            "个人信息",
            "姓名",
            "电话",
            "邮箱",
            "毕业院校",
            "技能栈",
        )
        hits = sum(1 for marker in resume_markers if marker in question)
        if hits >= 2:
            return True
    compact_markers = ("简历如下", "这是我的简历", "下面是我的简历", "我的简历内容", "帮我看下简历")
    return any(marker in q for marker in compact_markers)


def _rule_based_followup_route(question: str, state: AgentState, web_search_available: bool) -> dict | None:
    q = question.lower()
    has_jd_data = bool(state.get("jd_data"))
    has_resume_data = bool(state.get("resume_data"))
    interview_data = state.get("interview_data") or {}

    jd_followup_keywords = (
        "关于以上jd", "关于以上 jd", "关于这个jd", "关于这个 jd", "基于以上jd", "基于以上 jd",
        "根据这个jd", "根据这个 jd", "这个岗位", "该岗位", "这个jd", "这个 jd",
        "面试准备", "面试阶段", "岗位重点", "岗位要求", "技术栈",
    )
    resume_keywords = ("简历", "评估", "优化", "修改简历", "改简历", "简历建议", "润色", "重写")
    match_keywords = ("匹配度", "匹配", "是否匹配", "缺少什么", "差距", "缺口", "对比jd", "对比 jd", "改进什么", "补什么", "最该补", "最需要改进")
    interview_keywords = ("模拟面试", "面试官", "开始面试", "mock interview", "面试题", "继续面试", "下一题", "请出题")
    interview_summary_keywords = ("总结面试", "面试总结", "复盘", "面试复盘", "整体表现", "整体报告", "总评")
    summary_keywords = ("综合评价", "综合评估", "综合打分", "能力雷达", "能力雷达图", "雷达图", "推荐资源", "总结专家", "整体能力评估", "生成总评")
    latest_keywords = ("最新", "今天", "本周", "最近", "2026", "实时", "新闻")

    # 面试进行中 → 追问
    if interview_data.get("active"):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.INTERVIEW_FOLLOWUP.value,
            "planning_reason": "当前会话处于模拟面试中，按面试追问处理",
            "question_signature": "interview_followup:active",
            "response_mode": "interview_round",
        }

    # 面试已结束但有历史 → 用户要求总结/复盘 → 走 interview_expert 的总结路径
    interview_history = interview_data.get("history") or []
    if not interview_data.get("active") and interview_history and any(k in q for k in interview_summary_keywords):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.INTERVIEW_SIMULATION.value,
            "planning_reason": "面试已结束，用户要求总结复盘",
            "question_signature": "interview:summary",
            "response_mode": "interview_round",
        }

    # 面试已结束但有历史 → 用户追问面试相关话题（如"第3题怎么答"）
    interview_topic_keywords = ("那道题", "第.*题", "那题", "这道题", "刚才的题", "那个问题", "面试", "interview")
    if not interview_data.get("active") and interview_history and any(re.search(k, q) for k in interview_topic_keywords):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.INTERVIEW_FOLLOWUP.value,
            "planning_reason": "面试已结束，用户追问面试相关话题",
            "question_signature": "interview_followup:post",
            "response_mode": "interview_round",
        }

    if any(k in q for k in interview_keywords):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.INTERVIEW_SIMULATION.value,
            "planning_reason": "用户明确要求进入模拟面试或继续面试",
            "question_signature": "interview:start",
            "response_mode": "interview_round",
        }

    if any(k in q for k in summary_keywords) and (has_jd_data or has_resume_data or interview_history):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.SUMMARY_ASSESSMENT.value,
            "planning_reason": "用户要求结合当前 JD、简历或面试结果做综合评估",
            "question_signature": "summary_assessment:general",
            "response_mode": "summary_report",
        }

    if has_jd_data and (_is_resume_like_text(question) or (has_resume_data and any(k in q for k in match_keywords))):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.MATCH_FOLLOWUP.value if has_resume_data else TaskType.RESUME_ANALYSIS.value,
            "planning_reason": "已有 JD 与简历数据，当前问题明显是匹配差距/补强项追问",
            "question_signature": "match_followup:generic",
            "response_mode": "match_brief" if has_resume_data else "full_report",
        }
    if has_resume_data and any(k in q for k in resume_keywords):
        signature = "resume_followup:optimize"
        if any(k in q for k in ("亮点", "优势", "突出")):
            signature = "resume_followup:strengths"
        elif any(k in q for k in ("怎么写", "如何写")):
            signature = "resume_followup:writing"
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.RESUME_FOLLOWUP.value,
            "planning_reason": "已有简历数据，当前问题是简历优化/表达层面的追问",
            "question_signature": signature,
            "response_mode": "followup_brief",
        }
    if has_jd_data and any(k in q for k in jd_followup_keywords):
        signature = "jd_followup:generic"
        if "面试" in q:
            signature = "jd_followup:interview"
        elif any(k in q for k in ("技术栈", "技能", "能力")):
            signature = "jd_followup:skills"
        elif any(k in q for k in ("优先", "重点", "最需要")):
            signature = "jd_followup:priority"
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.JD_FOLLOWUP.value,
            "planning_reason": "已有 JD 数据，当前问题是岗位追问",
            "question_signature": signature,
            "response_mode": "followup_brief",
        }
    if any(k in q for k in latest_keywords):
        return {
            "route_type": RouteType.WEB.value if web_search_available else RouteType.RETRIEVE.value,
            "task_type": TaskType.QA.value,
            "planning_reason": "问题具有明显时效性，需要检索或联网",
            "question_signature": "qa:timely",
            "response_mode": "direct_answer",
        }
    return None


def _should_use_react_fallback(question: str, state: AgentState) -> bool:
    q = question.lower()
    combined_keywords = ("结合", "顺便", "同时", "综合", "再根据", "然后", "先", "一并", "一起")
    has_jd_data = bool(state.get("jd_data"))
    has_resume_data = bool(state.get("resume_data"))
    if any(keyword in q for keyword in combined_keywords):
        return True
    if "知识库" in question and (has_jd_data or has_resume_data):
        return True
    if "联网" in question and ("分析" in question or "总结" in question):
        return True
    return False


def _parse_json_from_response(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    text_clean = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", '"').replace("\u2019", '"')
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _classify_task(state: AgentState, *, web_search_available: bool) -> dict[str, str]:
    question = _latest_user_question(state.get("messages", []))
    if not question:
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.QA.value,
            "planning_reason": "未识别到有效用户问题，回退到直接回答",
            "question_signature": "qa:empty",
            "response_mode": "direct_answer",
        }

    rule_result = _rule_based_followup_route(question, state, web_search_available)
    if rule_result is not None:
        logger.info(
            "Supervisor 规则命中: route=%s, task=%s, signature=%s, mode=%s, question=%s",
            rule_result["route_type"],
            rule_result["task_type"],
            rule_result["question_signature"],
            rule_result["response_mode"],
            question[:80],
        )
        return rule_result

    if _should_use_react_fallback(question, state):
        logger.info("Supervisor 触发 react_fallback: question=%s", question[:80])
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.REACT_FALLBACK.value,
            "planning_reason": "问题包含明显的组合式或非标准请求，进入受控 ReAct fallback",
            "question_signature": "react_fallback:generic",
            "response_mode": "react_fallback",
        }

    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"当前用户问题：\n{question}\n\n"
                f"结构化状态摘要：\n{_build_state_summary(state)}\n\n"
                "请只输出一行 JSON。"
            ),
        },
    ]

    last_error = None
    for attempt in range(1 + _MAX_ROUTER_RETRIES):
        try:
            response = chat_completion(
                router_messages,
                temperature=0,
                max_tokens=512,
                thinking={"type": "disabled"},
            )
            data = _parse_json_from_response(response or "")
            if data is None:
                raise ValueError(f"无法从响应中提取 JSON，原始响应前200字符: {repr((response or '')[:200])}")

            decision = RouteDecision.model_validate(data)
            route_type = decision.route_type
            if route_type == RouteType.WEB and not web_search_available:
                route_type = RouteType.RETRIEVE

            logger.info(
                "Supervisor 任务分类: route=%s, task=%s, signature=%s, mode=%s, reasoning=%s",
                route_type.value,
                decision.task_type.value,
                f"{decision.task_type.value}:llm",
                "direct_answer" if decision.task_type == TaskType.QA else ("summary_report" if decision.task_type == TaskType.SUMMARY_ASSESSMENT else ("full_report" if decision.task_type in {TaskType.RESUME_ANALYSIS, TaskType.JD_ANALYSIS} else "followup_brief")),
                decision.reasoning[:100],
            )
            return {
                "route_type": route_type.value,
                "task_type": decision.task_type.value,
                "planning_reason": decision.reasoning,
                "question_signature": f"{decision.task_type.value}:llm",
                "response_mode": "direct_answer" if decision.task_type == TaskType.QA else ("summary_report" if decision.task_type == TaskType.SUMMARY_ASSESSMENT else ("full_report" if decision.task_type in {TaskType.RESUME_ANALYSIS, TaskType.JD_ANALYSIS} else "followup_brief")),
            }
        except Exception as e:
            last_error = e
            if attempt < _MAX_ROUTER_RETRIES:
                logger.warning("Supervisor 分类失败 (attempt=%d/%d), 重试中: %s", attempt + 1, 1 + _MAX_ROUTER_RETRIES, e)

    logger.warning("Supervisor 分类最终 fallback 到 react_fallback: %s", last_error)
    return {
        "route_type": RouteType.DIRECT.value,
        "task_type": TaskType.REACT_FALLBACK.value,
        "planning_reason": f"分类失败后回退到受控 ReAct fallback: {last_error}",
        "question_signature": "react_fallback:fallback",
        "response_mode": "react_fallback",
    }


def _infer_resume_jd_chain(question: str, has_jd_data: bool) -> bool:
    q = question.lower()
    keywords = ("匹配", "match", "匹配度", "对比", "结合岗位", "结合jd", "结合 jd", "目标岗位")
    return has_jd_data and any(k in q for k in keywords)


def supervisor_plan_node(state: AgentState, *, web_search_available: bool = False) -> dict:
    """
    计划下一步执行哪个专家。

    V1 仅支持受控计划，不做开放式 ReAct：
    - qa_flow -> respond
    - jd_expert -> respond
    - resume_expert -> respond
    - jd_expert -> resume_expert -> respond （当已存在 jd_data 且当前问题是简历评估/匹配）
    """
    execution_plan = list(state.get("execution_plan", []))
    current_step = int(state.get("current_step", 0) or 0)
    max_steps = int(state.get("max_steps", 3) or 3)

    if execution_plan and current_step < len(execution_plan):
        active_agent = _normalize_agent_name(execution_plan[current_step])
        if active_agent:
            _emit_custom_event({"type": "agent_start", "agent": active_agent})
            status_text = AGENT_STATUS_TEXT.get(active_agent, "")
            if status_text:
                _emit_custom_event({"type": "status", "content": status_text})
            return {
                "active_agent": active_agent,
                "max_steps": max_steps,
            }

    router_result = _classify_task(state, web_search_available=web_search_available)
    route_type = router_result.get("route_type", "direct")
    task_type = router_result.get("task_type", "qa")
    planning_reason = router_result.get("planning_reason", "")
    question_signature = router_result.get("question_signature", "")
    response_mode = router_result.get("response_mode", "direct_answer")

    messages = state.get("messages", [])
    question = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            question = content.strip()
            break

    has_jd_data = bool(state.get("jd_data"))
    has_resume_data = bool(state.get("resume_data"))

    resume_data = state.get("resume_data")
    jd_data = state.get("jd_data")

    if task_type in {"jd_analysis", "jd_followup"}:
        execution_plan = ["jd_expert", "respond"]
    elif task_type in {"resume_analysis", "resume_followup", "match_followup"}:
        execution_plan = ["resume_expert", "respond"]
    elif task_type in {"interview_simulation", "interview_followup"}:
        execution_plan = ["interview_expert", "respond"]
    elif task_type == "summary_assessment":
        execution_plan = ["summary_expert", "respond"]
    elif task_type == "react_fallback":
        execution_plan = ["react_fallback", "respond"]
    else:
        execution_plan = ["qa_flow", "respond"]

    # 已有简历和 JD，且用户明显在做匹配分析时，优先走简历专家复用现有 jd_data
    if task_type == "qa" and has_jd_data and has_resume_data and _infer_resume_jd_chain(question, has_jd_data):
        execution_plan = ["resume_expert", "respond"]
        task_type = "match_followup"
        planning_reason = "已有 JD 与简历数据，按匹配追问处理"
        question_signature = "match_followup:generic"
        response_mode = "match_brief"

    active_agent = _normalize_agent_name(execution_plan[0] if execution_plan else "respond") or "respond"
    _emit_custom_event({
        "type": "planning",
        "task_type": task_type,
        "route_type": route_type,
        "question_signature": question_signature,
        "response_mode": response_mode,
        "planning_reason": planning_reason,
    })
    _emit_custom_event({"type": "agent_start", "agent": active_agent})
    status_text = AGENT_STATUS_TEXT.get(active_agent, "")
    if status_text:
        _emit_custom_event({"type": "status", "content": status_text})

    # 可观测性：统一打印路由决策日志
    log_request_decision(
        {
            "session_id": state.get("session_id", ""),
            "task_type": task_type,
            "question_signature": question_signature,
            "response_mode": response_mode,
        },
    )

    return {
        "route_type": route_type,
        "task_type": task_type,
        "planning_reason": planning_reason,
        "question_signature": question_signature,
        "response_mode": response_mode,
        "execution_plan": execution_plan,
        "current_step": 0,
        "max_steps": max_steps,
        "active_agent": active_agent,
        "final_response_ready": active_agent == "respond",
        "agent_outputs": state.get("agent_outputs", {}),
        "tool_trace": state.get("tool_trace", []),
        "react_iterations": int(state.get("react_iterations", 0) or 0),
        "resume_data": resume_data if resume_data else ({"raw_text": question} if task_type in {"resume_analysis", "resume_followup", "match_followup"} and question else None),
        "jd_data": jd_data if jd_data else ({"raw_text": question} if task_type in {"jd_analysis", "jd_followup"} and question else None),
        "summary_data": state.get("summary_data"),
    }


def supervisor_plan_route(state: AgentState) -> str:
    return _normalize_agent_name(state.get("active_agent")) or "respond"


def supervisor_review_node(state: AgentState) -> dict:
    """记录当前专家已完成，并决定是否继续。"""
    active_agent = _normalize_agent_name(state.get("active_agent"))
    execution_plan = list(state.get("execution_plan", []))
    current_step = int(state.get("current_step", 0) or 0)
    max_steps = int(state.get("max_steps", 3) or 3)
    next_step = current_step + 1
    final_answer = state.get("final_answer", "")
    agent_outputs = dict(state.get("agent_outputs", {}) or {})
    handoff_agent = _normalize_agent_name(state.get("react_handoff_agent"))

    if active_agent in {"qa_flow", "jd_expert", "resume_expert", "interview_expert", "summary_expert", "react_fallback"}:
        summary_payload: dict[str, Any] = {
            "summary": final_answer[:500],
            "final_answer": final_answer,
        }
        if active_agent == "qa_flow":
            summary_payload["context_sources"] = state.get("context_sources", [])
        elif active_agent == "jd_expert":
            summary_payload["jd_data"] = state.get("jd_data")
        elif active_agent == "resume_expert":
            summary_payload["resume_data"] = state.get("resume_data")
            summary_payload["jd_data"] = state.get("jd_data")
        elif active_agent == "interview_expert":
            summary_payload["interview_data"] = state.get("interview_data")
            summary_payload["jd_data"] = state.get("jd_data")
            summary_payload["resume_data"] = state.get("resume_data")
        elif active_agent == "summary_expert":
            summary_payload["summary_data"] = state.get("summary_data")
            summary_payload["interview_data"] = state.get("interview_data")
            summary_payload["jd_data"] = state.get("jd_data")
            summary_payload["resume_data"] = state.get("resume_data")
        elif active_agent == "react_fallback":
            summary_payload["context_sources"] = state.get("context_sources", [])
            summary_payload["tool_trace"] = state.get("tool_trace", [])
            summary_payload["jd_data"] = state.get("jd_data")
            summary_payload["resume_data"] = state.get("resume_data")

        agent_outputs[active_agent] = summary_payload
        _emit_custom_event({"type": "agent_result", "agent": active_agent})

    if active_agent == "react_fallback" and handoff_agent in {"qa_flow", "jd_expert", "resume_expert"}:
        return {
            "agent_outputs": agent_outputs,
            "execution_plan": [handoff_agent, "respond"],
            "current_step": 0,
            "active_agent": None,
            "final_response_ready": False,
            "react_handoff_agent": None,
        }

    final_response_ready = next_step >= len(execution_plan) or next_step >= max_steps
    next_agent = "respond" if final_response_ready else _normalize_agent_name(execution_plan[next_step]) or "respond"

    return {
        "agent_outputs": agent_outputs,
        "current_step": next_step,
        "active_agent": next_agent,
        "final_response_ready": final_response_ready or next_agent == "respond",
    }


def supervisor_review_route(state: AgentState) -> str:
    if state.get("final_response_ready"):
        return "respond"
    return "continue"


def generate_final_node(state: AgentState) -> dict:
    """
    V1 最终汇总：
    - 单专家时直接复用已有 final_answer
    - 多专家时做一次轻量综合
    """
    final_answer = state.get("final_answer", "")
    agent_outputs = state.get("agent_outputs", {}) or {}
    execution_plan = [name for name in state.get("execution_plan", []) if name in {"qa_flow", "jd_expert", "resume_expert", "interview_expert", "summary_expert", "react_fallback"}]

    if len(execution_plan) <= 1:
        if final_answer:
            return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}

        for agent_name in reversed(execution_plan):
            if agent_name in agent_outputs:
                candidate = agent_outputs[agent_name].get("final_answer") or agent_outputs[agent_name].get("summary")
                if candidate:
                    return {"final_answer": candidate, "messages": [AIMessage(content=candidate)]}
        return {"final_answer": "", "messages": [AIMessage(content="")]}

    question = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str) and msg.content.strip():
            question = msg.content.strip()
            break

    expert_blocks = []
    for agent_name in execution_plan:
        payload = agent_outputs.get(agent_name)
        if not payload:
            continue
        answer = payload.get("final_answer") or payload.get("summary") or ""
        if not answer:
            continue
        label = {
            "qa_flow": "QA 专家",
            "jd_expert": "JD 专家",
            "resume_expert": "简历专家",
            "interview_expert": "面试专家",
            "summary_expert": "总结专家",
            "react_fallback": "ReAct 兜底",
        }.get(agent_name, agent_name)
        expert_blocks.append(f"## {label}\n{answer}")

    if not expert_blocks:
        return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}

    summary_messages = [
        {"role": "system", "content": MULTI_AGENT_FINAL_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：\n{question}\n\n"
                f"专家结果：\n\n" + "\n\n".join(expert_blocks)
            ),
        },
    ]

    task_type = str(state.get("task_type", ""))
    max_tokens = 900 if task_type in {"jd_followup", "resume_followup", "match_followup"} else 2048
    answer = chat_completion(
        summary_messages,
        temperature=0.3,
        max_tokens=max_tokens,
        thinking={"type": "disabled"},
    )
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
