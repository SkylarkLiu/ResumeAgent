"""
多 Agent V1 Supervisor：受控计划 + 最多 3 步执行。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from app.agent.nodes.router import route_query
from app.agent.prompts import MULTI_AGENT_FINAL_PROMPT
from app.agent.state import AgentState
from app.services.llm_service import chat_completion


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _normalize_agent_name(name: str | None) -> str | None:
    if name in {"qa_flow", "jd_expert", "resume_expert", "respond"}:
        return name
    return None


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
            return {
                "active_agent": active_agent,
                "max_steps": max_steps,
            }

    router_result = route_query(state, web_search_available=web_search_available)
    route_type = router_result.get("route_type", "direct")
    task_type = router_result.get("task_type", "qa")

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

    if task_type == "jd_analysis":
        execution_plan = ["jd_expert", "respond"]
    elif task_type == "resume_analysis":
        if _infer_resume_jd_chain(question, has_jd_data):
            execution_plan = ["resume_expert", "respond"]
        else:
            execution_plan = ["resume_expert", "respond"]
    else:
        execution_plan = ["qa_flow", "respond"]

    # 已有简历和 JD，且用户明显在做匹配分析时，优先走简历专家复用现有 jd_data
    if task_type == "qa" and has_jd_data and has_resume_data and _infer_resume_jd_chain(question, has_jd_data):
        execution_plan = ["resume_expert", "respond"]
        task_type = "resume_analysis"

    active_agent = _normalize_agent_name(execution_plan[0] if execution_plan else "respond") or "respond"
    _emit_custom_event({"type": "agent_start", "agent": active_agent})

    return {
        "route_type": route_type,
        "task_type": task_type,
        "execution_plan": execution_plan,
        "current_step": 0,
        "max_steps": max_steps,
        "active_agent": active_agent,
        "final_response_ready": active_agent == "respond",
        "agent_outputs": state.get("agent_outputs", {}),
        "resume_data": resume_data if resume_data else ({"raw_text": question} if task_type == "resume_analysis" and question else None),
        "jd_data": jd_data if jd_data else ({"raw_text": question} if task_type == "jd_analysis" and question else None),
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

    if active_agent in {"qa_flow", "jd_expert", "resume_expert"}:
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

        agent_outputs[active_agent] = summary_payload
        _emit_custom_event({"type": "agent_result", "agent": active_agent})

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
    execution_plan = [name for name in state.get("execution_plan", []) if name in {"qa_flow", "jd_expert", "resume_expert"}]

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

    answer = chat_completion(
        summary_messages,
        temperature=0.3,
        max_tokens=2048,
        thinking={"type": "disabled"},
    )
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
