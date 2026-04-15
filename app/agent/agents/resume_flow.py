"""
简历分析流程。

由 app/agent/subgraphs/resume_analysis.py 迁移而来，
节点名、事件格式、函数签名保持不变，仅调整模块位置与 logger 命名。
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.nodes.extract_resume import extract_resume
from app.agent.nodes.generate_analysis import generate_analysis_stream
from app.agent.nodes.retrieve_jd import retrieve_jd
from app.core.logger import setup_logger

logger = setup_logger("agent.resume_flow")


class ResumeAnalysisState(TypedDict, total=False):
    """简历分析流程关心的状态字段。"""

    messages: Annotated[list[BaseMessage], add_messages]
    context_sources: list[dict]
    working_context: str
    resume_data: dict | None
    jd_data: dict | None
    final_answer: str


def _emit_custom_event(payload: dict[str, Any]) -> None:
    """在 stream_mode=custom 下发送流程事件。"""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _extract_resume_node(state: ResumeAnalysisState) -> dict:
    """提取简历并发送 extracted 事件。"""
    existing_resume = state.get("resume_data") or {}
    has_structured_resume = any(
        existing_resume.get(key)
        for key in ("summary", "skills", "projects", "experience", "education", "target_position", "name")
    ) and not existing_resume.get("extract_error")

    if has_structured_resume:
        _emit_custom_event({"type": "status", "content": "正在读取已保存简历"})
        summary = {k: v for k, v in existing_resume.items() if k != "raw_text"}
        _emit_custom_event({"type": "extracted", "resume_data": summary})
        logger.info("复用已有结构化简历数据，跳过重复提取")
        return {"resume_data": existing_resume}

    result = extract_resume(state)
    resume_data = result.get("resume_data") or {}
    summary = {k: v for k, v in resume_data.items() if k != "raw_text"}
    _emit_custom_event({"type": "extracted", "resume_data": summary})
    return result


def _resolve_jd_context_node(state: ResumeAnalysisState) -> dict:
    """解析 JD 上下文并发送 sources 事件。"""
    existing_sources = state.get("context_sources") or []
    existing_context = state.get("working_context", "")
    jd_data = state.get("jd_data") or {}
    has_structured_jd = any(
        jd_data.get(key)
        for key in ("position", "summary", "skills_must", "responsibilities", "requirements")
    ) and not jd_data.get("extract_error")

    if has_structured_jd and existing_sources and existing_context:
        _emit_custom_event({"type": "status", "content": "正在复用已保存岗位上下文"})
        _emit_custom_event({"type": "sources", "sources": existing_sources})
        logger.info("复用已有 JD 上下文，跳过重复解析")
        return {
            "context_sources": existing_sources,
            "working_context": existing_context,
        }

    result = retrieve_jd(state)
    _emit_custom_event({"type": "sources", "sources": result.get("context_sources", [])})
    return result


async def _stream_generate_analysis_node(state: ResumeAnalysisState) -> dict:
    """流式生成简历分析报告，并把 token/error 作为自定义事件发出。"""
    final_result: dict[str, Any] | None = None

    async for event in generate_analysis_stream(state):
        event_type = event.get("type")
        if event_type == "token":
            _emit_custom_event({"type": "token", "content": event.get("content", "")})
        elif event_type == "done":
            final_result = {
                "final_answer": event.get("final_answer", ""),
                "messages": event.get("messages", [AIMessage(content=event.get("final_answer", ""))]),
            }
        elif event_type == "error":
            message = event.get("message", "❌ 简历分析报告生成失败")
            _emit_custom_event({"type": "error", "message": message})
            raise RuntimeError(message)

    if final_result is None:
        logger.warning("简历分析流程未收到 done 事件，返回兜底结果")
        final_result = {
            "final_answer": "❌ 简历分析未生成结果",
            "messages": [AIMessage(content="简历分析未生成结果")],
        }
    return final_result


def build_resume_analysis_flow():
    """构建简历分析流程。"""
    builder = StateGraph(ResumeAnalysisState)
    builder.add_node("extract_resume", _extract_resume_node)
    builder.add_node("resolve_jd_context", _resolve_jd_context_node)
    builder.add_node("generate_analysis", _stream_generate_analysis_node)

    builder.add_edge(START, "extract_resume")
    builder.add_edge("extract_resume", "resolve_jd_context")
    builder.add_edge("resolve_jd_context", "generate_analysis")
    builder.add_edge("generate_analysis", END)
    return builder.compile()


# ── 兼容旧调用方 ──
build_resume_analysis_subgraph = build_resume_analysis_flow
