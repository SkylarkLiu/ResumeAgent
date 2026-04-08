"""
JD 分析子图。
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.nodes.analyze_jd import analyze_jd_stream
from app.agent.nodes.extract_jd import extract_jd
from app.core.logger import setup_logger

logger = setup_logger("agent.subgraphs.jd_analysis")


class JDAnalysisState(TypedDict, total=False):
    """JD 分析子图关心的状态字段。"""

    messages: Annotated[list[BaseMessage], add_messages]
    jd_data: dict | None
    final_answer: str


def _emit_custom_event(payload: dict[str, Any]) -> None:
    """在 stream_mode=custom 下发送子图事件。"""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _extract_jd_node(state: JDAnalysisState) -> dict:
    """提取 JD 并发送 extracted 事件。"""
    existing_jd = state.get("jd_data") or {}
    has_structured_jd = any(
        existing_jd.get(key)
        for key in ("position", "summary", "skills_must", "responsibilities", "requirements")
    ) and not existing_jd.get("extract_error")

    if has_structured_jd:
        _emit_custom_event({"type": "status", "content": "正在复用已保存岗位信息"})
        summary = {k: v for k, v in existing_jd.items() if k != "raw_text"}
        _emit_custom_event({"type": "extracted", "jd_data": summary})
        return {"jd_data": existing_jd}

    _emit_custom_event({"type": "status", "content": "正在解析岗位描述"})
    result = extract_jd(state)
    jd_data = result.get("jd_data") or {}
    summary = {k: v for k, v in jd_data.items() if k != "raw_text"}
    _emit_custom_event({"type": "extracted", "jd_data": summary})
    return result


async def _stream_analyze_jd_node(state: JDAnalysisState) -> dict:
    """流式生成 JD 分析报告，并把 token/error 作为自定义事件发出。"""
    final_result: dict[str, Any] | None = None

    async for event in analyze_jd_stream(state):
        event_type = event.get("type")
        if event_type == "token":
            _emit_custom_event({"type": "token", "content": event.get("content", "")})
        elif event_type == "done":
            final_result = {
                "final_answer": event.get("final_answer", ""),
                "messages": event.get("messages", [AIMessage(content=event.get("final_answer", ""))]),
            }
        elif event_type == "error":
            message = event.get("message", "❌ JD 分析报告生成失败")
            _emit_custom_event({"type": "error", "message": message})
            raise RuntimeError(message)

    if final_result is None:
        logger.warning("JD 分析子图未收到 done 事件，返回兜底结果")
        final_result = {
            "final_answer": "❌ JD 分析未生成结果",
            "messages": [AIMessage(content="JD 分析未生成结果")],
        }
    return final_result


def build_jd_analysis_subgraph():
    """构建 JD 分析子图。"""
    builder = StateGraph(JDAnalysisState)
    builder.add_node("extract_jd", _extract_jd_node)
    builder.add_node("analyze_jd", _stream_analyze_jd_node)

    builder.add_edge(START, "extract_jd")
    builder.add_edge("extract_jd", "analyze_jd")
    builder.add_edge("analyze_jd", END)
    return builder.compile()
