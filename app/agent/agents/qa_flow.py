"""
QA Expert V1：复用现有 QA 路径能力的轻量子图。
"""
from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.generate import generate_streaming_node
from app.agent.nodes.kb_search import search_kb
from app.agent.nodes.normalize import normalize_kb, normalize_web
from app.agent.nodes.web_search import search_web
from app.agent.state import AgentState


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _qa_route_decision(state: AgentState) -> str:
    route_type = state.get("route_type", "direct")
    route_str = route_type.value if hasattr(route_type, "value") else str(route_type)
    return route_str


def _dispatch_node(_: AgentState) -> dict:
    return {}


def _search_kb_node(state: AgentState) -> dict:
    result = search_kb(state)
    _emit_custom_event({"type": "sources", "sources": result.get("context_sources", [])})
    return result


def _search_web_node(state: AgentState) -> dict:
    result = search_web(state)
    _emit_custom_event({"type": "sources", "sources": result.get("context_sources", [])})
    return result


def _finalize_qa_output(state: AgentState) -> dict:
    final_answer = state.get("final_answer", "")
    context_sources = state.get("context_sources", [])
    return {
        "agent_outputs": {
            "qa_flow": {
                "summary": final_answer[:500],
                "final_answer": final_answer,
                "context_sources": context_sources,
            }
        }
    }


def build_qa_flow_subgraph():
    """构建 QA Expert 子图。"""
    builder = StateGraph(AgentState)
    builder.add_node("dispatch", _dispatch_node)
    builder.add_node("search_kb", _search_kb_node)
    builder.add_node("search_web", _search_web_node)
    builder.add_node("normalize_kb", normalize_kb)
    builder.add_node("normalize_web", normalize_web)
    builder.add_node("generate", generate_streaming_node)
    builder.add_node("finalize_qa", _finalize_qa_output)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges(
        "dispatch",
        _qa_route_decision,
        {
            "retrieve": "search_kb",
            "web": "search_web",
            "direct": "generate",
        },
    )
    builder.add_edge("search_kb", "normalize_kb")
    builder.add_edge("normalize_kb", "generate")
    builder.add_edge("search_web", "normalize_web")
    builder.add_edge("normalize_web", "generate")
    builder.add_edge("generate", "finalize_qa")
    builder.add_edge("finalize_qa", END)
    return builder.compile()
