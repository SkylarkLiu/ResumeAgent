"""
QA Expert V1：复用现有 QA 路径能力的轻量子图。

检索降级机制：当 KB 检索结果为空或最高相关性分数低于阈值时，
自动降级到 web search，确保用户始终获得高质量回答。
"""
from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.generate import generate_streaming_node
from app.agent.nodes.kb_search import evaluate_kb_relevance, search_kb
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
    """dispatch 路由：根据 route_type 决定走 KB 检索、Web 搜索还是直接生成。"""
    route_type = state.get("route_type", "direct")
    route_str = route_type.value if hasattr(route_type, "value") else str(route_type)
    return route_str


def _evaluate_relevance_decision(state: AgentState) -> str:
    """检索质量评估路由：根据 retrieval_fallback 决定继续走 KB 还是降级到 Web。"""
    fallback = state.get("retrieval_fallback", False)
    if fallback:
        return "fallback"
    return "pass"


def _dispatch_node(_: AgentState) -> dict:
    return {}


def _search_kb_node(state: AgentState) -> dict:
    result = search_kb(state)
    _emit_custom_event({"type": "sources", "sources": result.get("context_sources", [])})
    return result


def _evaluate_relevance_node(state: AgentState) -> dict:
    """质量评估包装节点：调用 evaluate_kb_relevance 并发出 SSE 事件。"""
    result = evaluate_kb_relevance(state)
    if result.get("retrieval_fallback"):
        _emit_custom_event({
            "type": "status",
            "content": "知识库匹配度较低，降级搜索网络",
        })
    return result


def _search_web_fallback_node(state: AgentState) -> dict:
    """降级 Web 搜索节点：清除低质量的 KB 结果，替换为 Web 搜索结果。"""
    result = search_web(state)
    _emit_custom_event({"type": "sources", "sources": result.get("context_sources", [])})
    return result


def _search_web_node(state: AgentState) -> dict:
    result = search_web(state)
    _emit_custom_event({"type": "sources", "sources": result.get("context_sources", [])})
    return result


def _finalize_qa_output(state: AgentState) -> dict:
    final_answer = state.get("final_answer", "")
    context_sources = state.get("context_sources", [])
    retrieval_fallback = state.get("retrieval_fallback", False)
    return {
        "agent_outputs": {
            "qa_flow": {
                "summary": final_answer[:500],
                "final_answer": final_answer,
                "context_sources": context_sources,
                "retrieval_fallback": retrieval_fallback,
            }
        }
    }


def build_qa_flow_subgraph():
    """构建 QA Expert 子图（含 KB 检索降级机制）。

    拓扑：
        dispatch → search_kb → evaluate_relevance ──(pass)──→ normalize_kb → generate → finalize
                                                │
                                                └──(fallback)──→ search_web → normalize_web → generate
        dispatch → search_web → normalize_web → generate → finalize  (web 直达)
        dispatch → generate → finalize  (direct 直达)
    """
    builder = StateGraph(AgentState)
    builder.add_node("dispatch", _dispatch_node)
    builder.add_node("search_kb", _search_kb_node)
    builder.add_node("evaluate_relevance", _evaluate_relevance_node)
    builder.add_node("search_web", _search_web_node)
    builder.add_node("search_web_fallback", _search_web_fallback_node)
    builder.add_node("normalize_kb", normalize_kb)
    builder.add_node("normalize_web", normalize_web)
    builder.add_node("generate", generate_streaming_node)
    builder.add_node("finalize_qa", _finalize_qa_output)

    builder.add_edge(START, "dispatch")

    # dispatch 根据路由类型分发
    builder.add_conditional_edges(
        "dispatch",
        _qa_route_decision,
        {
            "retrieve": "search_kb",
            "web": "search_web",
            "direct": "generate",
        },
    )

    # ---- KB 路径：检索 → 质量评估 → 分流 ----
    builder.add_edge("search_kb", "evaluate_relevance")
    builder.add_conditional_edges(
        "evaluate_relevance",
        _evaluate_relevance_decision,
        {
            "pass": "normalize_kb",
            "fallback": "search_web_fallback",
        },
    )

    # KB 质量合格：normalize → generate
    builder.add_edge("normalize_kb", "generate")

    # KB 降级到 Web：search → normalize → generate
    builder.add_edge("search_web_fallback", "normalize_web")
    builder.add_edge("normalize_web", "generate")

    # Web 直达路径
    builder.add_edge("search_web", "normalize_web")

    # 生成 → 收尾
    builder.add_edge("generate", "finalize_qa")
    builder.add_edge("finalize_qa", END)

    return builder.compile()
