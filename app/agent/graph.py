"""
Agent 主图 - StateGraph 定义 + 编译 + Checkpointer 管理

阶段 3 改造：新增简历分析子图路径
- 条件边同时判断 task_type + route_type
- resume_analysis 走: extract_resume → retrieve_jd → generate_analysis
- qa 走现有路径: kb/web/direct → normalize → generate

阶段 4 改造：新增 JD 分析路径
- jd_analysis 走: extract_jd → analyze_jd
- JD 数据存入 session state，后续简历分析可复用
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.checkpointer import get_checkpointer
from app.agent.nodes.generate import generate_streaming_node, set_max_history
from app.agent.nodes.kb_search import search_kb, set_retrieval_service
from app.agent.nodes.normalize import normalize_kb, normalize_web, set_max_chars
from app.agent.nodes.retrieve_jd import set_retrieval_service_jd
from app.agent.nodes.router import route_query
from app.agent.nodes.web_search import search_web, set_web_search_service
from app.agent.state import AgentState, RouteType, TaskType
from app.agent.subgraphs import build_jd_analysis_subgraph, build_resume_analysis_subgraph
from app.core.logger import setup_logger

logger = setup_logger("agent.graph")

_resume_analysis_subgraph = None
_jd_analysis_subgraph = None

def get_resume_analysis_subgraph():
    """获取已编译的简历分析子图。"""
    return _resume_analysis_subgraph


def get_jd_analysis_subgraph():
    """获取已编译的 JD 分析子图。"""
    return _jd_analysis_subgraph


def _route_decision(state: AgentState) -> str:
    """
    条件边：根据 task_type + route_type 决定走哪条路径。

    路由映射：
    - task_type == resume_analysis → "resume_analysis"（子图路径）
    - task_type == qa + route_type == retrieve → "retrieve"
    - task_type == qa + route_type == web → "web"
    - task_type == qa + route_type == direct → "direct"
    """
    task_type = state.get("task_type", "qa")
    route_type = state.get("route_type", "direct")

    # 统一转为字符串比较
    task_type_str = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    route_type_str = route_type.value if isinstance(route_type, RouteType) else str(route_type)

    # 简历分析优先
    if task_type_str == "resume_analysis":
        return "resume_analysis"

    # JD 分析
    if task_type_str == "jd_analysis":
        return "jd_analysis"

    # QA 路径：按 route_type 分发
    return route_type_str


def build_agent_graph(
    retrieval_service: Any,
    web_search_service: Any,
    settings: Any,
) -> StateGraph:
    """
    构建并编译 Agent 主图。

    图结构（阶段 4）：
                                ┌── extract_resume ── retrieve_jd ── generate_analysis ──┐
    START → router ──task──┤                                                           ├── END
                                ├── jd_analysis ── extract_jd ── analyze_jd ────────────┤
                                └── (QA: retrieve / web / direct → normalize → generate)─┘

    Args:
        retrieval_service: RetrievalService 实例
        web_search_service: WebSearchService 实例
        settings: Settings 配置实例
    Returns:
        编译后的 CompiledGraph（带 checkpointer）
    """
    # ---- 注入依赖到各节点 ----
    set_retrieval_service(retrieval_service, top_k=settings.top_k)
    set_retrieval_service_jd(retrieval_service, top_k=settings.top_k)
    set_web_search_service(web_search_service, max_results=settings.web_search_max_results)
    set_max_history(settings.agent_max_history)
    set_max_chars(settings.web_search_result_max_chars)

    global _resume_analysis_subgraph, _jd_analysis_subgraph

    web_available = web_search_service.is_available
    _resume_analysis_subgraph = build_resume_analysis_subgraph()
    _jd_analysis_subgraph = build_jd_analysis_subgraph()

    # ---- 构建图 ----
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("router", partial(route_query, web_search_available=web_available))
    builder.add_node("search_kb", search_kb)
    builder.add_node("search_web", search_web)
    builder.add_node("normalize_kb", normalize_kb)
    builder.add_node("normalize_web", normalize_web)
    builder.add_node("generate", generate_streaming_node)

    # 分析子图节点
    builder.add_node("resume_analysis", _resume_analysis_subgraph)
    builder.add_node("jd_analysis", _jd_analysis_subgraph)

    # 添加边
    builder.add_edge(START, "router")

    # 条件边：router → resume_analysis / jd_analysis / retrieve / web / direct
    builder.add_conditional_edges(
        "router",
        _route_decision,
        {
            "resume_analysis": "resume_analysis",
            "jd_analysis": "jd_analysis",
            "retrieve": "search_kb",
            "web": "search_web",
            "direct": "generate",
        },
    )

    # QA 线性路径：kb → normalize → generate
    builder.add_edge("search_kb", "normalize_kb")
    builder.add_edge("normalize_kb", "generate")

    # QA 线性路径：web → normalize → generate
    builder.add_edge("search_web", "normalize_web")
    builder.add_edge("normalize_web", "generate")

    # 三条路径都汇入 END
    builder.add_edge("generate", END)
    builder.add_edge("resume_analysis", END)
    builder.add_edge("jd_analysis", END)

    # 编译（注入 checkpointer 实现 thread 持久化）
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    logger.info(
        "Agent 主图编译完成 (web_available=%s, checkpointer=%s, 简历分析+JD分析子图已启用)",
        web_available,
        type(checkpointer).__name__,
    )
    return graph
