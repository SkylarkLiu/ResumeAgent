"""
Agent 主图 - V1 多 Agent 编排。
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.agents import (
    build_jd_expert_node,
    build_qa_flow_subgraph,
    build_resume_expert_node,
    generate_final_node,
    supervisor_plan_node,
    supervisor_plan_route,
    supervisor_review_node,
    supervisor_review_route,
)
from app.agent.checkpointer import get_checkpointer
from app.agent.nodes.generate import set_max_history
from app.agent.nodes.kb_search import set_retrieval_service
from app.agent.nodes.normalize import set_max_chars
from app.agent.nodes.retrieve_jd import set_retrieval_service_jd
from app.agent.nodes.web_search import set_web_search_service
from app.agent.state import AgentState
from app.agent.subgraphs import build_jd_analysis_subgraph, build_resume_analysis_subgraph
from app.core.logger import setup_logger

logger = setup_logger("agent.graph")

_qa_flow_subgraph = None
_resume_analysis_subgraph = None
_jd_analysis_subgraph = None


def get_qa_flow_subgraph():
    """获取已编译的 QA 子图。"""
    return _qa_flow_subgraph


def get_resume_analysis_subgraph():
    """获取已编译的简历分析子图。"""
    return _resume_analysis_subgraph


def get_jd_analysis_subgraph():
    """获取已编译的 JD 分析子图。"""
    return _jd_analysis_subgraph


def build_agent_graph(
    retrieval_service: Any,
    web_search_service: Any,
    settings: Any,
) -> StateGraph:
    """
    构建并编译 Agent 主图。

    图结构（V1）：
    START -> supervisor_plan -> qa_flow / jd_expert / resume_expert / generate_final
                                  expert 执行完成后 -> supervisor_review -> continue/respond

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

    global _qa_flow_subgraph, _resume_analysis_subgraph, _jd_analysis_subgraph

    _resume_analysis_subgraph = build_resume_analysis_subgraph()
    _jd_analysis_subgraph = build_jd_analysis_subgraph()
    _qa_flow_subgraph = build_qa_flow_subgraph()

    # ---- 构建图 ----
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node(
        "supervisor_plan",
        lambda state: supervisor_plan_node(state, web_search_available=web_search_service.is_available),
    )
    builder.add_node("qa_flow", _qa_flow_subgraph)
    builder.add_node("jd_expert", build_jd_expert_node(_jd_analysis_subgraph))
    builder.add_node("resume_expert", build_resume_expert_node(_resume_analysis_subgraph))
    builder.add_node("supervisor_review", supervisor_review_node)
    builder.add_node("generate_final", generate_final_node)

    builder.add_edge(START, "supervisor_plan")

    builder.add_conditional_edges(
        "supervisor_plan",
        supervisor_plan_route,
        {
            "qa_flow": "qa_flow",
            "jd_expert": "jd_expert",
            "resume_expert": "resume_expert",
            "respond": "generate_final",
        },
    )

    builder.add_edge("qa_flow", "supervisor_review")
    builder.add_edge("jd_expert", "supervisor_review")
    builder.add_edge("resume_expert", "supervisor_review")

    builder.add_conditional_edges(
        "supervisor_review",
        supervisor_review_route,
        {
            "continue": "supervisor_plan",
            "respond": "generate_final",
        },
    )
    builder.add_edge("generate_final", END)

    # 编译（注入 checkpointer 实现 thread 持久化）
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    logger.info(
        "Agent 主图编译完成 (checkpointer=%s, V1 supervisor=%s, qa/resume/jd 子流程已启用)",
        type(checkpointer).__name__,
        True,
    )
    return graph
