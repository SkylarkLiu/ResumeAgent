"""
Agent 模块 - LangGraph 轻量 Agent 编排（阶段 2: checkpointer 持久化）
"""
from app.agent.graph import (
    build_agent_graph,
    get_jd_analysis_flow,
    get_jd_analysis_subgraph,
    get_resume_analysis_flow,
    get_resume_analysis_subgraph,
)
from app.agent.agents.cache_store import (
    get_cache_store_backend,
    init_cache_store,
    shutdown_cache_store,
)
from app.agent.checkpointer import (
    get_checkpointer,
    get_checkpointer_backend,
    init_checkpointer,
    shutdown_checkpointer,
)

__all__ = [
    "build_agent_graph",
    "init_cache_store",
    "shutdown_cache_store",
    "get_cache_store_backend",
    "get_checkpointer",
    "get_checkpointer_backend",
    "init_checkpointer",
    "shutdown_checkpointer",
    # 新命名（推荐）
    "get_resume_analysis_flow",
    "get_jd_analysis_flow",
    # 兼容旧调用方（后续可删除）
    "get_resume_analysis_subgraph",
    "get_jd_analysis_subgraph",
]
