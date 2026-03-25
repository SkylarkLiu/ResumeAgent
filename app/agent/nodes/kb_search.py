"""
KB 检索节点 - 复用现有 RetrievalService
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.core.logger import setup_logger

logger = setup_logger("agent.kb_search")

# 模块级依赖注入点，由 graph.py 在构建时设置
_retrieval_service = None
_top_k = 5


def set_retrieval_service(service, top_k: int = 5) -> None:
    """注入 RetrievalService 实例（由 graph.py 调用）"""
    global _retrieval_service, _top_k
    _retrieval_service = service
    _top_k = top_k


def search_kb(state: AgentState) -> dict:
    """
    知识库检索节点：从 FAISS 向量库检索相关文档片段。

    Returns:
        {"context_sources": [{"content", "source", "score", "type": "kb"}]}
    """
    messages = state.get("messages", [])
    if not messages:
        return {"context_sources": []}

    # 取最后一条用户消息作为 query
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        logger.warning("未找到用户消息，跳过 KB 检索")
        return {"context_sources": []}

    if _retrieval_service is None:
        logger.error("RetrievalService 未注入，无法检索")
        return {"context_sources": []}

    results = _retrieval_service.retrieve(query, top_k=_top_k)

    context_sources = []
    for item in results:
        context_sources.append({
            "content": item.get("content", ""),
            "source": item.get("source", ""),
            "score": item.get("score", 0.0),
            "page": item.get("page"),
            "type": "kb",
        })

    logger.info("KB 检索: query='%s', 返回 %d 条", query[:50], len(context_sources))
    return {"context_sources": context_sources}
