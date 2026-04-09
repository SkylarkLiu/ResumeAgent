"""
KB 检索节点 - 复用现有 RetrievalService + 检索质量评估
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.core.logger import setup_logger

logger = setup_logger("agent.kb_search")

# 模块级依赖注入点，由 graph.py 在构建时设置
_retrieval_service = None
_top_k = 5
_kb_relevance_threshold = 0.35


def set_retrieval_service(service, top_k: int = 5, kb_relevance_threshold: float = 0.35) -> None:
    """注入 RetrievalService 实例（由 graph.py 调用）"""
    global _retrieval_service, _top_k, _kb_relevance_threshold
    _retrieval_service = service
    _top_k = top_k
    _kb_relevance_threshold = kb_relevance_threshold


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


def evaluate_kb_relevance(state: AgentState) -> dict:
    """
    检索质量评估节点：检查 KB 检索结果的相关性分数。

    判断条件（满足任一则标记降级）：
    1. context_sources 为空（知识库无结果）
    2. 最高相关性分数 < kb_relevance_threshold（检索质量不足）

    Returns:
        {"retrieval_fallback": bool, "route_type": str}
        降级时同时更新 route_type 为 "web"，使后续节点使用正确的 prompt。
    """
    context_sources = state.get("context_sources", [])

    # 只筛选 kb 类型的检索结果进行评估（web 来源的高分不影响 KB 质量判断）
    kb_sources = [src for src in context_sources if src.get("type") == "kb"]

    # 无 KB 结果 → 直接降级
    if not kb_sources:
        logger.info("KB 检索结果为空，标记降级到 web search")
        return {"retrieval_fallback": True, "route_type": "web"}

    # 计算 KB 最高分
    max_score = max(src.get("score", 0.0) for src in kb_sources)

    if max_score < _kb_relevance_threshold:
        logger.info(
            "KB 检索最高分 %.4f < 阈值 %.4f，标记降级到 web search",
            max_score, _kb_relevance_threshold,
        )
        return {"retrieval_fallback": True, "route_type": "web"}

    # 质量合格，正常继续
    logger.info("KB 检索质量合格，最高分 %.4f >= 阈值 %.4f", max_score, _kb_relevance_threshold)
    return {"retrieval_fallback": False}
