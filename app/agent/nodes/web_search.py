"""
Web 搜索节点 - 调用 WebSearchService
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.core.logger import setup_logger

logger = setup_logger("agent.web_search")

# 模块级依赖注入点，由 graph.py 在构建时设置
_web_search_service = None
_max_results = 5


def set_web_search_service(service, max_results: int = 5) -> None:
    """注入 WebSearchService 实例（由 graph.py 调用）"""
    global _web_search_service, _max_results
    _web_search_service = service
    _max_results = max_results


def search_web(state: AgentState) -> dict:
    """
    Web 搜索节点：调用 Tavily API 搜索互联网。

    Returns:
        {"context_sources": [{"content", "source", "type": "web"}]}
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
        logger.warning("未找到用户消息，跳过 Web 搜索")
        return {"context_sources": []}

    if _web_search_service is None:
        logger.error("WebSearchService 未注入，无法搜索")
        return {"context_sources": []}

    results = _web_search_service.search(query, max_results=_max_results)

    context_sources = []
    for item in results:
        context_sources.append({
            "content": item.get("content", ""),
            "source": item.get("source", ""),
            "type": "web",
        })

    logger.info("Web 搜索: query='%s', 返回 %d 条", query[:50], len(context_sources))
    return {"context_sources": context_sources}
