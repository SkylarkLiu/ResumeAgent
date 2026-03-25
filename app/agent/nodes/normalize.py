"""
结果标准化节点 - KB/Web 结果统一拼装为 working_context
"""
from __future__ import annotations

from app.agent.state import AgentState
from app.core.logger import setup_logger

logger = setup_logger("agent.normalize")

# 由 graph.py 在构建时设置
_web_search_result_max_chars: int = 500


def set_max_chars(max_chars: int) -> None:
    """注入 Web 结果最大字符数（由 graph.py 调用）"""
    global _web_search_result_max_chars
    _web_search_result_max_chars = max_chars


def normalize_kb(state: AgentState) -> dict:
    """
    KB 检索结果标准化：从 context_sources 拼装 working_context 文本。

    格式：【来源{i+1} - 知识库】{source}
          {content}
    """
    context_sources = state.get("context_sources", [])
    kb_sources = [s for s in context_sources if s.get("type") == "kb"]

    if not kb_sources:
        logger.info("无 KB 检索结果，working_context 为空")
        return {"working_context": ""}

    parts = []
    for i, src in enumerate(kb_sources):
        label = f"【来源{i+1} - 知识库】"
        if src.get("source"):
            label += f" {src['source']}"
        if src.get("page"):
            label += f" 第{src['page']}页"
        parts.append(f"{label}\n{src['content']}")

    working_context = "\n\n".join(parts)
    logger.info("KB 标准化: %d 条来源, 共 %d 字符", len(kb_sources), len(working_context))
    return {"working_context": working_context}


def normalize_web(state: AgentState) -> dict:
    """
    Web 搜索结果标准化：截断 + 拼装 working_context 文本。

    格式：【来源{i+1} - 网络搜索】{source}
          {content（截断到 max_chars）}
    """
    context_sources = state.get("context_sources", [])
    web_sources = [s for s in context_sources if s.get("type") == "web"]

    if not web_sources:
        logger.info("无 Web 搜索结果，working_context 为空")
        return {"working_context": ""}

    parts = []
    for i, src in enumerate(web_sources):
        label = f"【来源{i+1} - 网络搜索】"
        if src.get("source"):
            label += f" {src['source']}"

        content = src.get("content", "")
        # 截断过长内容
        if len(content) > _web_search_result_max_chars:
            content = content[:_web_search_result_max_chars] + "..."

        parts.append(f"{label}\n{content}")

    working_context = "\n\n".join(parts)
    logger.info("Web 标准化: %d 条来源, 共 %d 字符", len(web_sources), len(working_context))
    return {"working_context": working_context}
