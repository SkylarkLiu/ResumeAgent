"""
JD 检索节点 - 从知识库中检索匹配的岗位要求标准

输入：state["resume_data"]（结构化简历信息）
输出：state["context_sources"]（JD 相关来源）+ state["working_context"]
"""
from __future__ import annotations

from app.agent.state import AgentState
from app.core.logger import setup_logger

logger = setup_logger("agent.retrieve_jd")

# 模块级依赖注入点，由 graph.py 在构建时设置
_retrieval_service = None
_top_k = 5


def set_retrieval_service_jd(service, top_k: int = 5) -> None:
    """注入 RetrievalService 实例（由 graph.py 调用，与 kb_search 共享）"""
    global _retrieval_service, _top_k
    _retrieval_service = service
    _top_k = top_k


def retrieve_jd(state: AgentState) -> dict:
    """
    JD 检索节点：优先使用 session 中已有的 jd_data（真实 JD），
    如果没有则从知识库中检索匹配的岗位要求标准。

    检索策略：
    1. 检查 state["jd_data"] 是否已有结构化数据
    2. 有 → 直接序列化为 working_context，跳过知识库检索
    3. 无 → 走原有逻辑（skills + target_position 检索知识库）

    Returns:
        {"context_sources": list[dict], "working_context": str}
    """
    # ---- 优先：使用 session 中的真实 JD ----
    jd_data = state.get("jd_data")
    if jd_data and isinstance(jd_data, dict) and not jd_data.get("extract_error"):
        # 序列化 jd_data 为文本上下文
        import json

        jd_for_context = {k: v for k, v in jd_data.items() if k not in ("raw_text", "extract_error")}
        jd_json = json.dumps(jd_for_context, ensure_ascii=False, indent=2)

        position = jd_data.get("position", "未知岗位")
        company = jd_data.get("company", "")
        label = f"【用户上传的 JD - {position}】"
        if company:
            label += f" {company}"

        working_context = f"{label}\n{jd_json}"
        context_sources = [{"content": working_context, "source": f"用户上传的 JD ({position})", "score": 1.0, "type": "jd_upload"}]

        logger.info(
            "使用 session 中的真实 JD: position=%s, company=%s",
            position,
            company,
        )

        return {
            "context_sources": context_sources,
            "working_context": working_context,
        }

    # ---- 降级：知识库检索 ----
    resume_data = state.get("resume_data") or {}

    if _retrieval_service is None:
        logger.warning("RetrievalService 未注入，跳过 JD 检索")
        return {"context_sources": [], "working_context": ""}

    # 提取检索关键词
    skills = resume_data.get("skills", [])
    target_position = resume_data.get("target_position", "")
    summary = resume_data.get("summary", "")

    # 构造多组查询以扩大召回
    queries = []

    # 查询 1：技能 + 目标岗位
    if skills and target_position:
        queries.append(f"{target_position} 岗位要求 {' '.join(skills[:10])}")
    elif skills:
        queries.append(" ".join(skills[:10]))

    # 查询 2：技能关键词子集（扩大召回面）
    if len(skills) > 5:
        queries.append(" ".join(skills[:5]))

    # 查询 3：简历摘要
    if summary:
        queries.append(f"后端开发岗位要求 {summary}")

    if not queries:
        logger.info("无有效检索关键词，跳过 JD 检索")
        return {"context_sources": [], "working_context": ""}

    # 执行多组查询并去重（按 content 前 50 字符去重）
    seen_contents = set()
    all_sources = []

    for query in queries:
        try:
            results = _retrieval_service.retrieve(query, top_k=_top_k)
            for item in results:
                content = item.get("content", "")
                # 简单去重
                dedup_key = content[:80].strip()
                if dedup_key in seen_contents:
                    continue
                seen_contents.add(dedup_key)

                all_sources.append({
                    "content": content,
                    "source": item.get("source", ""),
                    "score": item.get("score", 0.0),
                    "page": item.get("page"),
                    "type": "kb",
                    "query_used": query,
                })
        except Exception as e:
            logger.warning("JD 检索异常 (query='%s'): %s", query[:50], e)

    # 按 score 降序排列，取 top
    all_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_sources = all_sources[:_top_k * 2]  # 多取一些给分析节点用

    # 拼装 working_context
    if top_sources:
        parts = []
        for i, src in enumerate(top_sources):
            label = f"【参考资料{i + 1} - 知识库】"
            if src.get("source"):
                label += f" {src['source']}"
            parts.append(f"{label}\n{src['content']}")

        working_context = "\n\n".join(parts)
    else:
        working_context = ""

    logger.info(
        "JD 检索: queries=%d组, 去重后 %d 条来源, 上下文=%d字符",
        len(queries),
        len(top_sources),
        len(working_context),
    )

    return {
        "context_sources": top_sources,
        "working_context": working_context,
    }
