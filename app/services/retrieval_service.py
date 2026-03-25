"""
检索服务 - query embedding + top-k 召回
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.repositories.vector_store import FAISSVectorStore
from app.services.embedding_service import embed_single

logger = setup_logger("retrieval_service")

_settings = get_settings()


class RetrievalService:
    """知识库检索服务"""

    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        根据自然语言问题检索最相关的知识片段。

        Args:
            query: 用户问题
            top_k: 返回条数，默认从配置读取
        Returns:
            [{"content": str, "source": str, "page": int|None, "score": float}, ...]
        """
        if self.vector_store.is_empty():
            logger.warning("知识库为空，无法检索")
            return []

        top_k = top_k or _settings.top_k
        logger.info("检索: query='%s', top_k=%d", query[:50], top_k)

        # 1. query → embedding
        query_embedding = embed_single(query)
        if not query_embedding:
            logger.error("query embedding 失败，返回空结果")
            return []

        # 2. FAISS 相似度搜索
        results = self.vector_store.search(query_embedding, top_k=top_k)

        logger.info(
            "检索完成: 返回 %d 条, 最高分 %.4f",
            len(results),
            results[0]["score"] if results else 0,
        )
        return results
