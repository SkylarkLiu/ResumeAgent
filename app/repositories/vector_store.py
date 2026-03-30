"""
FAISS 向量存储 - 知识库持久化 + 相似度搜索
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("vector_store")

_settings = get_settings()

# FAISS 索引文件名约定
_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


@dataclass
class VectorRecord:
    """单条向量记录"""
    id: str
    content: str
    embedding: list[float]
    source: str = ""
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class FAISSVectorStore:
    """
    FAISS 向量存储引擎。
    
    - 使用 IndexFlatIP（内积）+ L2 归一化 = 余弦相似度
    - 元数据（content/source/page）存为 JSON 侧文件
    - 支持 save/load 持久化
    """

    def __init__(self, index_dir: str | Path | None = None):
        self.index_dir = Path(index_dir or _settings.faiss_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int = 0
        self.records: list[VectorRecord] = []
        self._loaded = False

    # ---- 构建索引 ----

    def build(self, records: list[VectorRecord]) -> None:
        """
        从 VectorRecord 列表构建 FAISS 索引。
        
        Args:
            records: 包含 content + embedding 的记录列表
        """
        if not records:
            logger.warning("传入空记录列表，跳过构建")
            return

        self.records = records
        self.dimension = len(records[0].embedding)

        # 构造归一化向量矩阵
        embeddings = np.array([r.embedding for r in records], dtype=np.float32)
        # L2 归一化 → 内积等价于余弦相似度
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self._loaded = True
        logger.info(
            "FAISS 索引构建完成: %d 条记录, 维度 %d",
            self.index.ntotal,
            self.dimension,
        )

    def add_records(self, new_records: list[VectorRecord]) -> None:
        """
        追加记录到已有索引。如果索引不存在则等同 build。
        """
        if not new_records:
            return

        if self.index is None:
            self.build(new_records)
            return

        # 校验维度一致
        if len(new_records[0].embedding) != self.dimension:
            raise ValueError(
                f"维度不匹配: 现有 {self.dimension}, 新增 {len(new_records[0].embedding)}"
            )

        self.records.extend(new_records)
        embeddings = np.array([r.embedding for r in new_records], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        logger.info("追加 %d 条记录, 总计 %d", len(new_records), self.index.ntotal)

    # ---- 相似度搜索 ----

    def search(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[dict]:
        """
        相似度搜索。

        Args:
            query_embedding: 查询向量
            top_k: 返回 top 几条，默认从配置读取
        Returns:
            [{"content": str, "source": str, "page": int|None, "score": float, "id": str}, ...]
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("索引为空，返回空结果")
            return []

        top_k = top_k or _settings.top_k
        top_k = min(top_k, self.index.ntotal)

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS 对不足 top_k 的填充
            record = self.records[idx]
            results.append({
                "id": record.id,
                "content": record.content,
                "source": record.source,
                "page": record.page,
                "score": float(score),
                "metadata": record.metadata,
            })

        logger.debug("搜索完成: 返回 %d 条, 最高分 %.4f", len(results), results[0]["score"] if results else 0)
        return results

    # ---- 持久化 ----

    def save(self) -> None:
        """保存索引和元数据到磁盘"""
        if self.index is None:
            logger.warning("索引为空，跳过保存")
            return

        index_path = self.index_dir / _INDEX_FILE
        meta_path = self.index_dir / _META_FILE

        faiss.write_index(self.index, str(index_path))

        # 元数据序列化（embedding 不存储，太大）
        meta = {
            "dimension": self.dimension,
            "total": len(self.records),
            "records": [
                {
                    "id": r.id,
                    "content": r.content,
                    "source": r.source,
                    "page": r.page,
                    "metadata": r.metadata,
                }
                for r in self.records
            ],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("索引已保存: %d 条 -> %s", len(self.records), self.index_dir)

    def load(self) -> bool:
        """
        从磁盘加载索引和元数据。
        Returns:
            是否加载成功
        """
        index_path = self.index_dir / _INDEX_FILE
        meta_path = self.index_dir / _META_FILE

        if not index_path.exists() or not meta_path.exists():
            logger.info("未找到已有索引，将从头构建")
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            self.dimension = meta["dimension"]
            self.records = [
                VectorRecord(
                    id=r["id"],
                    content=r["content"],
                    embedding=[],  # 加载时不恢复 embedding，节省内存
                    source=r.get("source", ""),
                    page=r.get("page"),
                    metadata=r.get("metadata", {}),
                )
                for r in meta["records"]
            ]

            self._loaded = True
            logger.info("索引已加载: %d 条, 维度 %d", self.index.ntotal, self.dimension)
            return True

        except Exception as e:
            logger.error("索引加载失败: %s", e)
            return False

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0

    def delete_by_source(self, source: str) -> int:
        """
        删除指定来源的所有向量记录，重建 FAISS 索引。

        Args:
            source: 要删除的来源文件名（与 VectorRecord.source 精确匹配）
        Returns:
            删除的记录数量
        """
        if not self.records:
            logger.info("索引为空，无需删除")
            return 0

        original_count = len(self.records)
        self.records = [r for r in self.records if r.source != source]
        deleted_count = original_count - len(self.records)

        if deleted_count == 0:
            logger.info("未找到来源为 '%s' 的记录", source)
            return 0

        # 重建索引
        if self.records:
            self.build(self.records)
        else:
            # 全部删除后清空索引
            self.index = None
            self.dimension = 0
            self._loaded = False

        self.save()
        logger.info(
            "已删除来源 '%s' 的 %d 条记录, 剩余 %d 条",
            source, deleted_count, len(self.records),
        )
        return deleted_count

    def get_sources(self) -> list[str]:
        """返回知识库中所有不重复的来源文件名"""
        return sorted(set(r.source for r in self.records))
