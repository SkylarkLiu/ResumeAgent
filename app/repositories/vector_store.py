"""
FAISS vector store that keeps only index + row map.
Chunk metadata/content lives in PostgreSQL.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.repositories.metadata_store import PostgresMetadataStore, load_legacy_metadata

logger = setup_logger("vector_store")

_settings = get_settings()

_INDEX_FILE = "index.faiss"
_ROW_MAP_FILE = "row_map.json"
_STORE_META_FILE = "meta.json"
_LEGACY_META_FILE = "metadata.json"


@dataclass
class VectorRecord:
    """Vector record for FAISS add/build operations."""

    id: str
    embedding: list[float]
    content: str = ""
    source: str = ""
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FAISSVectorStore:
    """
    FAISS storage with PostgreSQL metadata backend.

    - FAISS only stores vectors
    - row_map.json stores FAISS row -> PostgreSQL chunk id
    - PostgreSQL stores content/source/page/metadata
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        metadata_store: PostgresMetadataStore | None = None,
    ):
        self.index_dir = Path(index_dir or _settings.faiss_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int = 0
        self.row_map: list[int] = []
        self.metadata_store = metadata_store
        self._loaded = False

    def set_metadata_store(self, metadata_store: PostgresMetadataStore) -> None:
        self.metadata_store = metadata_store

    @property
    def _index_path(self) -> Path:
        return self.index_dir / _INDEX_FILE

    @property
    def _row_map_path(self) -> Path:
        return self.index_dir / _ROW_MAP_FILE

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / _STORE_META_FILE

    @property
    def _legacy_meta_path(self) -> Path:
        return self.index_dir / _LEGACY_META_FILE

    def build(self, records: list[VectorRecord]) -> None:
        if not records:
            logger.warning("传入空记录列表，跳过构建")
            return

        self.dimension = len(records[0].embedding)
        self.row_map = [int(record.id) for record in records]

        embeddings = np.array([r.embedding for r in records], dtype=np.float32)
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
        if not new_records:
            return

        if self.index is None:
            self.build(new_records)
            return

        if len(new_records[0].embedding) != self.dimension:
            raise ValueError(
                f"维度不匹配: 现有 {self.dimension}, 新增 {len(new_records[0].embedding)}"
            )

        embeddings = np.array([r.embedding for r in new_records], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.row_map.extend(int(record.id) for record in new_records)
        logger.info("追加 %d 条记录, 总计 %d", len(new_records), self.index.ntotal)

    def search(self, query_embedding: list[float], top_k: int | None = None) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("索引为空，返回空结果")
            return []
        if self.metadata_store is None:
            logger.warning("metadata store 未初始化，返回空结果")
            return []

        top_k = top_k or _settings.top_k
        candidate_k = min(self.index.ntotal, max(top_k * 5, top_k + 20))

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, candidate_k)

        ordered_chunk_ids: list[int] = []
        score_map: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.row_map):
                continue
            chunk_id = self.row_map[idx]
            ordered_chunk_ids.append(chunk_id)
            score_map[chunk_id] = float(score)

        chunks = self.metadata_store.get_chunks_by_ids(ordered_chunk_ids)
        results: list[dict] = []
        for chunk in chunks:
            chunk_id = int(chunk["id"])
            results.append(
                {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": score_map.get(chunk_id, 0.0),
                    "metadata": chunk.get("metadata", {}),
                    "source_type": chunk.get("source_type", ""),
                    "category": chunk.get("category", ""),
                    "title": chunk.get("title", ""),
                }
            )
            if len(results) >= top_k:
                break

        logger.debug(
            "搜索完成: 返回 %d 条, 最高分 %.4f",
            len(results),
            results[0]["score"] if results else 0,
        )
        return results

    def save(self) -> None:
        if self.index is None:
            logger.warning("索引为空，跳过保存")
            return

        faiss.write_index(self.index, str(self._index_path))
        self._row_map_path.write_text(
            json.dumps(self.row_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._meta_path.write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "total": len(self.row_map),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("索引已保存: %d 条 -> %s", len(self.row_map), self.index_dir)

    def load(self) -> bool:
        if not self._index_path.exists():
            logger.info("未找到已有索引，将从头构建")
            return False

        if self._row_map_path.exists():
            try:
                self.index = faiss.read_index(str(self._index_path))
                self.row_map = json.loads(self._row_map_path.read_text(encoding="utf-8"))
                if self._meta_path.exists():
                    meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                    self.dimension = int(meta.get("dimension", 0))
                elif self.index is not None:
                    self.dimension = self.index.d
                self._loaded = True
                logger.info("索引已加载: %d 条, 维度 %d", self.index.ntotal, self.dimension)
                return True
            except Exception as exc:
                logger.error("索引加载失败: %s", exc)
                return False

        if self._legacy_meta_path.exists():
            logger.info("检测到旧 metadata.json，等待迁移")
            return False

        logger.info("未找到 row_map，将从头构建")
        return False

    def has_legacy_metadata(self) -> bool:
        return self._index_path.exists() and self._legacy_meta_path.exists() and not self._row_map_path.exists()

    def migrate_legacy_metadata(self) -> bool:
        if not self.has_legacy_metadata():
            return False
        if self.metadata_store is None:
            raise RuntimeError("metadata store 未初始化，无法迁移旧 metadata.json")

        logger.info("开始迁移旧 metadata.json -> PostgreSQL")
        records = load_legacy_metadata(self._legacy_meta_path)
        row_map = self.metadata_store.import_legacy_records(records)

        self.index = faiss.read_index(str(self._index_path))
        self.dimension = self.index.d
        self.row_map = row_map
        self._loaded = True
        self.save()

        backup_path = self.index_dir / "metadata.legacy.json.bak"
        shutil.move(str(self._legacy_meta_path), str(backup_path))
        logger.info("旧 metadata.json 已迁移并备份到 %s", backup_path.name)
        return True

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0

    def delete_by_source(self, source: str) -> int:
        if self.metadata_store is None:
            logger.warning("metadata store 未初始化，无法删除 source=%s", source)
            return 0
        deleted_count = self.metadata_store.delete_by_source(source)
        if deleted_count:
            logger.info(
                "已删除 metadata source=%s, chunks=%d。FAISS 索引暂未压缩，检索时会自动跳过缺失 chunk。",
                source,
                deleted_count,
            )
        return deleted_count

    def get_sources(self) -> list[str]:
        if self.metadata_store is None:
            return []
        return self.metadata_store.list_sources()

    def compact(self) -> dict[str, int]:
        """
        Rebuild FAISS index by removing rows whose metadata no longer exists.
        Uses FAISS reconstruct to avoid re-embedding.
        """
        if self.index is None or not self.row_map:
            return {"before": 0, "after": 0, "removed": 0}
        if self.metadata_store is None:
            raise RuntimeError("metadata store 未初始化，无法 compact")

        existing_ids = self.metadata_store.get_existing_chunk_ids(self.row_map)
        keep_rows = [
            (row_idx, chunk_id)
            for row_idx, chunk_id in enumerate(self.row_map)
            if chunk_id in existing_ids
        ]

        before = len(self.row_map)
        if len(keep_rows) == before:
            logger.info("FAISS compact 跳过: 无失效 row")
            return {"before": before, "after": before, "removed": 0}

        if not keep_rows:
            self.index = None
            self.dimension = 0
            self.row_map = []
            self._loaded = False
            if self._index_path.exists():
                self._index_path.unlink()
            if self._row_map_path.exists():
                self._row_map_path.unlink()
            if self._meta_path.exists():
                self._meta_path.unlink()
            logger.info("FAISS compact 完成: 全部 row 已清空")
            return {"before": before, "after": 0, "removed": before}

        vectors = np.array(
            [self.index.reconstruct(row_idx) for row_idx, _chunk_id in keep_rows],
            dtype=np.float32,
        )
        new_index = faiss.IndexFlatIP(self.dimension)
        new_index.add(vectors)

        self.index = new_index
        self.row_map = [chunk_id for _row_idx, chunk_id in keep_rows]
        self._loaded = True
        self.save()

        after = len(self.row_map)
        removed = before - after
        logger.info("FAISS compact 完成: before=%d, after=%d, removed=%d", before, after, removed)
        return {"before": before, "after": after, "removed": removed}
