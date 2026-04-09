"""
PostgreSQL metadata store for FAISS-backed chunk metadata.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from app.core.config import Settings, get_settings
from app.core.logger import setup_logger

logger = setup_logger("metadata_store")


@dataclass
class ChunkPayload:
    """Chunk payload before persistence."""

    content: str
    source: str
    chunk_index: int
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PostgresMetadataStore:
    """Metadata repository backed by PostgreSQL."""

    def __init__(self, db_url: str | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db_url = db_url or self.settings.metadata_db_url or self.settings.checkpoint_db_url

    @property
    def is_available(self) -> bool:
        return bool(self.db_url)

    def _connect(self):
        if not self.db_url:
            raise RuntimeError("metadata_db_url 未配置，无法初始化 PostgreSQL metadata store")
        return psycopg.connect(self.db_url, autocommit=False, row_factory=dict_row)

    def setup(self) -> None:
        """Create tables if needed."""
        if not self.is_available:
            logger.warning("metadata store 未配置 PostgreSQL，跳过初始化")
            return

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id UUID PRIMARY KEY,
                    source_name TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL DEFAULT 'general_kb',
                    title TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    content TEXT NOT NULL,
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_chunks_document_id ON kb_chunks(document_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_chunks_source ON kb_chunks(source)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_documents_source_type ON kb_documents(source_type)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_documents_category ON kb_documents(category)"
            )
            conn.commit()
        logger.info("metadata store 初始化完成 (PostgreSQL)")

    def infer_document_type(self, source_name: str) -> str:
        lowered = source_name.lower()
        if "resume" in lowered or "cv" in lowered or "简历" in source_name:
            return "resume"
        if "jd" in lowered or "job" in lowered or "岗位" in source_name:
            return "jd"
        if "interview" in lowered or "面试" in source_name:
            return "interview_kb"
        return "general_kb"

    def infer_category(self, source_name: str, source_type: str) -> str:
        lowered = source_name.lower()
        if source_type in {"resume", "jd"}:
            return source_type
        if "python" in lowered:
            return "python"
        if "agent" in lowered or "智能体" in source_name:
            return "agent"
        if "rag" in lowered:
            return "rag"
        if source_type == "interview_kb":
            return "interview"
        return "general"

    def upsert_document_chunks(
        self,
        *,
        source_name: str,
        chunks: list[ChunkPayload],
        source_type: str | None = None,
        title: str | None = None,
        category: str | None = None,
    ) -> list[int]:
        """Replace one logical document and insert all chunks, returning chunk ids in order."""
        if not chunks:
            return []

        source_type = source_type or self.infer_document_type(source_name)
        title = title or source_name
        category = category or self.infer_category(source_name, source_type)
        document_id = str(uuid.uuid4())

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM kb_documents WHERE source_name = %s", (source_name,))
            cur.execute(
                """
                INSERT INTO kb_documents (id, source_name, source_type, title, category)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (document_id, source_name, source_type, title, category),
            )

            chunk_ids: list[int] = []
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO kb_chunks (
                        document_id, chunk_index, source, page, content, metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        document_id,
                        chunk.chunk_index,
                        chunk.source,
                        chunk.page,
                        chunk.content,
                        Json(chunk.metadata),
                    ),
                )
                inserted = cur.fetchone()
                chunk_ids.append(int(inserted["id"]))

            conn.commit()

        logger.info(
            "metadata 已写入 PostgreSQL: source=%s, type=%s, chunks=%d",
            source_name,
            source_type,
            len(chunk_ids),
        )
        return chunk_ids

    def import_legacy_records(self, records: list[dict[str, Any]]) -> list[int]:
        """Import legacy metadata.json records while preserving original order."""
        if not records:
            return []

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record.get("source", "")].append(record)

        record_to_chunk_id: dict[tuple[str, int], int] = {}
        source_offsets: dict[str, int] = defaultdict(int)

        for source, group in grouped.items():
            chunks = [
                ChunkPayload(
                    content=item.get("content", ""),
                    source=item.get("source", source),
                    page=item.get("page"),
                    chunk_index=index,
                    metadata=item.get("metadata", {}) or {},
                )
                for index, item in enumerate(group)
            ]
            inserted_ids = self.upsert_document_chunks(
                source_name=source or f"legacy-{uuid.uuid4().hex[:8]}",
                chunks=chunks,
                source_type=self.infer_document_type(source),
            )
            for idx, chunk_id in enumerate(inserted_ids):
                record_to_chunk_id[(source, idx)] = chunk_id

        ordered_ids: list[int] = []
        for record in records:
            source = record.get("source", "")
            offset = source_offsets[source]
            ordered_ids.append(record_to_chunk_id[(source, offset)])
            source_offsets[source] += 1
        return ordered_ids

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.source,
                    c.page,
                    c.content,
                    c.metadata_json,
                    d.source_type,
                    d.category,
                    d.title
                FROM kb_chunks c
                JOIN kb_documents d ON d.id = c.document_id
                WHERE c.id = ANY(%s)
                """,
                (chunk_ids,),
            )
            rows = cur.fetchall()

        row_by_id = {
            int(row["id"]): {
                "id": str(row["id"]),
                "content": row["content"],
                "source": row["source"],
                "page": row["page"],
                "metadata": row["metadata_json"] or {},
                "source_type": row["source_type"],
                "category": row["category"],
                "title": row["title"],
            }
            for row in rows
        }
        return [row_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in row_by_id]

    def list_sources(
        self,
        *,
        source_type: str | None = None,
        category: str | None = None,
    ) -> list[str]:
        with self._connect() as conn, conn.cursor() as cur:
            sql = "SELECT source_name FROM kb_documents WHERE 1=1"
            params: list[Any] = []
            if source_type:
                sql += " AND source_type = %s"
                params.append(source_type)
            if category:
                sql += " AND category = %s"
                params.append(category)
            sql += " ORDER BY source_name ASC"
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["source_name"] for row in rows]

    def list_documents(
        self,
        *,
        source_type: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            sql = """
                SELECT
                    d.id,
                    d.source_name,
                    d.source_type,
                    d.title,
                    d.category,
                    COUNT(c.id) AS chunk_count
                FROM kb_documents d
                LEFT JOIN kb_chunks c ON c.document_id = d.id
                WHERE 1=1
            """
            params: list[Any] = []
            if source_type:
                sql += " AND d.source_type = %s"
                params.append(source_type)
            if category:
                sql += " AND d.category = %s"
                params.append(category)
            sql += """
                GROUP BY d.id, d.source_name, d.source_type, d.title, d.category
                ORDER BY d.source_name ASC
            """
            cur.execute(sql, params)
            return cur.fetchall()

    def delete_by_source(self, source_name: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM kb_documents WHERE source_name = %s",
                (source_name,),
            )
            doc = cur.fetchone()
            if not doc:
                conn.rollback()
                return 0

            cur.execute(
                "SELECT COUNT(*) AS chunk_count FROM kb_chunks WHERE document_id = %s",
                (doc["id"],),
            )
            row = cur.fetchone()
            chunk_count = int(row["chunk_count"])
            cur.execute("DELETE FROM kb_documents WHERE id = %s", (doc["id"],))
            conn.commit()
        logger.info("metadata 已删除: source=%s, chunks=%d", source_name, chunk_count)
        return chunk_count

    def count_chunks(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM kb_chunks")
            row = cur.fetchone()
        return int(row["total"])

    def get_existing_chunk_ids(self, chunk_ids: list[int]) -> set[int]:
        if not chunk_ids:
            return set()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM kb_chunks WHERE id = ANY(%s)",
                (chunk_ids,),
            )
            rows = cur.fetchall()
        return {int(row["id"]) for row in rows}


def load_legacy_metadata(meta_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return payload.get("records", [])
