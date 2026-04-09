"""
Persistent-capable expert cache store abstraction.

Phase 6 first cut:
- Keep the current cache persisted inside AgentState so it remains compatible with
  LangGraph checkpointers (Memory/Postgres).
- Move cache read/write/touch logic behind a backend interface so we can later
  swap this to a dedicated Postgres table or external KV store without changing
  expert nodes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Protocol

from app.core.logger import setup_logger

logger = setup_logger("agent.cache_store")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_thread_id(state: dict[str, Any]) -> str:
    return str(state.get("session_id") or state.get("thread_id") or "")


class ExpertCacheStore(Protocol):
    backend_name: str

    async def fetch_entry(
        self,
        *,
        state: dict[str, Any],
        expert_name: str,
        cache_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, dict]]:
        ...

    async def put_entry(
        self,
        *,
        state: dict[str, Any],
        expert_cache: dict[str, dict] | None,
        expert_name: str,
        cache_key: str,
        entry: dict[str, Any],
        max_entries: int = 5,
    ) -> dict[str, dict]:
        ...

    async def setup(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...


class StateBackedExpertCacheStore:
    """
    Cache backend persisted inside AgentState.

    Because AgentState is checkpointed by LangGraph, this backend is already
    durable when PostgreSQL checkpointer is enabled. This gives us a persistent
    backend interface today without introducing a second storage system.
    """

    backend_name = "state_checkpointer"

    async def setup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def fetch_entry(
        self,
        *,
        state: dict[str, Any],
        expert_name: str,
        cache_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, dict]]:
        expert_cache = dict(state.get("expert_cache", {}) or {})
        bucket = dict(expert_cache.get(expert_name, {}) or {})
        cached_entry = bucket.get(cache_key)
        if not cached_entry:
            return None, expert_cache

        entry = deepcopy(cached_entry)
        meta = dict(entry.get("_meta", {}) or {})
        meta["backend"] = self.backend_name
        meta["hit_count"] = int(meta.get("hit_count", 0) or 0) + 1
        meta["last_hit_at"] = _utc_now_iso()
        entry["_meta"] = meta

        bucket[cache_key] = entry
        expert_cache[expert_name] = bucket
        return entry, expert_cache

    async def put_entry(
        self,
        *,
        state: dict[str, Any],
        expert_cache: dict[str, dict] | None,
        expert_name: str,
        cache_key: str,
        entry: dict[str, Any],
        max_entries: int = 5,
    ) -> dict[str, dict]:
        cache = dict(expert_cache or {})
        bucket = dict(cache.get(expert_name, {}) or {})

        payload = deepcopy(entry)
        meta = dict(payload.get("_meta", {}) or {})
        meta["backend"] = self.backend_name
        meta.setdefault("created_at", _utc_now_iso())
        meta.setdefault("last_hit_at", "")
        meta.setdefault("hit_count", 0)
        payload["_meta"] = meta

        bucket[cache_key] = payload
        while len(bucket) > max_entries:
            oldest_key = next(iter(bucket))
            bucket.pop(oldest_key, None)
        cache[expert_name] = bucket
        return cache


class PostgresExpertCacheStore:
    """
    Minimal PostgreSQL cache backend skeleton.

    Current Phase 6 scope:
    - expose a configurable backend boundary
    - keep behavior safe by delegating to the state-backed implementation
    - reserve a dedicated class for later table-backed implementation
    """

    backend_name = "postgres"

    def __init__(self, db_url: str, delegate: ExpertCacheStore | None = None) -> None:
        self.db_url = db_url
        self._delegate = delegate or StateBackedExpertCacheStore()
        self._ready = False
        self._psycopg_available = False

    async def setup(self) -> None:
        try:
            from psycopg import AsyncConnection
        except ImportError as exc:
            logger.warning("未安装 psycopg，Expert cache 回退到 state_checkpointer。错误: %s", exc)
            self._ready = False
            self._psycopg_available = False
            return

        async with await AsyncConnection.connect(self.db_url, autocommit=True) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_expert_cache (
                    thread_id TEXT NOT NULL,
                    expert_name TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    entry JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    hit_count BIGINT NOT NULL DEFAULT 0,
                    PRIMARY KEY (thread_id, expert_name, cache_key)
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_expert_cache_thread_expert
                ON agent_expert_cache (thread_id, expert_name, updated_at DESC)
                """
            )

        self._ready = True
        self._psycopg_available = True
        logger.info("Expert cache 使用 PostgreSQL 独立缓存表")

    async def shutdown(self) -> None:
        self._ready = False
        self._psycopg_available = False

    async def fetch_entry(
        self,
        *,
        state: dict[str, Any],
        expert_name: str,
        cache_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, dict]]:
        if not self._ready or not self._psycopg_available:
            entry, expert_cache = await self._delegate.fetch_entry(
                state=state,
                expert_name=expert_name,
                cache_key=cache_key,
            )
            if entry is not None:
                meta = dict(entry.get("_meta", {}) or {})
                meta["backend"] = "state_checkpointer_fallback"
                entry["_meta"] = meta
            return entry, expert_cache

        thread_id = _resolve_thread_id(state)
        if not thread_id:
            entry, expert_cache = await self._delegate.fetch_entry(
                state=state,
                expert_name=expert_name,
                cache_key=cache_key,
            )
            if entry is not None:
                meta = dict(entry.get("_meta", {}) or {})
                meta["backend"] = "state_missing_thread_fallback"
                entry["_meta"] = meta
            return entry, expert_cache

        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        async with await AsyncConnection.connect(
            self.db_url,
            autocommit=True,
            row_factory=dict_row,
        ) as conn:
            cursor = await conn.execute(
                """
                SELECT entry, created_at, updated_at, hit_count
                FROM agent_expert_cache
                WHERE thread_id = %s AND expert_name = %s AND cache_key = %s
                """,
                (thread_id, expert_name, cache_key),
            )
            row = await cursor.fetchone()
            if not row:
                return None, dict(state.get("expert_cache", {}) or {})

            await conn.execute(
                """
                UPDATE agent_expert_cache
                SET hit_count = hit_count + 1,
                    updated_at = NOW()
                WHERE thread_id = %s AND expert_name = %s AND cache_key = %s
                """,
                (thread_id, expert_name, cache_key),
            )

        entry = deepcopy(dict(row.get("entry") or {}))
        meta = dict(entry.get("_meta", {}) or {})
        meta["backend"] = self.backend_name
        meta["created_at"] = meta.get("created_at") or (row.get("created_at").isoformat() if row.get("created_at") else "")
        meta["last_hit_at"] = row.get("updated_at").isoformat() if row.get("updated_at") else _utc_now_iso()
        meta["hit_count"] = int(row.get("hit_count", 0) or 0) + 1
        entry["_meta"] = meta
        return entry, dict(state.get("expert_cache", {}) or {})

    async def put_entry(
        self,
        *,
        state: dict[str, Any],
        expert_cache: dict[str, dict] | None,
        expert_name: str,
        cache_key: str,
        entry: dict[str, Any],
        max_entries: int = 5,
    ) -> dict[str, dict]:
        if not self._ready or not self._psycopg_available:
            updated = await self._delegate.put_entry(
                state=state,
                expert_cache=expert_cache,
                expert_name=expert_name,
                cache_key=cache_key,
                entry=entry,
                max_entries=max_entries,
            )
            bucket = dict(updated.get(expert_name, {}) or {})
            saved_entry = dict(bucket.get(cache_key, {}) or {})
            if saved_entry:
                meta = dict(saved_entry.get("_meta", {}) or {})
                meta["backend"] = "state_checkpointer_fallback"
                saved_entry["_meta"] = meta
                bucket[cache_key] = saved_entry
                updated[expert_name] = bucket
            return updated

        thread_id = _resolve_thread_id(state)
        if not thread_id:
            updated = await self._delegate.put_entry(
                state=state,
                expert_cache=expert_cache,
                expert_name=expert_name,
                cache_key=cache_key,
                entry=entry,
                max_entries=max_entries,
            )
            bucket = dict(updated.get(expert_name, {}) or {})
            saved_entry = dict(bucket.get(cache_key, {}) or {})
            if saved_entry:
                meta = dict(saved_entry.get("_meta", {}) or {})
                meta["backend"] = "state_missing_thread_fallback"
                saved_entry["_meta"] = meta
                bucket[cache_key] = saved_entry
                updated[expert_name] = bucket
            return updated

        from psycopg import AsyncConnection
        from psycopg.types.json import Jsonb

        payload = deepcopy(entry)
        meta = dict(payload.get("_meta", {}) or {})
        meta["backend"] = self.backend_name
        meta.setdefault("created_at", _utc_now_iso())
        meta.setdefault("last_hit_at", "")
        meta.setdefault("hit_count", 0)
        payload["_meta"] = meta

        async with await AsyncConnection.connect(self.db_url, autocommit=True) as conn:
            await conn.execute(
                """
                INSERT INTO agent_expert_cache (thread_id, expert_name, cache_key, entry)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (thread_id, expert_name, cache_key)
                DO UPDATE SET
                    entry = EXCLUDED.entry,
                    updated_at = NOW()
                """,
                (thread_id, expert_name, cache_key, Jsonb(payload)),
            )
            await conn.execute(
                """
                DELETE FROM agent_expert_cache
                WHERE thread_id = %s
                  AND expert_name = %s
                  AND cache_key NOT IN (
                      SELECT cache_key
                      FROM agent_expert_cache
                      WHERE thread_id = %s AND expert_name = %s
                      ORDER BY updated_at DESC
                      LIMIT %s
                  )
                """,
                (thread_id, expert_name, thread_id, expert_name, max_entries),
            )

        return dict(expert_cache or {})


_DEFAULT_CACHE_STORE = StateBackedExpertCacheStore()
_CURRENT_CACHE_STORE: ExpertCacheStore = _DEFAULT_CACHE_STORE


async def init_cache_store(settings: Any) -> ExpertCacheStore:
    global _CURRENT_CACHE_STORE

    backend = str(getattr(settings, "expert_cache_backend", "state_checkpointer") or "state_checkpointer").strip().lower()
    db_url = (getattr(settings, "expert_cache_db_url", "") or getattr(settings, "checkpoint_db_url", "") or "").strip()

    if backend == "postgres":
        if not db_url:
            logger.warning("EXPERT_CACHE_BACKEND=postgres 但未配置数据库地址，回退到 state_checkpointer")
            _CURRENT_CACHE_STORE = StateBackedExpertCacheStore()
        else:
            store = PostgresExpertCacheStore(db_url=db_url)
            await store.setup()
            _CURRENT_CACHE_STORE = store
            return _CURRENT_CACHE_STORE
    else:
        _CURRENT_CACHE_STORE = StateBackedExpertCacheStore()

    await _CURRENT_CACHE_STORE.setup()
    logger.info("Expert cache backend=%s", _CURRENT_CACHE_STORE.backend_name)
    return _CURRENT_CACHE_STORE


async def shutdown_cache_store() -> None:
    await _CURRENT_CACHE_STORE.shutdown()


def get_cache_store() -> ExpertCacheStore:
    return _CURRENT_CACHE_STORE


def get_cache_store_backend() -> str:
    return _CURRENT_CACHE_STORE.backend_name


def get_default_cache_store() -> ExpertCacheStore:
    return _DEFAULT_CACHE_STORE
