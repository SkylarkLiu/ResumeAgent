"""
7D-4 回归测试：PostgreSQL checkpointer 持久化 + expert cache 持久化。

验证：
1. MemorySaver checkpointer 正常工作（默认后端）
2. StateBackedExpertCacheStore 正常工作（默认后端）
3. PostgreSQL 后端降级行为（未配置 DB URL 时回退到内存）
4. /health 端点返回正确的后端信息
5. /debug/runtime 端点在 debug_mode=False 时返回 404
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_returns_backend_info(client):
    """/health 端点返回 checkpointer 和 cache 后端信息。"""
    resp = await client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok", f"Expected status ok, got {data}"
    assert "checkpointer_backend" in data, "/health should include checkpointer_backend"
    assert "expert_cache_backend" in data, "/health should include expert_cache_backend"
    assert "index_records" in data, "/health should include index_records"


@pytest.mark.asyncio
async def test_health_memory_checkpointer(client):
    """测试环境下 checkpointer 应为内存模式。"""
    resp = await client.get("/health")
    data = resp.json()
    # 测试环境未配置 CHECKPOINT_DB_URL，应为 memory
    assert data["checkpointer_backend"] == "memory", \
        f"Expected memory checkpointer, got {data['checkpointer_backend']}"


@pytest.mark.asyncio
async def test_health_state_checkpointer_cache(client):
    """测试环境下 expert cache 后端名称不为 unknown。"""
    resp = await client.get("/health")
    data = resp.json()
    # 注意：lifespan 中的 init_cache_store 可能受真实配置影响
    # 只要 backend 不是 unknown 就说明初始化成功
    assert data["expert_cache_backend"] in ("state_checkpointer", "postgres"), \
        f"Expected valid cache backend, got {data['expert_cache_backend']}"


@pytest.mark.asyncio
async def test_debug_runtime_disabled_by_default(client):
    """/debug/runtime 在 debug_mode=False 时应返回 404。"""
    resp = await client.get("/debug/runtime")
    # debug_mode=False 时路由不注册，应 404
    assert resp.status_code == 404, f"Expected 404 when debug disabled, got {resp.status_code}"


# ---- 后端单元测试 ----

class TestStateBackedExpertCacheStore:
    """StateBackedExpertCacheStore 单元测试。"""

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_empty(self):
        """空缓存应返回 None。"""
        from app.agent.agents.cache_store import StateBackedExpertCacheStore

        store = StateBackedExpertCacheStore()
        entry, cache = await store.fetch_entry(
            state={"session_id": "test-thread"},
            expert_name="resume_expert",
            cache_key="abc123",
        )
        assert entry is None
        assert isinstance(cache, dict)

    @pytest.mark.asyncio
    async def test_put_then_fetch(self):
        """写入后读取应命中。"""
        from app.agent.agents.cache_store import StateBackedExpertCacheStore

        store = StateBackedExpertCacheStore()
        entry = {
            "final_answer": "测试回答",
            "_meta": {"expert": "resume_expert"},
        }

        updated_cache = await store.put_entry(
            state={"session_id": "test-thread", "expert_cache": {}},
            expert_cache={},
            expert_name="resume_expert",
            cache_key="key1",
            entry=entry,
        )

        # fetch 应命中
        fetched, cache = await store.fetch_entry(
            state={"session_id": "test-thread", "expert_cache": updated_cache},
            expert_name="resume_expert",
            cache_key="key1",
        )
        assert fetched is not None
        assert fetched["final_answer"] == "测试回答"
        # hit_count 应为 1
        assert fetched["_meta"]["hit_count"] == 1

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self):
        """超过 max_entries 应淘汰最旧条目。"""
        from app.agent.agents.cache_store import StateBackedExpertCacheStore

        store = StateBackedExpertCacheStore()
        cache = {}

        # 写入 6 条（max_entries=5）
        for i in range(6):
            updated = await store.put_entry(
                state={"session_id": "test-thread", "expert_cache": cache},
                expert_cache=cache,
                expert_name="resume_expert",
                cache_key=f"key_{i}",
                entry={"final_answer": f"answer_{i}", "_meta": {}},
                max_entries=5,
            )
            cache = updated

        # 第一条应被淘汰
        fetched, _ = await store.fetch_entry(
            state={"session_id": "test-thread", "expert_cache": cache},
            expert_name="resume_expert",
            cache_key="key_0",
        )
        assert fetched is None, "Oldest entry should be evicted"

        # 最后一条应存在
        fetched, _ = await store.fetch_entry(
            state={"session_id": "test-thread", "expert_cache": cache},
            expert_name="resume_expert",
            cache_key="key_5",
        )
        assert fetched is not None, "Latest entry should exist"


class TestPostgresExpertCacheStoreFallback:
    """PostgresExpertCacheStore 降级行为测试。"""

    @pytest.mark.asyncio
    async def test_fallback_when_psycopg_missing(self):
        """未安装 psycopg 时应降级到 StateBackedExpertCacheStore。"""
        from app.agent.agents.cache_store import PostgresExpertCacheStore

        store = PostgresExpertCacheStore(db_url="postgresql://fake:5432/test")
        await store.setup()
        # psycopg 未安装 → _ready=False
        assert not store._ready

        # 写入（通过 delegate）
        updated = await store.put_entry(
            state={"session_id": "test", "expert_cache": {}},
            expert_cache={},
            expert_name="jd_expert",
            cache_key="key1",
            entry={"final_answer": "test", "_meta": {}},
        )

        # 读取应命中
        fetched, _ = await store.fetch_entry(
            state={"session_id": "test", "expert_cache": updated},
            expert_name="jd_expert",
            cache_key="key1",
        )
        assert fetched is not None
        # 降级时 backend 应为 state_checkpointer_fallback
        assert fetched["_meta"]["backend"] == "state_checkpointer_fallback"


# ---- Checkpointer 初始化测试 ----

class TestCheckpointerInit:
    """Checkpointer 初始化测试。"""

    @pytest.mark.asyncio
    async def test_memory_saver_when_no_db_url(self):
        """未配置 DB URL 时应使用 MemorySaver。"""
        from app.agent.checkpointer import init_checkpointer, get_checkpointer_backend
        from langgraph.checkpoint.memory import MemorySaver

        settings = type("Settings", (), {"checkpoint_db_url": ""})()
        cp = await init_checkpointer(settings)
        assert isinstance(cp, MemorySaver)
        assert get_checkpointer_backend() == "memory"
