"""
Debug / Runtime API — 仅在 debug_mode=True 时可用。

提供：
- /debug/runtime — 运行时状态快照（checkpointer 后端、缓存后端、会话列表等）
- /debug/session/{session_id} — 单个会话的完整 state dump
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.agent import _agent_graph, _checkpointer
from app.core.config import get_settings
from app.core.logger import setup_logger
from app.agent.agents.cache_store import get_cache_store, get_cache_store_backend

logger = setup_logger("api.debug")
settings = get_settings()

router = APIRouter(prefix="/debug", tags=["Debug"])


def _check_debug_allowed() -> None:
    """检查是否允许访问 debug 接口，不允许则 404。"""
    if not settings.debug_mode:
        raise HTTPException(status_code=404, detail="Debug endpoints are disabled")


@router.get("/runtime")
async def debug_runtime():
    """
    运行时状态快照。

    返回：
    - checkpointer 后端和类型
    - expert cache 后端
    - 当前活跃 thread 列表（仅 MemorySaver 支持）
    - 配置摘要
    """
    _check_debug_allowed()

    checkpointer_backend = "unknown"
    thread_list: list[dict[str, Any]] = []

    if _checkpointer is not None:
        checkpointer_type = type(_checkpointer).__name__
        checkpointer_backend = checkpointer_type

        # MemorySaver 支持列举 thread
        if hasattr(_checkpointer, "storage") and isinstance(_checkpointer.storage, dict):
            for thread_id, state_data in _checkpointer.storage.items():
                values = state_data.get("values", {}) if isinstance(state_data, dict) else {}
                messages = values.get("messages", []) if isinstance(values, dict) else []
                thread_list.append({
                    "thread_id": thread_id,
                    "message_count": len(messages),
                    "has_jd_data": bool(values.get("jd_data")),
                    "has_resume_data": bool(values.get("resume_data")),
                    "expert_cache_entries": _count_cache_entries(values.get("expert_cache", {})),
                })

        # AsyncPostgresSaver: 直接查 PG 获取 thread 列表
        # 注意：AsyncPostgresSaver 没有 alist_threads 方法
        elif hasattr(_checkpointer, "alist"):
            try:
                from app.core.config import get_settings as _get_settings
                _settings = _get_settings()
                db_url = (getattr(_settings, "checkpoint_db_url", "") or "").strip()
                if db_url:
                    from psycopg import AsyncConnection
                    from psycopg.rows import dict_row
                    async with await AsyncConnection.connect(
                        db_url, autocommit=True, row_factory=dict_row
                    ) as conn:
                        cur = await conn.execute("""
                            SELECT thread_id, COUNT(*) as cnt
                            FROM checkpoints
                            GROUP BY thread_id
                        """)
                        for row in await cur.fetchall():
                            thread_list.append({
                                "thread_id": row["thread_id"],
                                "message_count": 0,
                                "checkpoint_count": row["cnt"],
                            })
            except Exception as e:
                logger.debug("[debug] PG 线程列表查询失败: %s", e)

    cache_store = get_cache_store()
    cache_backend = get_cache_store_backend()

    return {
        "checkpointer": {
            "backend": checkpointer_backend,
            "thread_count": len(thread_list),
        },
        "expert_cache": {
            "backend": cache_backend,
        },
        "threads": thread_list,
        "config": {
            "app_env": settings.app_env,
            "log_level": settings.log_level,
            "llm_model": settings.llm_model_name,
            "debug_mode": settings.debug_mode,
            "checkpoint_db_url_set": bool(settings.checkpoint_db_url),
            "expert_cache_db_url_set": bool(settings.expert_cache_db_url),
        },
    }


@router.get("/session/{session_id}")
async def debug_session(session_id: str):
    """
    单个会话的完整 state dump。

    注意：大 state 序列化可能很慢，仅用于排障。
    """
    _check_debug_allowed()

    if _agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not initialized")

    try:
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = await _agent_graph.aget_state(config)
        if not state_snapshot:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        values = getattr(state_snapshot, "values", {}) or {}

        # 截断大字段，避免返回超大 JSON
        safe_values: dict[str, Any] = {}
        for key, val in values.items():
            if key == "messages":
                safe_values["message_count"] = len(val) if isinstance(val, list) else 0
                # 保留最后 3 条消息的摘要
                safe_values["recent_messages"] = _summarize_messages(val[-3:] if isinstance(val, list) else [])
            elif key == "expert_cache":
                safe_values["expert_cache"] = _summarize_cache(val)
            elif key in ("resume_data", "jd_data"):
                safe_values[key] = _summarize_structured_data(val)
            else:
                safe_values[key] = val

        # 添加 checkpoint 元信息
        safe_values["_checkpoint"] = {
            "next": getattr(state_snapshot, "next", []),
            "created_at": str(getattr(state_snapshot, "created_at", "")),
            "parent_config": str(getattr(state_snapshot, "parent_config", None)),
        }

        return {"session_id": session_id, "state": safe_values}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Debug session dump failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _count_cache_entries(expert_cache: Any) -> int:
    """统计 expert_cache 中总条目数。"""
    if not isinstance(expert_cache, dict):
        return 0
    return sum(len(bucket) for bucket in expert_cache.values() if isinstance(bucket, dict))


def _summarize_messages(messages: list) -> list[dict]:
    """只保留消息的 type 和前 100 字符内容。"""
    result = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            content = content[:100]
        elif isinstance(content, list):
            content = str(content)[:100]
        result.append({
            "type": getattr(msg, "type", type(msg).__name__),
            "content_preview": content,
        })
    return result


def _summarize_cache(expert_cache: Any) -> dict:
    """只保留 cache 的键和 meta，不保留完整 entry。"""
    if not isinstance(expert_cache, dict):
        return {}
    summary = {}
    for expert_name, bucket in expert_cache.items():
        if not isinstance(bucket, dict):
            continue
        entries = []
        for cache_key, entry in bucket.items():
            meta = entry.get("_meta", {}) if isinstance(entry, dict) else {}
            entries.append({
                "cache_key": f"{cache_key[:8]}..{cache_key[-4:]}",
                "hit_count": meta.get("hit_count", 0),
                "created_at": meta.get("created_at", ""),
                "last_hit_at": meta.get("last_hit_at", ""),
                "backend": meta.get("backend", ""),
            })
        summary[expert_name] = entries
    return summary


def _summarize_structured_data(data: Any) -> dict:
    """结构化数据只保留 key 名和类型，不保留完整内容。"""
    if not isinstance(data, dict):
        return {"type": type(data).__name__}
    return {k: type(v).__name__ for k, v in data.items()}
