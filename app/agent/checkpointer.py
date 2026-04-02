"""
Checkpointer 初始化与持久化后端选择。
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.core.logger import setup_logger

logger = setup_logger("agent.checkpointer")

_checkpointer: Any = MemorySaver()
_checkpointer_context: Any = None
_backend_name = "memory"


async def init_checkpointer(settings: Any) -> Any:
    """
    初始化全局 checkpointer。

    优先使用 PostgreSQL 持久化；未配置时回退到 MemorySaver。
    """
    global _checkpointer, _checkpointer_context, _backend_name

    db_url = (getattr(settings, "checkpoint_db_url", "") or "").strip()
    if not db_url:
        _checkpointer = MemorySaver()
        _checkpointer_context = None
        _backend_name = "memory"
        logger.info("Checkpointer 使用内存模式（未配置 CHECKPOINT_DB_URL）")
        return _checkpointer

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        logger.warning(
            "未安装 PostgreSQL checkpointer 依赖，回退到内存模式。请安装 langgraph-checkpoint-postgres。错误: %s",
            exc,
        )
        _checkpointer = MemorySaver()
        _checkpointer_context = None
        _backend_name = "memory"
        return _checkpointer

    context = AsyncPostgresSaver.from_conn_string(db_url)
    checkpointer = await context.__aenter__()
    await checkpointer.setup()

    _checkpointer = checkpointer
    _checkpointer_context = context
    _backend_name = "postgres"
    logger.info("Checkpointer 使用 PostgreSQL 持久化")
    return _checkpointer


async def shutdown_checkpointer() -> None:
    """关闭全局 checkpointer 相关资源。"""
    global _checkpointer_context
    if _checkpointer_context is not None:
        await _checkpointer_context.__aexit__(None, None, None)
        _checkpointer_context = None


def get_checkpointer() -> Any:
    """获取当前全局 checkpointer。"""
    return _checkpointer


def get_checkpointer_backend() -> str:
    """返回当前 checkpointer 后端名称。"""
    return _backend_name
