"""
可观测性日志工具 — 统一打印每次请求的关键决策字段。

用法:
    from app.core.observation import log_request_decision

    log_request_decision(
        state,
        extra={"cache_hit": True, "cache_backend": "state_checkpointer"},
    )

输出的 logger 名称为 "obs.request"，可在日志系统中独立过滤。
"""
from __future__ import annotations

from typing import Any

from app.core.logger import setup_logger
from app.agent.agents.cache_store import get_cache_store_backend

_obs_logger = setup_logger("obs.request")


def log_request_decision(
    state: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    打印一次请求路由决策的结构化日志。

    固定字段:
      - task_type
      - question_signature
      - response_mode
      - cache_backend
      - cache_hit  (需要调用方在 extra 中传入)
      - thread_id  (= session_id)
    """
    thread_id = str(state.get("session_id") or state.get("thread_id") or "")
    task_type = str(state.get("task_type") or "")
    question_signature = str(state.get("question_signature") or "")
    response_mode = str(state.get("response_mode") or "")
    cache_backend = get_cache_store_backend()

    payload = {
        "thread_id": thread_id,
        "task_type": task_type,
        "question_signature": question_signature,
        "response_mode": response_mode,
        "cache_backend": cache_backend,
    }

    if extra:
        payload.update(extra)

    # 用 | 分隔，方便 grep/awk
    parts = [f"{k}={v}" for k, v in payload.items()]
    _obs_logger.info("REQ_DECISION | %s", " | ".join(parts))


def log_cache_access(
    *,
    expert_name: str,
    cache_key: str,
    hit: bool,
    backend: str = "",
    hit_count: int = 0,
    thread_id: str = "",
) -> None:
    """打印缓存访问结果。"""
    _obs_logger.info(
        "CACHE_ACCESS | expert=%s | key=%s..%s | hit=%s | backend=%s | hits=%d | thread=%s",
        expert_name,
        cache_key[:8],
        cache_key[-4:],
        hit,
        backend or get_cache_store_backend(),
        hit_count,
        thread_id,
    )


def log_subgraph_skip(
    *,
    node_name: str,
    reason: str,
    thread_id: str = "",
) -> None:
    """打印子图节点跳过（去重执行）日志。"""
    _obs_logger.info(
        "SUBGRAPH_SKIP | node=%s | reason=%s | thread=%s",
        node_name,
        reason,
        thread_id,
    )
