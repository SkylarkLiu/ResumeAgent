"""
Expert wrapper nodes with lightweight session cache.
"""
from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

from app.agent.agents.cache_store import get_cache_store
from app.agent.agents.expert_cache import (
    build_jd_expert_cache_entry,
    build_jd_expert_cache_key,
    build_resume_expert_cache_entry,
    build_resume_expert_cache_key,
    build_summary_expert_cache_entry,
    build_summary_expert_cache_key,
)
from app.core.observation import log_cache_access


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


async def _run_analysis_flow(flow, state: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    async for mode, payload in flow.astream(
        state,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom":
            _emit_custom_event(payload)
        elif mode == "updates":
            for node_output in payload.values():
                if isinstance(node_output, dict):
                    merged.update(node_output)
    return merged


def build_resume_expert_node(flow):
     # 外层负责“配置”，内层负责“执行”
    async def _resume_expert_node(state: dict[str, Any]) -> dict[str, Any]:
        cache_store = get_cache_store()
        cache_key = build_resume_expert_cache_key(state)
        cached_entry, expert_cache = await cache_store.fetch_entry(
            state=state,
            expert_name="resume_expert",
            cache_key=cache_key,
        )
        if cached_entry:
            meta = dict(cached_entry.get("_meta", {}) or {})
            _emit_custom_event({"type": "status", "content": "正在复用已有简历分析结果"})
            if cached_entry.get("resume_data"):
                summary = {k: v for k, v in (cached_entry.get("resume_data") or {}).items() if k != "raw_text"}
                _emit_custom_event({"type": "extracted", "resume_data": summary})
            if cached_entry.get("context_sources"):
                _emit_custom_event({"type": "sources", "sources": cached_entry.get("context_sources", [])})
            _emit_custom_event({
                "type": "agent_cache_hit",
                "agent": "resume_expert",
                "task_type": state.get("task_type", ""),
                "question_signature": state.get("question_signature", ""),
                "response_mode": state.get("response_mode", ""),
                "backend": meta.get("backend", cache_store.backend_name),
                "hit_count": meta.get("hit_count", 0),
                "cached_at": meta.get("created_at", ""),
            })
            # 可观测性：缓存命中日志
            log_cache_access(
                expert_name="resume_expert",
                cache_key=cache_key,
                hit=True,
                backend=meta.get("backend", cache_store.backend_name),
                hit_count=meta.get("hit_count", 0),
                thread_id=str(state.get("session_id", "")),
            )
            return {
                **cached_entry,
                "expert_cache": expert_cache,
            }

        result = await _run_analysis_flow(flow, state)
        cache_entry = build_resume_expert_cache_entry(state, result, cache_key=cache_key)
        updated_cache = await cache_store.put_entry(
            state=state,
            expert_cache=state.get("expert_cache", {}) or {},
            expert_name="resume_expert",
            cache_key=cache_key,
            entry=cache_entry,
        )
        # 可观测性：缓存未命中日志
        log_cache_access(
            expert_name="resume_expert",
            cache_key=cache_key,
            hit=False,
            thread_id=str(state.get("session_id", "")),
        )
        return {
            **result,
            "expert_cache": updated_cache,
        }

    return _resume_expert_node


def build_jd_expert_node(flow):
    # 外层负责“配置”，产出可注册到builder中的函数，而当graph运行到该节点时，传入state，内层负责“执行”
    async def _jd_expert_node(state: dict[str, Any]) -> dict[str, Any]:
        cache_store = get_cache_store()
        cache_key = build_jd_expert_cache_key(state)
        cached_entry, expert_cache = await cache_store.fetch_entry(
            state=state,
            expert_name="jd_expert",
            cache_key=cache_key,
        )
        if cached_entry:
            meta = dict(cached_entry.get("_meta", {}) or {})
            _emit_custom_event({"type": "status", "content": "正在复用已有岗位分析结果"})
            if cached_entry.get("jd_data"):
                summary = {k: v for k, v in (cached_entry.get("jd_data") or {}).items() if k != "raw_text"}
                _emit_custom_event({"type": "extracted", "jd_data": summary})
            _emit_custom_event({
                "type": "agent_cache_hit",
                "agent": "jd_expert",
                "task_type": state.get("task_type", ""),
                "question_signature": state.get("question_signature", ""),
                "response_mode": state.get("response_mode", ""),
                "backend": meta.get("backend", cache_store.backend_name),
                "hit_count": meta.get("hit_count", 0),
                "cached_at": meta.get("created_at", ""),
            })
            # 可观测性：缓存命中日志
            log_cache_access(
                expert_name="jd_expert",
                cache_key=cache_key,
                hit=True,
                backend=meta.get("backend", cache_store.backend_name),
                hit_count=meta.get("hit_count", 0),
                thread_id=str(state.get("session_id", "")),
            )
            return {
                **cached_entry,
                "expert_cache": expert_cache,
            }

        result = await _run_analysis_flow(flow, state)
        cache_entry = build_jd_expert_cache_entry(state, result, cache_key=cache_key)
        updated_cache = await cache_store.put_entry(
            state=state,
            expert_cache=state.get("expert_cache", {}) or {},
            expert_name="jd_expert",
            cache_key=cache_key,
            entry=cache_entry,
        )
        # 可观测性：缓存未命中日志
        log_cache_access(
            expert_name="jd_expert",
            cache_key=cache_key,
            hit=False,
            thread_id=str(state.get("session_id", "")),
        )
        return {
            **result,
            "expert_cache": updated_cache,
        }

    return _jd_expert_node


def build_summary_expert_node(expert_node):
    async def _summary_expert_node(state: dict[str, Any]) -> dict[str, Any]:
        cache_store = get_cache_store()
        cache_key = build_summary_expert_cache_key(state)
        cached_entry, expert_cache = await cache_store.fetch_entry(
            state=state,
            expert_name="summary_expert",
            cache_key=cache_key,
        )
        if cached_entry:
            meta = dict(cached_entry.get("_meta", {}) or {})
            _emit_custom_event({"type": "status", "content": "正在复用已有综合评估结果"})
            if cached_entry.get("summary_data"):
                _emit_custom_event({"type": "summary_data", "summary_data": cached_entry.get("summary_data", {})})
            _emit_custom_event({
                "type": "agent_cache_hit",
                "agent": "summary_expert",
                "task_type": state.get("task_type", ""),
                "question_signature": state.get("question_signature", ""),
                "response_mode": state.get("response_mode", ""),
                "backend": meta.get("backend", cache_store.backend_name),
                "hit_count": meta.get("hit_count", 0),
                "cached_at": meta.get("created_at", ""),
            })
            log_cache_access(
                expert_name="summary_expert",
                cache_key=cache_key,
                hit=True,
                backend=meta.get("backend", cache_store.backend_name),
                hit_count=meta.get("hit_count", 0),
                thread_id=str(state.get("session_id", "")),
            )
            return {
                **cached_entry,
                "expert_cache": expert_cache,
            }

        result = await expert_node(state)
        cache_entry = build_summary_expert_cache_entry(state, result, cache_key=cache_key)
        updated_cache = await cache_store.put_entry(
            state=state,
            expert_cache=state.get("expert_cache", {}) or {},
            expert_name="summary_expert",
            cache_key=cache_key,
            entry=cache_entry,
        )
        log_cache_access(
            expert_name="summary_expert",
            cache_key=cache_key,
            hit=False,
            thread_id=str(state.get("session_id", "")),
        )
        return {
            **result,
            "expert_cache": updated_cache,
        }

    return _summary_expert_node
