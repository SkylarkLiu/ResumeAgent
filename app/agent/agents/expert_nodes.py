"""
Expert wrapper nodes with lightweight session cache.
"""
from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

from app.agent.agents.expert_cache import (
    build_jd_expert_cache_entry,
    build_jd_expert_cache_key,
    build_resume_expert_cache_entry,
    build_resume_expert_cache_key,
    upsert_cache_bucket,
)


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


async def _run_subgraph(subgraph, state: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    async for mode, payload in subgraph.astream(
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


def build_resume_expert_node(subgraph):
    async def _resume_expert_node(state: dict[str, Any]) -> dict[str, Any]:
        cache_key = build_resume_expert_cache_key(state)
        expert_cache = state.get("expert_cache", {}) or {}
        cached_entry = (expert_cache.get("resume_expert") or {}).get(cache_key)
        if cached_entry:
            _emit_custom_event({"type": "status", "content": "正在复用已有简历分析结果"})
            if cached_entry.get("resume_data"):
                summary = {k: v for k, v in (cached_entry.get("resume_data") or {}).items() if k != "raw_text"}
                _emit_custom_event({"type": "extracted", "resume_data": summary})
            if cached_entry.get("context_sources"):
                _emit_custom_event({"type": "sources", "sources": cached_entry.get("context_sources", [])})
            _emit_custom_event({"type": "agent_cache_hit", "agent": "resume_expert"})
            return {
                **cached_entry,
                "expert_cache": expert_cache,
            }

        result = await _run_subgraph(subgraph, state)
        cache_entry = build_resume_expert_cache_entry(state, result)
        updated_cache = upsert_cache_bucket(expert_cache, "resume_expert", cache_key, cache_entry)
        return {
            **result,
            "expert_cache": updated_cache,
        }

    return _resume_expert_node


def build_jd_expert_node(subgraph):
    async def _jd_expert_node(state: dict[str, Any]) -> dict[str, Any]:
        cache_key = build_jd_expert_cache_key(state)
        expert_cache = state.get("expert_cache", {}) or {}
        cached_entry = (expert_cache.get("jd_expert") or {}).get(cache_key)
        if cached_entry:
            _emit_custom_event({"type": "status", "content": "正在复用已有岗位分析结果"})
            if cached_entry.get("jd_data"):
                summary = {k: v for k, v in (cached_entry.get("jd_data") or {}).items() if k != "raw_text"}
                _emit_custom_event({"type": "extracted", "jd_data": summary})
            _emit_custom_event({"type": "agent_cache_hit", "agent": "jd_expert"})
            return {
                **cached_entry,
                "expert_cache": expert_cache,
            }

        result = await _run_subgraph(subgraph, state)
        cache_entry = build_jd_expert_cache_entry(state, result)
        updated_cache = upsert_cache_bucket(expert_cache, "jd_expert", cache_key, cache_entry)
        return {
            **result,
            "expert_cache": updated_cache,
        }

    return _jd_expert_node
