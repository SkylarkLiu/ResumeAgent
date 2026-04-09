"""
Expert cache helpers for multi-agent V1.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _trim_resume_payload(resume_data: dict[str, Any] | None) -> dict[str, Any]:
    resume_data = dict(resume_data or {})
    return {
        "name": resume_data.get("name", ""),
        "summary": resume_data.get("summary", ""),
        "target_position": resume_data.get("target_position", ""),
        "skills": resume_data.get("skills", []),
        "projects": resume_data.get("projects", []),
        "experience": resume_data.get("experience", []),
        "education": resume_data.get("education", []),
        "raw_text": resume_data.get("raw_text", ""),
    }


def _trim_jd_payload(jd_data: dict[str, Any] | None) -> dict[str, Any]:
    jd_data = dict(jd_data or {})
    return {
        "position": jd_data.get("position", ""),
        "summary": jd_data.get("summary", ""),
        "skills_must": jd_data.get("skills_must", []),
        "skills_preferred": jd_data.get("skills_preferred", []),
        "tech_stack": jd_data.get("tech_stack", {}),
        "keywords": jd_data.get("keywords", []),
        "raw_text": jd_data.get("raw_text", ""),
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_cache_meta(state: dict[str, Any], expert_name: str, cache_key: str) -> dict[str, Any]:
    return {
        "expert": expert_name,
        "task_type": str(state.get("task_type", "")),
        "question_signature": str(state.get("question_signature", "")),
        "response_mode": str(state.get("response_mode", "")),
        "cache_key": cache_key,
        "created_at": _utc_now_iso(),
        "last_hit_at": "",
        "hit_count": 0,
    }


def build_resume_expert_cache_key(state: dict[str, Any]) -> str:
    task_type = str(state.get("task_type", ""))
    payload = {
        "task_type": task_type,
        "question_signature": str(state.get("question_signature", "")),
        "response_mode": str(state.get("response_mode", "")),
        "resume_data": _trim_resume_payload(state.get("resume_data")),
        "jd_data": _trim_jd_payload(state.get("jd_data")),
    }
    return _stable_hash(payload)


def build_jd_expert_cache_key(state: dict[str, Any]) -> str:
    task_type = str(state.get("task_type", ""))
    payload = {
        "task_type": task_type,
        "question_signature": str(state.get("question_signature", "")),
        "response_mode": str(state.get("response_mode", "")),
        "jd_data": _trim_jd_payload(state.get("jd_data")),
        "resume_data": _trim_resume_payload(state.get("resume_data")),
    }
    return _stable_hash(payload)


def build_resume_expert_cache_entry(state: dict[str, Any], result: dict[str, Any], *, cache_key: str) -> dict[str, Any]:
    return {
        "final_answer": result.get("final_answer", ""),
        "messages": result.get("messages", []),
        "resume_data": result.get("resume_data"),
        "jd_data": state.get("jd_data"),
        "context_sources": result.get("context_sources", state.get("context_sources", [])),
        "working_context": result.get("working_context", state.get("working_context", "")),
        "_meta": _build_cache_meta(state, "resume_expert", cache_key),
    }


def build_jd_expert_cache_entry(state: dict[str, Any], result: dict[str, Any], *, cache_key: str) -> dict[str, Any]:
    return {
        "final_answer": result.get("final_answer", ""),
        "messages": result.get("messages", []),
        "jd_data": result.get("jd_data"),
        "resume_data": state.get("resume_data"),
        "_meta": _build_cache_meta(state, "jd_expert", cache_key),
    }
