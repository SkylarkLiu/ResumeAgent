"""
Expert cache helpers for multi-agent V1.
"""
from __future__ import annotations

import hashlib
import json
import re
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


def _latest_user_question(messages: list[Any]) -> str:
    for msg in reversed(messages or []):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _normalize_question_signature(task_type: str, question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "empty"

    if task_type == "jd_followup":
        if "面试" in q:
            return "jd_followup:interview"
        if any(k in q for k in ("技术栈", "技能", "能力")):
            return "jd_followup:skills"
        if any(k in q for k in ("优先", "重点", "最需要")):
            return "jd_followup:priority"
        return "jd_followup:generic"

    if task_type == "resume_followup":
        if any(k in q for k in ("优化", "润色", "改写", "重写")):
            return "resume_followup:optimize"
        if any(k in q for k in ("亮点", "优势", "突出")):
            return "resume_followup:strengths"
        if any(k in q for k in ("怎么写", "如何写")):
            return "resume_followup:writing"
        return "resume_followup:generic"

    if task_type == "match_followup":
        if any(k in q for k in ("缺", "差距", "缺口", "不足")):
            return "match_followup:gap"
        if any(k in q for k in ("优先", "最需要", "先补", "先学")):
            return "match_followup:priority"
        if any(k in q for k in ("匹配", "匹配度")):
            return "match_followup:match"
        return "match_followup:generic"

    compact = re.sub(r"\s+", " ", q)
    return compact[:200]


def build_resume_expert_cache_key(state: dict[str, Any]) -> str:
    task_type = str(state.get("task_type", ""))
    question = _latest_user_question(state.get("messages", []))
    payload = {
        "task_type": task_type,
        "question_signature": _normalize_question_signature(task_type, question),
        "resume_data": _trim_resume_payload(state.get("resume_data")),
        "jd_data": _trim_jd_payload(state.get("jd_data")),
    }
    return _stable_hash(payload)


def build_jd_expert_cache_key(state: dict[str, Any]) -> str:
    task_type = str(state.get("task_type", ""))
    question = _latest_user_question(state.get("messages", []))
    payload = {
        "task_type": task_type,
        "question_signature": _normalize_question_signature(task_type, question),
        "jd_data": _trim_jd_payload(state.get("jd_data")),
        "resume_data": _trim_resume_payload(state.get("resume_data")),
    }
    return _stable_hash(payload)


def build_resume_expert_cache_entry(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_answer": result.get("final_answer", ""),
        "messages": result.get("messages", []),
        "resume_data": result.get("resume_data"),
        "jd_data": state.get("jd_data"),
        "context_sources": result.get("context_sources", state.get("context_sources", [])),
        "working_context": result.get("working_context", state.get("working_context", "")),
    }


def build_jd_expert_cache_entry(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_answer": result.get("final_answer", ""),
        "messages": result.get("messages", []),
        "jd_data": result.get("jd_data"),
        "resume_data": state.get("resume_data"),
    }


def upsert_cache_bucket(expert_cache: dict[str, dict] | None, expert_name: str, cache_key: str, entry: dict[str, Any], max_entries: int = 5) -> dict[str, dict]:
    cache = dict(expert_cache or {})
    bucket = dict(cache.get(expert_name, {}) or {})
    bucket[cache_key] = entry
    while len(bucket) > max_entries:
        oldest_key = next(iter(bucket))
        bucket.pop(oldest_key, None)
    cache[expert_name] = bucket
    return cache
