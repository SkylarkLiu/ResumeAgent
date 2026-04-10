"""
ReAct fallback 工具定义与最小缓存。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agent.prompts import MATCH_FOLLOWUP_DIRECT_PROMPT
from app.services.llm_service import chat_completion


def build_react_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_sources",
                "description": "列出知识库中的来源文件名，可按 source_type 或 category 过滤。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_type": {"type": "string", "description": "文档类型，如 general_kb、interview_kb、jd、resume"},
                        "category": {"type": "string", "description": "文档分类，如 rag、agent、python、interview"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "列出知识库中的逻辑文档，可按 source_type 或 category 过滤。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_type": {"type": "string", "description": "文档类型，如 general_kb、interview_kb、jd、resume"},
                        "category": {"type": "string", "description": "文档分类，如 rag、agent、python、interview"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compact_faiss",
                "description": "压缩 FAISS 索引，移除 metadata 中已删除文档对应的失效向量行。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": "检索本地知识库，适用于项目文档、面试知识库等内部内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题"},
                        "top_k": {"type": "integer", "description": "返回数量", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filter_kb_by_type",
                "description": "按 source_type/category 过滤知识库检索结果，适合限定在面试知识库、JD、简历或某一类主题内。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题"},
                        "source_type": {"type": "string", "description": "文档类型过滤"},
                        "category": {"type": "string", "description": "分类过滤"},
                        "top_k": {"type": "integer", "description": "返回数量", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "搜索互联网公开信息，适合实时或知识库中不存在的问题。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索问题"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_jd",
                "description": "从岗位描述原文中抽取结构化 JD 信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raw_text": {"type": "string", "description": "岗位描述原文"},
                    },
                    "required": ["raw_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_resume",
                "description": "从简历原文中抽取结构化简历信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raw_text": {"type": "string", "description": "简历原文"},
                    },
                    "required": ["raw_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "match_resume_jd",
                "description": "基于已有简历和 JD 结构化信息，直接产出匹配差距、优先补强项和简短建议。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "用户关于匹配/差距/补强的问题"},
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": "在已经收集到足够信息后，生成最终简洁回答。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "用户问题"},
                        "working_context": {"type": "string", "description": "可选的补充上下文"},
                    },
                    "required": ["question"],
                },
            },
        },
    ]


def build_tool_cache_key(tool_name: str, arguments: dict[str, Any], *, session_id: str = "", question_signature: str = "") -> str:
    raw = json.dumps(
        {
            "tool": tool_name,
            "arguments": arguments,
            "session_id": session_id,
            "question_signature": question_signature,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_cacheable_tool(tool_name: str) -> bool:
    return tool_name in {
        "list_sources",
        "list_documents",
        "search_kb",
        "filter_kb_by_type",
        "search_web",
        "extract_jd",
        "extract_resume",
    }


def build_report_messages(question: str, *, working_context: str = "", jd_data: dict | None = None, resume_data: dict | None = None) -> list[dict[str, str]]:
    payload = [f"用户问题：\n{question}"]
    if jd_data:
        payload.append(f"JD 结构化信息：\n{json.dumps(jd_data, ensure_ascii=False)}")
    if resume_data:
        payload.append(f"简历结构化信息：\n{json.dumps(resume_data, ensure_ascii=False)}")
    if working_context:
        payload.append(f"补充上下文：\n{working_context}")
    return [
        {
            "role": "system",
            "content": (
                "你是 ResumeAgent 的补充分析助手。请基于已收集到的工具结果，"
                "直接回答用户问题。优先给出结论，保持简洁，不要暴露工具调用过程。"
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(payload),
        },
    ]


def generate_report_answer(question: str, *, working_context: str = "", jd_data: dict | None = None, resume_data: dict | None = None) -> str:
    return chat_completion(
        build_report_messages(
            question,
            working_context=working_context,
            jd_data=jd_data,
            resume_data=resume_data,
        ),
        temperature=0.3,
        max_tokens=900,
        thinking={"type": "disabled"},
    )


def _build_resume_compact_summary(resume_data: dict[str, Any]) -> dict[str, Any]:
    projects = resume_data.get("projects") or []
    experience = resume_data.get("experience") or []
    return {
        "summary": resume_data.get("summary", ""),
        "target_position": resume_data.get("target_position", ""),
        "skills_top": list((resume_data.get("skills") or [])[:10]),
        "project_names": [item.get("name", "") for item in projects[:3] if isinstance(item, dict) and item.get("name")],
        "experience_positions": [item.get("position", "") for item in experience[:3] if isinstance(item, dict) and item.get("position")],
    }


def _build_jd_compact_summary(jd_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": jd_data.get("position", ""),
        "summary": jd_data.get("summary", ""),
        "skills_must_top": list((jd_data.get("skills_must") or [])[:10]),
        "keywords_top": list((jd_data.get("keywords") or [])[:10]),
    }


def generate_match_answer(question: str, *, jd_data: dict[str, Any], resume_data: dict[str, Any]) -> str:
    prompt = (
        MATCH_FOLLOWUP_DIRECT_PROMPT
        + "\n\n### 简历摘要\n{resume_data}\n\n### JD 摘要\n{jd_context}\n\n### 用户问题\n{user_question}"
    ).format(
        resume_data=json.dumps(_build_resume_compact_summary(resume_data), ensure_ascii=False, indent=2),
        jd_context=json.dumps(_build_jd_compact_summary(jd_data), ensure_ascii=False, indent=2),
        user_question=question,
    )
    return chat_completion(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=900,
        thinking={"type": "disabled"},
    )
