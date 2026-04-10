"""
受控 ReAct fallback 节点。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from app.agent.prompts import REACT_SYSTEM_PROMPT
from app.agent.agents.react_tools import (
    build_react_tools_schema,
    build_report_messages,
    build_tool_cache_key,
    generate_match_answer,
    generate_report_answer,
    is_cacheable_tool,
)
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion, chat_completion_stream_async, chat_completion_with_tools

logger = setup_logger("agent.react_fallback")


def _emit_custom_event(payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def _latest_user_question(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content.strip() if isinstance(msg.content, str) else ""
            if content:
                return content
    return ""


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _stringify_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)


def _tool_result_summary(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name in {"search_kb", "filter_kb_by_type", "search_web"}:
        return f"sources={len(result.get('sources', []))}, context_chars={len(str(result.get('working_context', '') or ''))}"
    if tool_name == "list_sources":
        return f"sources_list={len(result.get('sources_list', []))}"
    if tool_name == "list_documents":
        return f"documents={len(result.get('documents', []))}"
    if tool_name == "extract_jd":
        jd_data = result.get("jd_data") or {}
        return f"jd_position={jd_data.get('position', '')}, skills_must={len(jd_data.get('skills_must', []) or [])}"
    if tool_name == "extract_resume":
        resume_data = result.get("resume_data") or {}
        return f"resume_name={resume_data.get('name', '')}, skills={len(resume_data.get('skills', []) or [])}"
    if tool_name == "compact_faiss":
        compact_result = result.get("compact_result") or {}
        return (
            f"before={compact_result.get('before', 0)}, "
            f"after={compact_result.get('after', 0)}, "
            f"removed={compact_result.get('removed', 0)}"
        )
    if tool_name in {"match_resume_jd", "generate_report"}:
        return f"answer_chars={len(str(result.get('answer', '') or ''))}"
    if result.get("error"):
        return f"error={result.get('error')}"
    return "ok"


def _dedupe_sources(*groups: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []
    for group in groups:
        for item in group or []:
            key = (
                str(item.get("source", "")),
                str(item.get("content", ""))[:200],
                str(item.get("type", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _summarize_context_sources(sources: list[dict], max_items: int = 5) -> str:
    blocks: list[str] = []
    for item in (sources or [])[:max_items]:
        source = item.get("source", "未知来源")
        content = str(item.get("content", "")).strip()
        item_type = item.get("type", "")
        if not content:
            continue
        prefix = f"[{item_type}] " if item_type else ""
        blocks.append(f"{prefix}{source}\n{content[:500]}")
    return "\n\n".join(blocks)


def _summarize_documents(documents: list[dict], max_items: int = 8) -> str:
    lines: list[str] = []
    for item in (documents or [])[:max_items]:
        lines.append(
            f"{item.get('source_name', '')} | type={item.get('source_type', '')} | "
            f"category={item.get('category', '')} | chunks={item.get('chunk_count', 0)}"
        )
    return "\n".join(lines)


def _question_needs_structured_extraction(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ("jd", "岗位", "简历", "resume", "匹配", "match"))


def _infer_handoff_agent(question: str, *, jd_data: dict | None, resume_data: dict | None) -> str | None:
    q = question.lower()
    if jd_data and resume_data:
        return "resume_expert"
    if jd_data and any(k in q for k in ("jd", "岗位", "职位", "面试", "要求", "技术栈")):
        return "jd_expert"
    if resume_data and any(k in q for k in ("简历", "优化", "润色", "匹配", "改进", "补强", "resume")):
        return "resume_expert"
    return None


def _has_enough_context_to_finalize(question: str, *, working_context: str, jd_data: dict | None, resume_data: dict | None, last_tool_name: str, tool_trace: list[dict]) -> bool:
    if last_tool_name == "generate_report":
        return True
    if jd_data and resume_data:
        return True
    if last_tool_name in {"extract_jd", "extract_resume"} and not _question_needs_structured_extraction(question):
        return True
    if last_tool_name in {"search_kb", "filter_kb_by_type", "search_web", "list_documents", "list_sources"} and len(working_context) >= 160:
        return True
    if len(tool_trace) >= 2:
        last_two = [item.get("tool") for item in tool_trace[-2:]]
        if len(last_two) == 2 and last_two[0] == last_two[1]:
            return True
    return False


def _is_allowed_tool_transition(tool_trace: list[dict], next_tool: str) -> bool:
    if not tool_trace:
        return True
    previous_tool = str(tool_trace[-1].get("tool") or "")
    allowed_next: dict[str, set[str]] = {
        "list_sources": {"list_documents", "search_kb", "filter_kb_by_type", "generate_report"},
        "list_documents": {"search_kb", "filter_kb_by_type", "generate_report"},
        "search_kb": {"generate_report", "filter_kb_by_type", "search_web"},
        "filter_kb_by_type": {"generate_report", "search_web"},
        "search_web": {"generate_report"},
        "extract_jd": {"extract_resume", "match_resume_jd", "generate_report"},
        "extract_resume": {"extract_jd", "match_resume_jd", "generate_report"},
        "match_resume_jd": {"generate_report"},
        "compact_faiss": {"list_sources", "list_documents", "search_kb", "filter_kb_by_type", "generate_report"},
        "generate_report": set(),
    }
    if previous_tool not in allowed_next:
        return True
    return next_tool in allowed_next[previous_tool]


def build_react_fallback_node(retrieval_service, web_search_service):
    """
    构建受控 ReAct fallback 节点。

    设计约束：
    - 单节点内部循环
    - 最多 3 轮
    - 每轮只执行 1 个 tool_call
    """
    tools = build_react_tools_schema()

    async def react_fallback_node(state: AgentState) -> dict[str, Any]:
        question = _latest_user_question(state.get("messages", []))
        working_context = str(state.get("working_context", "") or "")
        context_sources = list(state.get("context_sources", []) or [])
        jd_data = dict(state.get("jd_data") or {}) or None
        resume_data = dict(state.get("resume_data") or {}) or None
        tool_cache = dict(state.get("tool_cache", {}) or {})
        tool_trace = list(state.get("tool_trace", []) or [])
        react_handoff_agent = None
        max_iterations = 3

        logger.info(
            "ReAct fallback 开始: session=%s, signature=%s, question=%s",
            state.get("session_id", ""),
            state.get("question_signature", ""),
            question[:100],
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{question}\n\n"
                    f"已有 JD：{json.dumps(jd_data or {}, ensure_ascii=False)}\n\n"
                    f"已有简历：{json.dumps(resume_data or {}, ensure_ascii=False)}\n\n"
                    f"已有上下文：\n{working_context or '(无)'}"
                ),
            },
        ]

        def build_safe_fallback_text() -> str:
            if working_context:
                return (
                    "当前外部模型服务暂时不稳定，我先基于已经获取到的上下文给你一个简短结论：\n\n"
                    f"{working_context[:600]}"
                )
            if jd_data or resume_data:
                parts: list[str] = ["当前外部模型服务暂时不稳定，我先基于已有结构化信息给你一个简短说明。"]
                if jd_data and jd_data.get("summary"):
                    parts.append(f"JD 摘要：{jd_data.get('summary')}")
                if resume_data and resume_data.get("summary"):
                    parts.append(f"简历摘要：{resume_data.get('summary')}")
                if len(parts) > 1:
                    return "\n\n".join(parts)
            return "当前外部模型服务连接不稳定，请稍后重试。"

        async def run_tool(name: str, arguments: dict[str, Any], iteration: int) -> dict[str, Any]:
            nonlocal working_context, context_sources, jd_data, resume_data, tool_cache
            _emit_custom_event({"type": "tool_start", "tool": name})
            logger.info(
                "ReAct fallback tool 开始: iteration=%d, tool=%s, args=%s",
                iteration + 1,
                name,
                json.dumps(arguments, ensure_ascii=False),
            )
            result: dict[str, Any]

            if is_cacheable_tool(name):
                cache_key = build_tool_cache_key(
                    name,
                    arguments,
                    session_id=str(state.get("session_id", "")),
                    question_signature=str(state.get("question_signature", "")),
                )
                cached = tool_cache.get(cache_key)
                if cached is not None:
                    result = cached
                    if name in {"search_kb", "search_web"}:
                        context_sources = _dedupe_sources(context_sources, result.get("sources", []))
                        working_context = str(result.get("working_context", "") or working_context)
                    elif name == "extract_jd":
                        jd_data = result.get("jd_data") or jd_data
                    elif name == "extract_resume":
                        resume_data = result.get("resume_data") or resume_data
                    _emit_custom_event({"type": "tool_cache_hit", "tool": name})
                    logger.info(
                        "ReAct fallback tool 命中缓存: iteration=%d, tool=%s, summary=%s",
                        iteration + 1,
                        name,
                        _tool_result_summary(name, result),
                    )
                    tool_trace.append({"tool": name, "arguments": arguments, "iteration": iteration, "cache_hit": True})
                    _emit_custom_event({"type": "tool_result", "tool": name})
                    return result

            metadata_store = getattr(getattr(retrieval_service, "vector_store", None), "metadata_store", None)

            if name == "list_sources":
                sources = metadata_store.list_sources(
                    source_type=str(arguments.get("source_type") or "") or None,
                    category=str(arguments.get("category") or "") or None,
                ) if metadata_store else []
                result = {
                    "sources_list": sources,
                    "working_context": "\n".join(sources[:20]),
                }
                working_context = result["working_context"] or working_context
            elif name == "list_documents":
                documents = metadata_store.list_documents(
                    source_type=str(arguments.get("source_type") or "") or None,
                    category=str(arguments.get("category") or "") or None,
                ) if metadata_store else []
                result = {
                    "documents": documents,
                    "working_context": _summarize_documents(documents),
                }
                working_context = result["working_context"] or working_context
            elif name == "search_kb":
                query = str(arguments.get("query") or question)
                top_k = int(arguments.get("top_k") or 5)
                kb_results = retrieval_service.retrieve(query, top_k=top_k) if retrieval_service else []
                kb_sources = [
                    {
                        "content": item.get("content", ""),
                        "source": item.get("source", ""),
                        "score": item.get("score", 0.0),
                        "page": item.get("page"),
                        "type": "kb",
                    }
                    for item in kb_results
                ]
                context_sources = _dedupe_sources(context_sources, kb_sources)
                working_context = _summarize_context_sources(context_sources)
                result = {"sources": kb_sources, "working_context": working_context}
                if kb_sources:
                    _emit_custom_event({"type": "sources", "sources": kb_sources})
            elif name == "filter_kb_by_type":
                query = str(arguments.get("query") or question)
                top_k = int(arguments.get("top_k") or 5)
                source_type = str(arguments.get("source_type") or "") or None
                category = str(arguments.get("category") or "") or None
                kb_results = retrieval_service.retrieve(query, top_k=max(top_k * 4, top_k)) if retrieval_service else []
                filtered_sources = []
                for item in kb_results:
                    if source_type and str(item.get("source_type") or "") != source_type:
                        continue
                    if category and str(item.get("category") or "") != category:
                        continue
                    filtered_sources.append(
                        {
                            "content": item.get("content", ""),
                            "source": item.get("source", ""),
                            "score": item.get("score", 0.0),
                            "page": item.get("page"),
                            "type": "kb",
                            "source_type": item.get("source_type", ""),
                            "category": item.get("category", ""),
                        }
                    )
                    if len(filtered_sources) >= top_k:
                        break
                context_sources = _dedupe_sources(context_sources, filtered_sources)
                working_context = _summarize_context_sources(context_sources)
                result = {"sources": filtered_sources, "working_context": working_context}
                if filtered_sources:
                    _emit_custom_event({"type": "sources", "sources": filtered_sources})
            elif name == "search_web":
                query = str(arguments.get("query") or question)
                web_results = web_search_service.search(query, max_results=5) if web_search_service and web_search_service.is_available else []
                web_sources = [
                    {
                        "content": item.get("content", ""),
                        "source": item.get("source", ""),
                        "type": "web",
                    }
                    for item in web_results
                ]
                context_sources = _dedupe_sources(context_sources, web_sources)
                working_context = _summarize_context_sources(context_sources)
                result = {"sources": web_sources, "working_context": working_context}
                if web_sources:
                    _emit_custom_event({"type": "sources", "sources": web_sources})
            elif name == "extract_jd":
                raw_text = str(arguments.get("raw_text") or question)
                from app.agent.nodes.extract_jd import extract_jd
                extracted = extract_jd({"jd_data": {"raw_text": raw_text}})
                jd_data = extracted.get("jd_data") or jd_data
                result = {"jd_data": jd_data}
                if jd_data:
                    _emit_custom_event({"type": "extracted", "jd_data": jd_data})
            elif name == "extract_resume":
                raw_text = str(arguments.get("raw_text") or question)
                from app.agent.nodes.extract_resume import extract_resume
                extracted = extract_resume({"resume_data": {"raw_text": raw_text}})
                resume_data = extracted.get("resume_data") or resume_data
                result = {"resume_data": resume_data}
                if resume_data:
                    _emit_custom_event({"type": "extracted", "resume_data": resume_data})
            elif name == "generate_report":
                local_question = str(arguments.get("question") or question)
                local_context = str(arguments.get("working_context") or working_context)
                answer = generate_report_answer(
                    local_question,
                    working_context=local_context,
                    jd_data=jd_data,
                    resume_data=resume_data,
                )
                result = {"answer": answer}
            elif name == "match_resume_jd":
                if not jd_data or not resume_data:
                    result = {"error": "缺少 JD 或简历结构化信息，无法执行匹配"}
                else:
                    answer = generate_match_answer(
                        str(arguments.get("question") or question),
                        jd_data=jd_data,
                        resume_data=resume_data,
                    )
                    result = {"answer": answer}
            elif name == "compact_faiss":
                vector_store = getattr(retrieval_service, "vector_store", None)
                compact_result = vector_store.compact() if vector_store else {"before": 0, "after": 0, "removed": 0}
                result = {"compact_result": compact_result}
                working_context = (
                    f"FAISS compact 完成: before={compact_result.get('before', 0)}, "
                    f"after={compact_result.get('after', 0)}, removed={compact_result.get('removed', 0)}"
                )
            else:
                result = {"error": f"未知工具: {name}"}

            if is_cacheable_tool(name):
                cache_key = build_tool_cache_key(
                    name,
                    arguments,
                    session_id=str(state.get("session_id", "")),
                    question_signature=str(state.get("question_signature", "")),
                )
                tool_cache[cache_key] = result

            tool_trace.append({"tool": name, "arguments": arguments, "iteration": iteration, "cache_hit": False})
            logger.info(
                "ReAct fallback tool 完成: iteration=%d, tool=%s, summary=%s",
                iteration + 1,
                name,
                _tool_result_summary(name, result),
            )
            _emit_custom_event({"type": "tool_result", "tool": name})
            return result

        final_answer = ""
        for iteration in range(max_iterations):
            try:
                response = chat_completion_with_tools(
                    messages,
                    tools,
                    temperature=0.2,
                    max_tokens=1024,
                    tool_choice="auto",
                    thinking={"type": "disabled"},
                )
            except Exception as e:
                logger.warning("ReAct fallback tool 规划失败，降级到普通回答: %s", e)
                _emit_custom_event({"type": "status", "content": "工具规划失败，正在降级为直接回答"})
                try:
                    direct_messages = build_report_messages(
                        question,
                        working_context=working_context,
                        jd_data=jd_data,
                        resume_data=resume_data,
                    )
                    chunks: list[str] = []
                    async for delta in chat_completion_stream_async(
                        direct_messages,
                        temperature=0.3,
                        max_tokens=900,
                        thinking={"type": "disabled"},
                    ):
                        if not delta:
                            continue
                        chunks.append(delta)
                        _emit_custom_event({"type": "token", "content": delta})
                    final_answer = "".join(chunks).strip()
                    break
                except Exception as inner_e:
                    logger.warning("ReAct fallback 普通回答也失败，返回安全降级文本: %s", inner_e)
                    _emit_custom_event({"type": "status", "content": "外部模型服务不稳定，返回降级结果"})
                    final_answer = build_safe_fallback_text()
                    _emit_custom_event({"type": "token", "content": final_answer})
                    break

            content = str(response.get("content") or "")
            tool_calls = list(response.get("tool_calls") or [])

            logger.info(
                "ReAct fallback 规划结果: iteration=%d, tool_calls=%d, content_chars=%d",
                iteration + 1,
                len(tool_calls),
                len(content),
            )

            if not tool_calls:
                logger.info("ReAct fallback 直接收口: iteration=%d, reason=no_tool_calls", iteration + 1)
                final_answer = content.strip()
                break

            tool_call = tool_calls[0]
            tool_name = str(tool_call.get("name") or "")
            arguments = _safe_json_loads(str(tool_call.get("arguments") or "{}"))

            if not _is_allowed_tool_transition(tool_trace, tool_name):
                logger.info("ReAct fallback 拦截不允许的工具组合: prev=%s, next=%s", tool_trace[-1].get("tool") if tool_trace else "", tool_name)
                _emit_custom_event({"type": "status", "content": f"工具组合受限，改为直接生成结果"})
                final_answer = generate_report_answer(
                    question,
                    working_context=working_context,
                    jd_data=jd_data,
                    resume_data=resume_data,
                ).strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tool_call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )

            tool_result = await run_tool(tool_name, arguments, iteration)
            if tool_name == "match_resume_jd":
                logger.info("ReAct fallback 收口: reason=match_resume_jd_complete")
                final_answer = str(tool_result.get("answer") or "").strip()
                break
            if tool_name == "generate_report":
                logger.info("ReAct fallback 收口: reason=generate_report_complete")
                final_answer = str(tool_result.get("answer") or "").strip()
                break

            handoff_agent = _infer_handoff_agent(question, jd_data=jd_data, resume_data=resume_data)
            if handoff_agent and tool_name in {"extract_jd", "extract_resume"}:
                react_handoff_agent = handoff_agent
                logger.info(
                    "ReAct fallback handoff: iteration=%d, tool=%s, handoff_agent=%s",
                    iteration + 1,
                    tool_name,
                    handoff_agent,
                )
                _emit_custom_event({"type": "status", "content": f"识别为标准任务，切换到{handoff_agent}"})
                break

            if _has_enough_context_to_finalize(
                question,
                working_context=working_context,
                jd_data=jd_data,
                resume_data=resume_data,
                last_tool_name=tool_name,
                tool_trace=tool_trace,
            ):
                logger.info(
                    "ReAct fallback 收口: iteration=%d, reason=enough_context, last_tool=%s",
                    iteration + 1,
                    tool_name,
                )
                final_answer = generate_report_answer(
                    question,
                    working_context=working_context,
                    jd_data=jd_data,
                    resume_data=resume_data,
                ).strip()
                break

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": _stringify_tool_result(tool_result),
                }
            )

        if not final_answer:
            report_messages = build_report_messages(
                question,
                working_context=working_context,
                jd_data=jd_data,
                resume_data=resume_data,
            )
            try:
                chunks: list[str] = []
                async for delta in chat_completion_stream_async(
                    report_messages,
                    temperature=0.3,
                    max_tokens=900,
                    thinking={"type": "disabled"},
                ):
                    if not delta:
                        continue
                    chunks.append(delta)
                    _emit_custom_event({"type": "token", "content": delta})
                final_answer = "".join(chunks).strip()
            except Exception as e:
                logger.warning("ReAct fallback 最终生成失败，返回安全降级文本: %s", e)
                _emit_custom_event({"type": "status", "content": "最终生成失败，返回降级结果"})
                final_answer = build_safe_fallback_text()
                _emit_custom_event({"type": "token", "content": final_answer})
        else:
            # generate_report 走的是非流式 tool，这里补一次 token 流，保证前端体验一致。
            _emit_custom_event({"type": "token", "content": final_answer})

        logger.info(
            "ReAct fallback 完成: iterations=%d, tools=%d, handoff=%s, answer=%d字符",
            min(max_iterations, max(len(tool_trace), 0)),
            len(tool_trace),
            react_handoff_agent or "",
            len(final_answer),
        )
        return {
            "final_answer": final_answer,
            "tool_trace": tool_trace,
            "react_iterations": min(max_iterations, max(len(tool_trace), 0)),
            "working_context": working_context,
            "context_sources": context_sources,
            "jd_data": jd_data,
            "resume_data": resume_data,
            "tool_cache": tool_cache,
            "react_handoff_agent": react_handoff_agent,
            "execution_plan": [react_handoff_agent, "respond"] if react_handoff_agent else state.get("execution_plan", ["react_fallback", "respond"]),
            "current_step": 0 if react_handoff_agent else state.get("current_step", 0),
            "final_response_ready": False if react_handoff_agent else state.get("final_response_ready", False),
        }

    return react_fallback_node
