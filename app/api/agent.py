"""
Agent API 路由 - /agent/chat + /agent/session + /agent/resume-analysis + /agent/resume-upload + /agent/jd-analysis

阶段 3 改造：新增简历分析接口
- /agent/resume-analysis: 文本粘贴方式分析简历
- /agent/resume-upload: 文件上传方式分析简历（PDF/图片/文本）

阶段 4 改造：新增流式输出接口 + JD 分析接口
- /agent/chat/stream: SSE 流式返回 Agent 回答
- /agent/jd-analysis: JD 岗位分析（文本粘贴/文件上传）
"""
from __future__ import annotations

import base64
import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import get_jd_analysis_subgraph, get_resume_analysis_subgraph
from app.core.config import get_settings
from app.core.logger import setup_logger
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentSourceItem,
    JDAnalysisRequest,
    JDAnalysisResponse,
    ResumeAnalysisRequest,
    ResumeAnalysisResponse,
    SessionInfo,
)

logger = setup_logger("api.agent")
settings = get_settings()

router = APIRouter(prefix="/agent", tags=["Agent"])

# 模块级依赖注入点，由 main.py 在 lifespan 中设置
_agent_graph = None
_checkpointer = None


def set_agent_graph(graph) -> None:
    """注入编译后的 Agent 图（由 main.py 调用）"""
    global _agent_graph
    _agent_graph = graph


def set_checkpointer(checkpointer) -> None:
    """注入 checkpointer（由 main.py 调用，用于 get_state 查询）"""
    global _checkpointer
    _checkpointer = checkpointer


# ---- 工具函数 ----

# 匹配 SSE 不安全的控制字符（保留普通空白：空格、普通换行、制表）
_SSE_UNSAFE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_sse_content(text: str) -> str:
    """清理 SSE payload 中的不安全控制字符，防止前端解析异常。"""
    return _SSE_UNSAFE_RE.sub("", text)


def _ensure_session(session_id: str) -> str:
    """确保 session_id 有效，为空时生成新的"""
    sid = session_id.strip()
    if not sid:
        sid = uuid.uuid4().hex
        logger.info("生成新 session_id (thread_id): %s", sid)
    return sid


def _build_sources(context_sources: list[dict]) -> list[AgentSourceItem]:
    """将 context_sources 转为 API 响应格式"""
    return [
        AgentSourceItem(
            content=src.get("content", ""),
            source=src.get("source", ""),
            score=src.get("score", 0.0),
            type=src.get("type", "kb"),
        )
        for src in context_sources
    ]


def _sse_event(payload: dict[str, Any]) -> str:
    """格式化 SSE data 事件。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_chat_turn_input_state(question: str) -> dict[str, Any]:
    """
    为每一轮新的 chat 请求构造初始状态。

    注意：
    - messages 追加当前用户消息
    - jd_data / resume_data / 历史消息 由 checkpointer 自动保留
    - execution_plan / current_step / active_agent 等计划态必须每轮重置，
      否则会错误复用上一轮的调度结果
    """
    return {
        "messages": [HumanMessage(content=question)],
        "context_sources": [],
        "working_context": "",
        "final_answer": "",
        "execution_plan": [],
        "current_step": 0,
        "max_steps": 3,
        "active_agent": None,
        "final_response_ready": False,
    }


async def _load_session_values(session_id: str) -> dict[str, Any]:
    """读取当前 session 已持久化的状态值。"""
    if _agent_graph is None:
        return {}

    try:
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = await _agent_graph.aget_state(config)
        if state_snapshot:
            return getattr(state_snapshot, "values", {}) or {}
    except Exception as e:
        logger.warning("读取 session 状态失败: session=%s, error=%s", session_id, e)
    return {}


async def _persist_analysis_state(
    *,
    session_id: str,
    question: str,
    final_answer: str,
    ai_messages: list | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """把分析子图结果写回主图 checkpointer，保留消息和结构化数据。"""
    if _agent_graph is None:
        return

    try:
        config = {"configurable": {"thread_id": session_id}}
        update_payload = dict(extra_state or {})
        task_type = str(update_payload.get("task_type", ""))
        as_node = (
            "resume_expert" if task_type == "resume_analysis"
            else "jd_expert" if task_type == "jd_analysis"
            else "generate_final"
        )

        if ai_messages:
            update_payload["messages"] = [HumanMessage(content=question), *ai_messages]
        else:
            update_payload["messages"] = [HumanMessage(content=question), AIMessage(content=final_answer)]

        if hasattr(_agent_graph, "aupdate_state"):
            await _agent_graph.aupdate_state(config, update_payload, as_node=as_node)
        else:
            _agent_graph.update_state(config, update_payload, as_node=as_node)
        logger.info("分析结果已更新 checkpointer: thread=%s, keys=%s", session_id, list(update_payload.keys()))
    except Exception as update_err:
        logger.error("分析结果更新 checkpointer 失败: %s", update_err, exc_info=True)


async def _resume_stream_event_generator(
    resume_data: dict,
    question: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    简历分析流式事件生成器。

    通过简历分析子图执行：extract_resume → resolve_jd_context → generate_analysis

    SSE 事件格式：
    - data: {"type": "extracted", "resume_data": {...}}   - 简历提取完成
    - data: {"type": "sources", "sources": [...]}          - JD 来源
    - data: {"type": "token", "content": "..."}            - 增量文本
    - data: {"type": "done", "session_id": "...", "resume_data": {...}} - 完成
    - data: {"type": "error", "message": "..."}            - 错误
    """
    try:
        subgraph = get_resume_analysis_subgraph()
        if subgraph is None:
            yield _sse_event({"type": "error", "message": "简历分析子图未初始化，请检查服务启动日志。"})
            return

        config = {"configurable": {"thread_id": session_id}}
        session_values = await _load_session_values(session_id)
        input_state = {
            "messages": [HumanMessage(content=question)],
            "context_sources": [],
            "working_context": "",
            "final_answer": "",
            "resume_data": resume_data,
            "jd_data": session_values.get("jd_data"),
        }

        extracted_resume = resume_data
        resume_summary = {k: v for k, v in resume_data.items() if k != "raw_text"}
        context_sources: list[dict] = []
        working_context = ""
        final_answer = ""
        ai_messages = None
        error_emitted = False

        async for mode, payload in subgraph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom":
                event_type = payload.get("type")
                if event_type == "extracted":
                    resume_summary = payload.get("resume_data", resume_summary)
                    yield _sse_event({"type": "extracted", "resume_data": resume_summary})
                elif event_type == "sources":
                    context_sources = payload.get("sources", [])
                    sources_api = _build_sources(context_sources)
                    yield _sse_event({"type": "sources", "sources": [s.model_dump() for s in sources_api]})
                elif event_type == "token":
                    yield _sse_event({"type": "token", "content": _sanitize_sse_content(payload.get("content", ""))})
                elif event_type == "status":
                    yield _sse_event({"type": "status", "content": payload.get("content", "")})
                elif event_type == "error":
                    error_emitted = True
                    yield _sse_event({"type": "error", "message": payload.get("message", "未知错误")})
            elif mode == "updates":
                if "resolve_jd_context" in payload:
                    context_sources = payload["resolve_jd_context"].get("context_sources", context_sources)
                    working_context = payload["resolve_jd_context"].get("working_context", working_context)
                if "generate_analysis" in payload:
                    final_answer = payload["generate_analysis"].get("final_answer", final_answer)
                    ai_messages = payload["generate_analysis"].get("messages")
                if "extract_resume" in payload:
                    extracted_resume = payload["extract_resume"].get("resume_data", extracted_resume)

        if error_emitted:
            return

        await _persist_analysis_state(
            session_id=session_id,
            question=question,
            final_answer=final_answer,
            ai_messages=ai_messages,
            extra_state={
                "task_type": "resume_analysis",
                "resume_data": extracted_resume,
                "context_sources": context_sources,
                "working_context": working_context,
                "final_answer": final_answer,
            },
        )

        yield _sse_event({"type": "done", "session_id": session_id, "answer": final_answer, "resume_data": resume_summary})
        logger.info("简历分析流式完成: session=%s, answer=%d字符", session_id, len(final_answer))

    except Exception as e:
        logger.error("简历分析流式异常: %s", e, exc_info=True)
        yield _sse_event({"type": "error", "message": str(e)})


# ---- 原有接口 ----

@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Agent 多轮对话接口。

    核心机制：
    1. session_id 作为 thread_id，通过 config 传入 ainvoke
    2. checkpointer 自动加载该 thread 的历史 messages 并追加新消息
    3. 每轮只需传入当前用户消息（HumanMessage），无需手动拼装历史
    4. 图执行完成后，checkpointer 自动持久化新的 state
    """
    if _agent_graph is None:
        return AgentChatResponse(
            answer="Agent 服务未初始化，请检查服务启动日志。",
            session_id=request.session_id,
            route_type="direct",
            task_type="qa",
        )

    session_id = _ensure_session(request.session_id)
    question = request.question

    config = {"configurable": {"thread_id": session_id}}

    input_state = _build_chat_turn_input_state(question)

    try:
        result = await _agent_graph.ainvoke(input_state, config=config)

        answer = result.get("final_answer", "")
        route_type = result.get("route_type", "direct")
        task_type = result.get("task_type", "qa")
        context_sources = result.get("context_sources", [])

        sources = _build_sources(context_sources)

        logger.info(
            "Agent 回复: thread=%s, route=%s, answer=%d字符, sources=%d条",
            session_id,
            route_type.value if hasattr(route_type, "value") else str(route_type),
            len(answer),
            len(sources),
        )

        return AgentChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            route_type=route_type.value if hasattr(route_type, "value") else str(route_type),
            task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
        )

    except Exception as e:
        logger.error("Agent 执行异常: %s", e, exc_info=True)
        return AgentChatResponse(
            answer=f"Agent 执行出错：{e}",
            session_id=session_id,
            route_type="direct",
            task_type="qa",
        )


@router.post("/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """
    Agent 多轮对话流式接口（SSE）。

    流程：
    1. 调用主图执行路由 / 检索 / 标准化 / 生成
    2. 将图的 updates/custom 事件翻译成 SSE
    3. checkpointer 由主图自动持久化

    SSE 事件格式：
    - data: {"type": "route", "route": "retrieve", "task": "qa"}
    - data: {"type": "agent_start", "agent": "qa_flow|jd_expert|resume_expert"}
    - data: {"type": "agent_result", "agent": "..."}
    - data: {"type": "sources", "sources": [...]}
    - data: {"type": "token", "content": "..."}  (多次)
    - data: {"type": "done", "session_id": "..."}
    """
    if _agent_graph is None:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 服务未初始化'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session_id = _ensure_session(request.session_id)
    question = request.question
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        try:
            logger.info("Agent 流式请求: thread=%s, question=%s", session_id, question[:50])

            input_state = _build_chat_turn_input_state(question)

            route_str = "direct"
            task_str = "qa"
            final_answer = ""
            sources_sent = False

            async for mode, payload in _agent_graph.astream(
                input_state,
                config=config,
                stream_mode=["custom", "updates"],
            ):
                if mode == "custom":
                    event_type = payload.get("type")
                    if event_type == "token":
                        yield _sse_event({"type": "token", "content": _sanitize_sse_content(payload.get("content", ""))})
                    elif event_type == "agent_start":
                        yield _sse_event({"type": "agent_start", "agent": payload.get("agent", "")})
                    elif event_type == "agent_result":
                        yield _sse_event({"type": "agent_result", "agent": payload.get("agent", "")})
                    elif event_type == "agent_cache_hit":
                        yield _sse_event({"type": "agent_cache_hit", "agent": payload.get("agent", "")})
                    elif event_type == "status":
                        yield _sse_event({"type": "status", "content": payload.get("content", "")})
                    elif event_type == "sources":
                        sources_api = _build_sources(payload.get("sources", []))
                        yield _sse_event({"type": "sources", "sources": [s.model_dump() for s in sources_api]})
                        sources_sent = True
                    elif event_type == "extracted":
                        if "resume_data" in payload:
                            yield _sse_event({"type": "extracted", "resume_data": payload.get("resume_data", {})})
                        elif "jd_data" in payload:
                            yield _sse_event({"type": "extracted", "jd_data": payload.get("jd_data", {})})
                    elif event_type == "error":
                        yield _sse_event({"type": "error", "message": payload.get("message", "未知错误")})
                elif mode == "updates":
                    if "supervisor_plan" in payload:
                        route_type = payload["supervisor_plan"].get("route_type", route_str)
                        task_type = payload["supervisor_plan"].get("task_type", task_str)
                        route_str = route_type.value if hasattr(route_type, "value") else str(route_type)
                        task_str = task_type.value if hasattr(task_type, "value") else str(task_type)
                        logger.info("路由决策: route=%s, task=%s", route_str, task_str)
                        yield _sse_event({"type": "route", "route": route_str, "task": task_str})
                        if route_str == "direct" and not sources_sent:
                            yield _sse_event({"type": "sources", "sources": []})
                            sources_sent = True

                    if "qa_flow" in payload:
                        final_answer = payload["qa_flow"].get("final_answer", final_answer)
                    if "resume_expert" in payload:
                        final_answer = payload["resume_expert"].get("final_answer", final_answer)
                    if "jd_expert" in payload:
                        final_answer = payload["jd_expert"].get("final_answer", final_answer)
                    if "generate_final" in payload:
                        final_answer = payload["generate_final"].get("final_answer", final_answer)

            if not sources_sent:
                yield _sse_event({"type": "sources", "sources": []})

            yield _sse_event({"type": "done", "session_id": session_id, "answer": final_answer})

            logger.info(
                "Agent 流式回复完成: thread=%s, route=%s, answer=%d字符",
                session_id,
                route_str,
                len(final_answer),
            )

        except Exception as e:
            logger.error("Agent 流式执行异常: %s", e, exc_info=True)
            yield _sse_event({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---- 简历分析接口（流式 SSE） ----

@router.post("/resume-analysis")
async def resume_analysis_text(request: ResumeAnalysisRequest):
    """
    简历分析接口（文本粘贴方式）- SSE 流式输出。

    用户直接粘贴简历文本，后端提取 → 检索 JD → 流式生成分析报告。

    SSE 事件格式：
    - data: {"type": "extracted", "resume_data": {...}}   - 简历提取完成
    - data: {"type": "sources", "sources": [...]}          - JD 来源
    - data: {"type": "token", "content": "..."}            - 增量文本
    - data: {"type": "done", "session_id": "...", "answer": "...", "resume_data": {...}}
    - data: {"type": "error", "message": "..."}
    """
    if _agent_graph is None:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 服务未初始化，请检查服务启动日志。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    if not request.resume_text.strip():
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': '请提供简历文本内容。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session_id = _ensure_session(request.session_id)

    resume_data = {
        "raw_text": request.resume_text.strip(),
        "target_position": request.target_position.strip(),
    }

    return StreamingResponse(
        _resume_stream_event_generator(resume_data, request.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/resume-upload")
async def resume_analysis_upload(
    file: UploadFile = File(..., description="简历文件（PDF/图片/文本）"),
    question: str = Form(default="请对我的简历进行全面分析评估", description="分析要求"),
    session_id: str = Form(default="", description="会话 ID"),
    target_position: str = Form(default="", description="目标岗位"),
):
    """
    简历分析接口（文件上传方式）- SSE 流式输出。

    支持上传 PDF、图片（PNG/JPG）、文本文件。
    文件在内存中处理，不做持久化存储。
    """
    if _agent_graph is None:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 服务未初始化，请检查服务启动日志。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 校验文件类型
    allowed_ext = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed_ext:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'不支持的文件格式：{ext}。请上传 PDF、图片或文本文件。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 校验文件大小
    content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'文件过大（{len(content) / 1024 / 1024:.1f}MB），最大支持 {settings.max_upload_size_mb}MB。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session_id = _ensure_session(session_id)

    # 根据文件类型构建 resume_data
    if ext in (".txt", ".md"):
        try:
            raw_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = content.decode("gbk", errors="replace")
        resume_data = {
            "raw_text": raw_text,
            "target_position": target_position.strip(),
        }
    elif ext in (".png", ".jpg", ".jpeg"):
        b64 = base64.b64encode(content).decode("utf-8")
        resume_data = {
            "file_base64": b64,
            "target_position": target_position.strip(),
        }
    else:
        import tempfile
        import os

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, f"resume_{uuid.uuid4().hex[:8]}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(content)
        resume_data = {
            "file_path": tmp_path,
            "target_position": target_position.strip(),
        }

    return StreamingResponse(
        _resume_stream_event_generator(resume_data, question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---- JD 分析接口 ----

async def _jd_stream_event_generator(
    jd_data: dict,
    question: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    JD 分析流式事件生成器。

    通过 JD 分析子图执行：extract_jd → analyze_jd

    SSE 事件格式：
    - data: {"type": "extracted", "jd_data": {...}}       - JD 提取完成
    - data: {"type": "token", "content": "..."}            - 增量文本
    - data: {"type": "done", "session_id": "...", "answer": "...", "jd_data": {...}}
    - data: {"type": "error", "message": "..."}
    """
    try:
        subgraph = get_jd_analysis_subgraph()
        if subgraph is None:
            yield _sse_event({"type": "error", "message": "JD 分析子图未初始化，请检查服务启动日志。"})
            return

        config = {"configurable": {"thread_id": session_id}}
        session_values = await _load_session_values(session_id)
        input_state = {
            "messages": [HumanMessage(content=question)],
            "context_sources": [],
            "working_context": "",
            "final_answer": "",
            "jd_data": jd_data,
            "resume_data": session_values.get("resume_data"),
        }

        extracted_jd = jd_data
        jd_summary = {k: v for k, v in jd_data.items() if k != "raw_text"}
        final_answer = ""
        ai_messages = None
        error_emitted = False

        async for mode, payload in subgraph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom":
                event_type = payload.get("type")
                if event_type == "extracted":
                    jd_summary = payload.get("jd_data", jd_summary)
                    yield _sse_event({"type": "extracted", "jd_data": jd_summary})
                elif event_type == "token":
                    yield _sse_event({"type": "token", "content": _sanitize_sse_content(payload.get("content", ""))})
                elif event_type == "status":
                    yield _sse_event({"type": "status", "content": payload.get("content", "")})
                elif event_type == "error":
                    error_emitted = True
                    yield _sse_event({"type": "error", "message": payload.get("message", "未知错误")})
            elif mode == "updates":
                if "extract_jd" in payload:
                    extracted_jd = payload["extract_jd"].get("jd_data", extracted_jd)
                if "analyze_jd" in payload:
                    final_answer = payload["analyze_jd"].get("final_answer", final_answer)
                    ai_messages = payload["analyze_jd"].get("messages")

        if error_emitted:
            return

        await _persist_analysis_state(
            session_id=session_id,
            question=question,
            final_answer=final_answer,
            ai_messages=ai_messages,
            extra_state={
                "task_type": "jd_analysis",
                "jd_data": extracted_jd,
                "final_answer": final_answer,
            },
        )

        yield _sse_event({"type": "done", "session_id": session_id, "answer": final_answer, "jd_data": jd_summary})
        logger.info("JD 分析流式完成: session=%s, answer=%d字符", session_id, len(final_answer))

    except Exception as e:
        logger.error("JD 分析流式异常: %s", e, exc_info=True)
        yield _sse_event({"type": "error", "message": str(e)})


# ---- JD 分析接口（流式 SSE） ----

@router.post("/jd-analysis")
async def jd_analysis_text(request: JDAnalysisRequest):
    """
    JD 分析接口（文本粘贴方式）- SSE 流式输出。

    用户直接粘贴 JD 文本，后端提取 → 流式生成岗位解读 + 简历写作建议。
    JD 数据自动存入 session state，后续同一 session 的简历分析将使用真实 JD。

    SSE 事件格式：
    - data: {"type": "extracted", "jd_data": {...}}
    - data: {"type": "token", "content": "..."}
    - data: {"type": "done", "session_id": "...", "answer": "...", "jd_data": {...}}
    - data: {"type": "error", "message": "..."}
    """
    if _agent_graph is None:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 服务未初始化，请检查服务启动日志。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    if not request.jd_text.strip():
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': '请提供 JD 岗位描述文本内容。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session_id = _ensure_session(request.session_id)

    jd_data = {
        "raw_text": request.jd_text.strip(),
    }

    return StreamingResponse(
        _jd_stream_event_generator(jd_data, request.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/jd-upload")
async def jd_analysis_upload(
    file: UploadFile = File(..., description="JD 文件（PDF/图片/文本）"),
    question: str = Form(default="请分析该岗位的核心要求并给出简历写作建议", description="分析要求"),
    session_id: str = Form(default="", description="会话 ID"),
):
    """
    JD 分析接口（文件上传方式）- SSE 流式输出。

    支持上传 PDF、图片（PNG/JPG）、文本文件。
    """
    if _agent_graph is None:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 服务未初始化，请检查服务启动日志。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 校验文件类型
    allowed_ext = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed_ext:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'不支持的文件格式：{ext}。请上传 PDF、图片或文本文件。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 校验文件大小
    content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'文件过大（{len(content) / 1024 / 1024:.1f}MB），最大支持 {settings.max_upload_size_mb}MB。'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session_id = _ensure_session(session_id)

    # 根据文件类型构建 jd_data
    if ext in (".txt", ".md"):
        try:
            raw_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = content.decode("gbk", errors="replace")
        jd_data = {"raw_text": raw_text}
    elif ext in (".png", ".jpg", ".jpeg"):
        b64 = base64.b64encode(content).decode("utf-8")
        jd_data = {"raw_text": "", "file_base64": b64, "is_image": True}
    else:
        import os
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, f"jd_{uuid.uuid4().hex[:8]}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(content)
        jd_data = {"raw_text": "", "file_path": tmp_path, "is_file": True}

    # 对于图片/PDF，需要先提取文本再走 JD 分析
    if jd_data.get("is_image"):
        from app.agent.nodes.extract_resume import _extract_from_image_base64
        raw_text = _extract_from_image_base64(jd_data["file_base64"])
        if raw_text:
            jd_data["raw_text"] = raw_text
            del jd_data["file_base64"]
        else:
            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'message': '图片 JD 文本提取失败，请尝试粘贴 JD 文本或上传 PDF/文本文件。'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")
    elif jd_data.get("is_file"):
        from app.agent.nodes.extract_resume import _extract_from_file
        raw_text = _extract_from_file(jd_data["file_path"])
        if raw_text:
            jd_data["raw_text"] = raw_text
            del jd_data["file_path"]
        else:
            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'message': '文件 JD 文本提取失败，请尝试粘贴 JD 文本或上传其他格式文件。'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")

    return StreamingResponse(
        _jd_stream_event_generator(jd_data, question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---- 会话管理接口 ----

@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """获取会话信息（通过 checkpointer 查询 thread 快照）"""
    if _agent_graph is None or _checkpointer is None:
        return SessionInfo(session_id=session_id, message_count=0)

    try:
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = await _agent_graph.aget_state(config)
        if state_snapshot:
            values = getattr(state_snapshot, "values", {}) or {}
            messages = values.get("messages", [])
            return SessionInfo(session_id=session_id, message_count=len(messages))
    except Exception as e:
        logger.warning("查询 thread 状态失败: session=%s, error=%s", session_id, e)

    return SessionInfo(session_id=session_id, message_count=0)


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    清空指定会话（清除 thread 的 checkpoint 数据）。

    注：MemorySaver 没有原生的 delete_thread API，
    这里通过写入空 messages 来重置 thread 状态。
    后续如需真正删除，可使用 SqliteSaver 或其他持久化 checkpointer。
    """
    if _agent_graph is None:
        return {"status": "ok", "session_id": session_id}

    try:
        config = {"configurable": {"thread_id": session_id}}
        # MemorySaver 没有原生的 delete/清空 API
        # 通过直接写入空状态来重置（需要操作 checkpointer 内部存储）
        if _checkpointer and hasattr(_checkpointer, "storage"):
            _checkpointer.storage.pop(config["configurable"]["thread_id"], None)
            logger.info("会话已重置: %s", session_id)
        else:
            logger.warning("checkpointer 不支持清空操作，跳过重置")
    except Exception as e:
        logger.warning("重置 thread 失败: session=%s, error=%s", session_id, e)

    return {"status": "ok", "session_id": session_id}
