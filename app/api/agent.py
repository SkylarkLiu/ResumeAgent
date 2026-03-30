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

import asyncio
import base64
import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.analyze_jd import analyze_jd_stream
from app.agent.nodes.extract_jd import extract_jd
from app.agent.nodes.extract_resume import extract_resume
from app.agent.nodes.generate import generate_stream
from app.agent.nodes.generate_analysis import generate_analysis_stream
from app.agent.nodes.kb_search import search_kb
from app.agent.nodes.normalize import normalize_kb, normalize_web
from app.agent.nodes.retrieve_jd import retrieve_jd
from app.agent.nodes.router import route_query
from app.agent.nodes.web_search import search_web
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


async def _resume_stream_event_generator(
    resume_data: dict,
    question: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    简历分析流式事件生成器。

    手动编排节点调用：extract_resume → retrieve_jd → generate_analysis_stream

    SSE 事件格式：
    - data: {"type": "extracted", "resume_data": {...}}   - 简历提取完成
    - data: {"type": "sources", "sources": [...]}          - JD 来源
    - data: {"type": "token", "content": "..."}            - 增量文本
    - data: {"type": "done", "session_id": "...", "resume_data": {...}} - 完成
    - data: {"type": "error", "message": "..."}            - 错误
    """
    try:
        input_state = {
            "messages": [HumanMessage(content=question)],
            "context_sources": [],
            "working_context": "",
            "final_answer": "",
            "resume_data": resume_data,
        }

        # === 阶段 1: 提取简历 ===
        logger.info("简历分析流式: 开始提取简历...")
        extract_result = await asyncio.to_thread(extract_resume, input_state)
        extracted_resume = extract_result.get("resume_data", resume_data)

        # 发送提取结果
        resume_summary = {k: v for k, v in extracted_resume.items() if k != "raw_text"}
        yield f"data: {json.dumps({'type': 'extracted', 'resume_data': resume_summary})}\n\n"

        if extracted_resume.get("extract_error"):
            error_msg = extracted_resume["extract_error"]
            error_answer = f"❌ 简历解析失败：{error_msg}\n\n请尝试：\n1. 重新上传简历文件\n2. 将简历内容粘贴到输入框中"
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'answer': error_answer, 'resume_data': resume_summary})}\n\n"
            return

        # === 阶段 2: 检索 JD ===
        logger.info("简历分析流式: 开始检索 JD...")
        retrieve_state = {**input_state, **extract_result}
        retrieve_result = await asyncio.to_thread(retrieve_jd, retrieve_state)
        context_sources = retrieve_result.get("context_sources", [])

        sources_api = _build_sources(context_sources)
        yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources_api]})}\n\n"

        # === 阶段 3: 流式生成分析报告 ===
        logger.info("简历分析流式: 开始生成分析报告...")
        gen_state = {**retrieve_result}

        final_answer = ""
        async for event in generate_analysis_stream(gen_state):
            if event["type"] == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': _sanitize_sse_content(event['content'])})}\n\n"
            elif event["type"] == "done":
                final_answer = event.get("final_answer", "")
            elif event["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"

        # === 阶段 4: 更新 checkpointer ===
        if _agent_graph:
            try:
                config = {"configurable": {"thread_id": session_id}}
                ai_message = AIMessage(content=final_answer)
                current_message = HumanMessage(content=question)
                _agent_graph.update_state(
                    config,
                    {"messages": [current_message, ai_message]},
                )
                logger.info("简历分析: 已更新 checkpointer, thread=%s", session_id)
            except Exception as update_err:
                logger.error("简历分析: 更新 checkpointer 失败: %s", update_err, exc_info=True)

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'answer': final_answer, 'resume_data': resume_summary})}\n\n"
        logger.info("简历分析流式完成: session=%s, answer=%d字符", session_id, len(final_answer))

    except Exception as e:
        logger.error("简历分析流式异常: %s", e, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


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

    input_state = {
        "messages": [HumanMessage(content=question)],
        "context_sources": [],
        "working_context": "",
        "final_answer": "",
    }

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
    1. 执行路由决策（非流式）
    2. 根据路由执行检索/搜索（非流式）
    3. 流式生成回答（yield 每个 token）

    SSE 事件格式：
    - data: {"type": "route", "route": "retrieve", "task": "qa"}
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
            # === 阶段 1: 从 checkpointer 加载历史 ===
            history_messages = []
            if _checkpointer:
                snapshot = _checkpointer.get(config)
                if snapshot:
                    channel_values = snapshot.get("channel_values", {})
                    history_messages = channel_values.get("messages", [])

            logger.info(
                "Agent 流式请求: thread=%s, question=%s, history=%d条",
                session_id,
                question[:50],
                len(history_messages),
            )

            # 追加当前用户消息
            current_message = HumanMessage(content=question)
            messages = history_messages + [current_message]

            # === 阶段 2: 路由决策 ===
            logger.debug("执行路由决策...")
            route_state = {
                "messages": messages,
                "route_type": "direct",
                "task_type": "qa",
            }
            route_result = route_query(route_state)
            route_type = route_result.get("route_type", "direct")
            task_type = route_result.get("task_type", "qa")

            route_str = route_type.value if hasattr(route_type, "value") else str(route_type)
            task_str = task_type.value if hasattr(task_type, "value") else str(task_type)

            logger.info("路由决策: route=%s, task=%s", route_str, task_str)
            yield f"data: {json.dumps({'type': 'route', 'route': route_str, 'task': task_str})}\n\n"

            # === 阶段 3: 根据路由执行检索/搜索 ===
            context_sources = []
            working_context = ""

            if route_str == "retrieve":
                # KB 检索
                logger.debug("执行 KB 检索...")
                kb_state = {
                    "messages": messages,
                    "route_type": route_type,
                    "task_type": task_type,
                    "context_sources": [],
                    "working_context": "",
                }
                kb_result = search_kb(kb_state)
                context_sources = kb_result.get("context_sources", [])
                norm_result = normalize_kb(kb_result)
                working_context = norm_result.get("working_context", "")
                logger.info("KB 检索完成: %d 条结果", len(context_sources))

            elif route_str == "web":
                # Web 搜索
                logger.debug("执行 Web 搜索...")
                web_state = {
                    "messages": messages,
                    "route_type": route_type,
                    "task_type": task_type,
                    "context_sources": [],
                    "working_context": "",
                }
                web_result = search_web(web_state)
                context_sources = web_result.get("context_sources", [])
                norm_result = normalize_web(web_result)
                working_context = norm_result.get("working_context", "")
                logger.info("Web 搜索完成: %d 条结果", len(context_sources))

            # 发送 sources 事件
            sources_api = _build_sources(context_sources)
            yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources_api]})}\n\n"

            # === 阶段 4: 流式生成 ===
            logger.info("开始流式生成...")
            gen_state = {
                "messages": messages,
                "route_type": route_type,
                "task_type": task_type,
                "context_sources": context_sources,
                "working_context": working_context,
                "final_answer": "",
            }

            final_answer = ""
            ai_message = None
            async for event in generate_stream(gen_state):
                if event["type"] == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': _sanitize_sse_content(event['content'])})}\n\n"
                elif event["type"] == "done":
                    final_answer = event.get("final_answer", "")
                    ai_message = event["messages"][0] if event.get("messages") else None

            # === 阶段 5: 更新 checkpointer ===
            logger.debug("阶段 5: 更新 checkpointer, graph=%s, ai_msg=%s", _agent_graph is not None, ai_message is not None)
            if _agent_graph and ai_message:
                # 使用 update_state 更新历史（追加消息）
                # 注意：update_state 是同步方法，在异步上下文中直接调用
                try:
                    logger.debug("调用 update_state: config=%s, messages=%d", config, 2)
                    result_config = _agent_graph.update_state(
                        config,
                        {"messages": [current_message, ai_message]},
                    )
                    logger.info("已更新 checkpointer: thread=%s, result_config=%s", session_id, result_config)

                    # 验证是否成功
                    verify_state = _checkpointer.get(config) if _checkpointer else None
                    if verify_state:
                        verify_messages = verify_state.get("channel_values", {}).get("messages", [])
                        logger.debug("验证: checkpointer 现有 %d 条消息", len(verify_messages))
                    else:
                        logger.warning("验证失败: checkpointer.get() 返回 None")

                except Exception as update_err:
                    logger.error("更新 checkpointer 失败: %s", update_err, exc_info=True)
            else:
                logger.warning("跳过 checkpointer 更新: graph=%s, ai_msg=%s", _agent_graph is not None, ai_message is not None)

            # 发送 done 事件
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

            logger.info(
                "Agent 流式回复完成: thread=%s, route=%s, answer=%d字符",
                session_id,
                route_str,
                len(final_answer),
            )

        except Exception as e:
            logger.error("Agent 流式执行异常: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

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

    手动编排节点调用：extract_jd → analyze_jd_stream

    SSE 事件格式：
    - data: {"type": "extracted", "jd_data": {...}}       - JD 提取完成
    - data: {"type": "token", "content": "..."}            - 增量文本
    - data: {"type": "done", "session_id": "...", "answer": "...", "jd_data": {...}}
    - data: {"type": "error", "message": "..."}
    """
    try:
        input_state = {
            "messages": [HumanMessage(content=question)],
            "context_sources": [],
            "working_context": "",
            "final_answer": "",
            "jd_data": jd_data,
        }

        # === 阶段 1: 提取 JD ===
        logger.info("JD 分析流式: 开始提取 JD...")
        extract_result = await asyncio.to_thread(extract_jd, input_state)
        extracted_jd = extract_result.get("jd_data", jd_data)

        # 发送提取结果
        jd_summary = {k: v for k, v in extracted_jd.items() if k not in ("raw_text",)}
        yield f"data: {json.dumps({'type': 'extracted', 'jd_data': jd_summary})}\n\n"

        if extracted_jd.get("extract_error"):
            error_msg = extracted_jd["extract_error"]
            error_answer = f"❌ JD 解析失败：{error_msg}\n\n请尝试：\n1. 重新粘贴 JD 内容\n2. 确保包含完整的岗位描述信息"
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'answer': error_answer, 'jd_data': jd_summary})}\n\n"
            return

        # === 阶段 2: 流式生成分析报告 ===
        logger.info("JD 分析流式: 开始生成分析报告...")
        gen_state = {**input_state, **extract_result}

        final_answer = ""
        async for event in analyze_jd_stream(gen_state):
            if event["type"] == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': _sanitize_sse_content(event['content'])})}\n\n"
            elif event["type"] == "done":
                final_answer = event.get("final_answer", "")
            elif event["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"

        # === 阶段 3: 更新 checkpointer ===
        if _agent_graph:
            try:
                config = {"configurable": {"thread_id": session_id}}
                ai_message = AIMessage(content=final_answer)
                current_message = HumanMessage(content=question)
                _agent_graph.update_state(
                    config,
                    {"messages": [current_message, ai_message]},
                )
                logger.info("JD 分析: 已更新 checkpointer, thread=%s", session_id)
            except Exception as update_err:
                logger.error("JD 分析: 更新 checkpointer 失败: %s", update_err, exc_info=True)

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'answer': final_answer, 'jd_data': jd_summary})}\n\n"
        logger.info("JD 分析流式完成: session=%s, answer=%d字符", session_id, len(final_answer))

    except Exception as e:
        logger.error("JD 分析流式异常: %s", e, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


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
    if _checkpointer is None:
        return SessionInfo(session_id=session_id, message_count=0)

    try:
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = _checkpointer.get(config)
        if state_snapshot:
            # langgraph 1.1.x: MemorySaver.get() 返回 dict，数据在 channel_values 中
            channel_values = state_snapshot.get("channel_values", {})
            messages = channel_values.get("messages", [])
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
