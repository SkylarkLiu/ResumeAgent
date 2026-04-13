"""
FastAPI 应用入口
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router, set_rag_service
from app.api.ingest import router as ingest_router, set_vector_store
from app.agent import (
    build_agent_graph,
    get_cache_store_backend,
    get_checkpointer,
    get_checkpointer_backend,
    init_cache_store,
    init_checkpointer,
    shutdown_cache_store,
    shutdown_checkpointer,
)
from app.api.agent import router as agent_router, set_agent_graph, set_checkpointer
from app.api.debug import router as debug_router
from app.core.config import get_settings
from app.core.logger import setup_logger
from app.repositories.metadata_store import PostgresMetadataStore
from app.repositories.vector_store import FAISSVectorStore
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.web_search_service import WebSearchService

logger = setup_logger("main")
settings = get_settings()

# 全局向量存储和 RAG 服务
_store: FAISSVectorStore | None = None
_rag: RAGService | None = None
_metadata_store: PostgresMetadataStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载/创建索引，关闭时保存"""
    global _store, _rag, _metadata_store

    logger.info("=" * 50)
    logger.info("智能简历优化与模拟面试助手 V2 启动中... (LangGraph checkpointer)")
    logger.info("=" * 50)

    # 初始化 metadata store
    _metadata_store = PostgresMetadataStore(settings=settings)
    _metadata_store.setup()

    # 初始化向量存储
    _store = FAISSVectorStore(
        index_dir=settings.faiss_index_dir,
        metadata_store=_metadata_store,
    )
    loaded = _store.load()

    if not loaded and _store.has_legacy_metadata():
        if _metadata_store.is_available:
            loaded = _store.migrate_legacy_metadata()
        else:
            logger.warning("检测到旧 metadata.json，但 metadata_db_url / checkpoint_db_url 未配置，暂时跳过迁移")

    if loaded:
        logger.info("已加载已有索引: %d 条记录", _store.total)
    else:
        logger.info("未找到已有索引，将从空知识库开始")

    # 初始化 RAG 服务
    _rag = RAGService(_store)

    # 注入到各路由模块
    set_vector_store(_store)
    set_rag_service(_rag)

    # ---- 初始化 Agent 模块 ----
    await init_checkpointer(settings)
    await init_cache_store(settings)
    logger.info("初始化 Agent 模块 (checkpointer=%s)...", get_checkpointer_backend())
    logger.info("初始化 Expert Cache (backend=%s)...", get_cache_store_backend())
    logger.info(
        "初始化 Metadata Store (backend=%s)...",
        "postgres" if _metadata_store.is_available else "disabled",
    )

    # 检索服务（复用现有 vector_store）
    retrieval_service = RetrievalService(_store)

    # Web 搜索服务
    web_search_service = WebSearchService(api_key=settings.tavily_api_key)
    if web_search_service.is_available:
        logger.info("Web 搜索服务已启用 (Tavily)")
    else:
        logger.info("Web 搜索服务未启用 (缺少 TAVILY_API_KEY)")

    # 编译 Agent 主图（使用已初始化的 checkpointer）
    agent_graph = build_agent_graph(
        retrieval_service=retrieval_service,
        web_search_service=web_search_service,
        settings=settings,
    )

    # 获取 checkpointer 并注入到 API 层
    checkpointer = get_checkpointer()

    # 注入到 Agent API 模块
    set_agent_graph(agent_graph)
    set_checkpointer(checkpointer)

    logger.info("Agent 模块初始化完成 (thread 级状态持久化已启用)")
    logger.info("服务就绪: http://%s:%d", settings.host, settings.port)
    yield

    # 关闭时保存索引
    if _store and not _store.is_empty():
        _store.save()
        logger.info("索引已保存")
    await shutdown_cache_store()
    await shutdown_checkpointer()


# 创建 FastAPI 应用
app = FastAPI(
    title="智能简历优化与模拟面试助手",
    description="基于智谱 GLM-4V + FAISS 的多模态 RAG 问答系统",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS（开发环境允许全部来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(agent_router)
if settings.debug_mode:
    app.include_router(debug_router)

# 挂载静态文件（前端页面）
static_dir = "static"
import os
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """根路径，返回前端页面"""
    from fastapi.responses import FileResponse
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "智能简历优化与模拟面试助手 V2", "docs": "/docs"}


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "index_records": _store.total if _store else 0,
        "checkpointer_backend": get_checkpointer_backend(),
        "expert_cache_backend": get_cache_store_backend(),
        "metadata_store_backend": "postgres" if _metadata_store and _metadata_store.is_available else "disabled",
    }
