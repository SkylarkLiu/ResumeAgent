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
from app.core.config import get_settings
from app.core.logger import setup_logger
from app.repositories.vector_store import FAISSVectorStore
from app.services.rag_service import RAGService

logger = setup_logger("main")
settings = get_settings()

# 全局向量存储和 RAG 服务
_store: FAISSVectorStore | None = None
_rag: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载/创建索引，关闭时保存"""
    global _store, _rag

    logger.info("=" * 50)
    logger.info("多模态文档问答助手 V1 启动中...")
    logger.info("=" * 50)

    # 初始化向量存储
    _store = FAISSVectorStore(index_dir=settings.faiss_index_dir)
    loaded = _store.load()

    if loaded:
        logger.info("已加载已有索引: %d 条记录", _store.total)
    else:
        logger.info("未找到已有索引，将从空知识库开始")

    # 初始化 RAG 服务
    _rag = RAGService(_store)

    # 注入到各路由模块
    set_vector_store(_store)
    set_rag_service(_rag)

    logger.info("服务就绪: http://%s:%d", settings.host, settings.port)
    yield

    # 关闭时保存索引
    if _store and not _store.is_empty():
        _store.save()
        logger.info("索引已保存")


# 创建 FastAPI 应用
app = FastAPI(
    title="多模态文档问答助手",
    description="基于智谱 GLM-4V + FAISS 的多模态 RAG 问答系统",
    version="1.0.0",
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
    return {"message": "多模态文档问答助手 V1", "docs": "/docs"}


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "index_records": _store.total if _store else 0,
    }
