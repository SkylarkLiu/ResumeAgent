"""
文件导入路由 - POST /ingest/file / DELETE /ingest/file/{filename}
处理流程：上传文件 → 分类 → 加载 → 分块 → embed → 入库
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.schemas.file import FileType
from app.services.embedding_service import embed_texts
from app.utils.file_router import classify_file
from app.utils.splitter import split_into_documents

logger = setup_logger("api.ingest")
settings = get_settings()
router = APIRouter(prefix="/ingest", tags=["ingest"])

# 运行时状态：已上传的文件清单（简易实现，V1 用内存）
_ingested_files: list[dict] = []

# 向量存储和 RAG 服务在 main.py 初始化后注入
vector_store = None


def set_vector_store(store):
    """由 main.py 在启动时注入全局向量存储实例"""
    global vector_store
    vector_store = store


@router.post("/file")
async def ingest_file(file: UploadFile):
    """
    上传知识文件到知识库。
    支持 .txt / .md / .png / .jpg / .jpeg / .pdf
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    # 1. 校验文件类型
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，允许: {settings.allowed_extensions}",
        )

    file_type = classify_file(filename)
    logger.info("收到文件: %s (类型: %s)", filename, file_type.value)

    # 2. 保存到 raw 目录
    raw_dir = settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / f"{uuid.uuid4().hex[:8]}_{filename}"

    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大: {len(content)} bytes，上限 {settings.max_upload_size_mb}MB",
        )

    save_path.write_bytes(content)
    logger.info("文件已保存: %s", save_path)

    # 3. 根据类型加载文件
    try:
        if file_type == FileType.TEXT:
            from app.loaders.text_loader import TextLoader
            meta, text = TextLoader().load(save_path)

        elif file_type == FileType.IMAGE:
            from app.loaders.image_loader import ImageLoader
            meta, text = ImageLoader().load(save_path)

        elif file_type == FileType.PDF:
            from app.loaders.pdf_loader import PDFLoader
            meta, text = PDFLoader().load(save_path)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_type}")
    except Exception as e:
        logger.error("文件加载失败: %s - %s", filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

    # 4. 分块 → Document
    page_count = meta.page_count
    docs = split_into_documents(
        text=text,
        source=filename,
        page=page_count if file_type == FileType.PDF else None,
    )
    chunk_count = len(docs)
    logger.info("分块完成: %s -> %d 个 Document", filename, chunk_count)

    # 5. Embedding
    if chunk_count > 0:
        try:
            texts = [doc.page_content for doc in docs]
            embeddings = embed_texts(texts)

            # 6. 入库
            from app.repositories.metadata_store import ChunkPayload
            from app.repositories.vector_store import VectorRecord

            if vector_store.metadata_store is None:
                raise RuntimeError("metadata store 未初始化")

            chunk_payloads = [
                ChunkPayload(
                    content=doc.page_content,
                    source=filename,
                    page=doc.metadata.get("page"),
                    chunk_index=i,
                    metadata=doc.metadata,
                )
                for i, doc in enumerate(docs)
            ]
            chunk_ids = vector_store.metadata_store.upsert_document_chunks(
                source_name=filename,
                chunks=chunk_payloads,
            )
            records = [
                VectorRecord(
                    id=str(chunk_id),
                    embedding=emb,
                    content=doc.page_content,
                    source=filename,
                    page=doc.metadata.get("page"),
                    metadata=doc.metadata,
                )
                for chunk_id, doc, emb in zip(chunk_ids, docs, embeddings)
            ]

            vector_store.add_records(records)
            vector_store.save()
            logger.info("入库完成: %d 条 -> PostgreSQL + FAISS", len(records))
        except Exception as e:
            if vector_store is not None and vector_store.metadata_store is not None:
                try:
                    vector_store.metadata_store.delete_by_source(filename)
                except Exception:
                    logger.warning("回滚 metadata 失败: %s", filename, exc_info=True)
            logger.error("Embedding/入库失败: %s - %s", filename, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"向量化入库失败: {str(e)}")

    # 7. 记录到文件清单
    _ingested_files.append({
        "name": filename,
        "type": file_type.value,
        "chunks": chunk_count,
        "pages": page_count,
        "size": len(content),
    })

    from app.schemas.ingest import IngestResponse
    return IngestResponse(
        message=f"文件导入成功: {filename}",
        file_type=file_type.value,
        chunks=chunk_count,
        pages=page_count,
    ).model_dump()


@router.get("/files")
async def list_files():
    """返回已上传的文件列表"""
    return {"files": _ingested_files}


@router.get("/sources")
async def list_sources(source_type: str | None = None, category: str | None = None):
    """
    返回知识库向量存储中所有不重复的来源文件名。
    用于查看实际入库了哪些文件，可用于删除时参考。
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    if vector_store.metadata_store is not None:
        sources = vector_store.metadata_store.list_sources(
            source_type=source_type,
            category=category,
        )
    else:
        sources = vector_store.get_sources()
    return {"sources": sources}


@router.get("/documents")
async def list_documents(source_type: str | None = None, category: str | None = None):
    """返回 PostgreSQL metadata 中的逻辑文档列表。"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")
    if vector_store.metadata_store is None:
        raise HTTPException(status_code=503, detail="metadata store 未初始化")

    documents = vector_store.metadata_store.list_documents(
        source_type=source_type,
        category=category,
    )
    return {"documents": documents}


@router.post("/compact")
async def compact_index():
    """
    手动压缩 FAISS 索引，清除已在 metadata 中删除的失效 row。
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    try:
        result = vector_store.compact()
    except Exception as e:
        logger.error("FAISS compact 失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"FAISS compact 失败: {str(e)}")

    return {
        "message": "FAISS compact 完成",
        **result,
    }


@router.delete("/file/{filename}")
async def delete_file(filename: str):
    """
    从知识库中删除指定来源的所有向量记录。

    Args:
        filename: 要删除的来源文件名（与上传时的原始文件名匹配）

    Returns:
        删除结果信息
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    try:
        deleted_count = vector_store.delete_by_source(filename)
    except Exception as e:
        logger.error("删除文件失败: %s - %s", filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

    if deleted_count == 0:
        return JSONResponse(
            status_code=404,
            content={
                "message": f"未找到来源为 '{filename}' 的记录",
                "deleted": 0,
            },
        )

    # 同步清理内存中的文件清单（用 in-place 修改，不替换引用）
    global _ingested_files
    _ingested_files = [f for f in _ingested_files if f.get("name") != filename]

    logger.info("删除知识库文件完成: %s, 删除 %d 条记录", filename, deleted_count)
    return {
        "message": f"已删除来源 '{filename}' 的所有记录",
        "deleted": deleted_count,
    }
