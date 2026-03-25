"""
聊天路由 - POST /chat + POST /chat/image
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.logger import setup_logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ImageChatRequest,
    SourceItem,
)
from app.services.vision_service import vision_chat

logger = setup_logger("api.chat")
router = APIRouter(prefix="/chat", tags=["chat"])

# RAG 服务由 main.py 注入
rag_service = None


def set_rag_service(service):
    """由 main.py 在启动时注入 RAG 服务实例"""
    global rag_service
    rag_service = service


@router.post("")
async def chat(req: ChatRequest):
    """知识库问答"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未初始化")

    logger.info("知识库问答: '%s'", req.question[:80])

    try:
        result = rag_service.answer(req.question)
        sources = [
            SourceItem(
                content=s.get("content", ""),
                source=s.get("source", ""),
                page=s.get("page"),
                score=s.get("score", 0.0),
            )
            for s in result.get("sources", [])
        ]
        return ChatResponse(answer=result["answer"], sources=sources).model_dump()
    except Exception as e:
        logger.error("问答失败: %s - %s", req.question[:50], e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@router.post("/image")
async def chat_image(req: ImageChatRequest):
    """
    图片即时问答。
    不走知识库，直接调用视觉模型回答。
    """
    logger.info("图片问答: '%s' (base64 长度: %d)", req.question[:50], len(req.image_base64))

    try:
        answer = vision_chat(
            prompt=req.question,
            image_base64=req.image_base64,
        )
        return ChatResponse(answer=answer, sources=[]).model_dump()
    except Exception as e:
        logger.error("图片问答失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"图片问答失败: {str(e)}")
