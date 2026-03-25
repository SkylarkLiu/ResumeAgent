"""
Chat 相关数据模型
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """文本问答请求"""
    question: str = Field(..., min_length=1, description="用户问题")


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str = Field(..., description="模型回答")
    sources: list[SourceItem] = Field(default_factory=list, description="来源引用")


class ImageChatRequest(BaseModel):
    """图片即时问答请求"""
    question: str = Field(default="请描述这张图片的内容", description="关于图片的问题")
    image_base64: str = Field(..., description="图片 base64 编码（不含前缀）")


class SourceItem(BaseModel):
    """检索来源片段"""
    content: str = Field(..., description="命中的文本片段")
    source: str = Field(default="", description="来源文件名")
    page: int | None = Field(default=None, description="PDF 页码（如有）")
    score: float = Field(default=0.0, description="相似度分数")
