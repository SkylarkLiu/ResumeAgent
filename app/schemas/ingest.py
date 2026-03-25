"""
Ingest 相关数据模型
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """文件导入响应"""
    message: str = Field(..., description="处理结果描述")
    file_type: str = Field(..., description="文件类型：text / image / pdf")
    chunks: int = Field(default=0, description="生成的文本块数量")
    pages: int | None = Field(default=None, description="PDF 页数（如有）")
