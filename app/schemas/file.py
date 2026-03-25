"""
File 相关数据模型
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FileType(str, Enum):
    """支持的文件类型"""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"


class FileMeta(BaseModel):
    """文件元信息"""
    filename: str
    file_type: FileType
    file_path: str | None = None
    file_size: int = 0
    page_count: int | None = None
    chunks: int = 0
