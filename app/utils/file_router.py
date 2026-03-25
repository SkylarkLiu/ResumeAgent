"""
文件路由 - 根据扩展名分类文件类型
"""
from __future__ import annotations

from pathlib import Path

from app.core.logger import setup_logger
from app.schemas.file import FileType

logger = setup_logger("file_router")

# 扩展名 → 文件类型映射
EXT_TYPE_MAP: dict[str, FileType] = {
    ".txt": FileType.TEXT,
    ".md": FileType.TEXT,
    ".text": FileType.TEXT,
    ".pdf": FileType.PDF,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".gif": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".webp": FileType.IMAGE,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def classify_file(file_name: str | Path) -> FileType:
    """
    根据文件扩展名判断文件类型。
    未知类型默认归为 text。
    """
    ext = Path(file_name).suffix.lower()
    file_type = EXT_TYPE_MAP.get(ext, FileType.TEXT)
    logger.debug("文件分类: %s -> %s", file_name, file_type.value)
    return file_type


def is_image_file(file_name: str | Path) -> bool:
    return Path(file_name).suffix.lower() in IMAGE_EXTS


def get_extension(file_name: str | Path) -> str:
    return Path(file_name).suffix.lower()
