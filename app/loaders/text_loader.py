"""
文本加载器 - 处理 .txt / .md 文件
"""
from __future__ import annotations

from pathlib import Path

from app.core.logger import setup_logger
from app.schemas.file import FileMeta, FileType

logger = setup_logger("text_loader")


class TextLoader:
    """纯文本文件加载器"""

    def load(self, file_path: str | Path) -> tuple[FileMeta, str]:
        """
        加载文本文件。

        Args:
            file_path: 文件路径
        Returns:
            (FileMeta, 文本内容)
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        text = p.read_text(encoding="utf-8").strip()
        meta = FileMeta(
            filename=p.name,
            file_type=FileType.TEXT,
            file_path=str(p.resolve()),
            file_size=p.stat().st_size,
            chunks=0,  # 后续由调用方分块后更新
        )
        logger.info("文本加载: %s (%d 字符)", p.name, len(text))
        return meta, text
