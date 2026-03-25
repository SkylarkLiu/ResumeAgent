"""
图片加载器 - 调用视觉理解将图片文本化
"""
from __future__ import annotations

import base64
from pathlib import Path

from app.core.logger import setup_logger
from app.schemas.file import FileMeta, FileType
from app.services.vision_service import understand_image

logger = setup_logger("image_loader")


class ImageLoader:
    """图片加载器 - 将图片通过视觉理解转为文本知识单元"""

    def load(self, file_path: str | Path) -> tuple[FileMeta, str]:
        """
        加载图片并生成文本化描述。

        Args:
            file_path: 图片文件路径
        Returns:
            (FileMeta, 文本化描述内容)
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info("图片视觉理解中: %s", p.name)

        # 读取为 base64
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        # 调用视觉理解
        text = understand_image(image_base64=b64)

        meta = FileMeta(
            filename=p.name,
            file_type=FileType.IMAGE,
            file_path=str(p.resolve()),
            file_size=p.stat().st_size,
            chunks=0,
        )

        logger.info("图片理解完成: %s -> %d 字符", p.name, len(text))
        return meta, text
