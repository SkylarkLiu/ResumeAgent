"""
PDF 加载器 - 双模处理：可提取文本 / 扫描件走视觉理解
"""
from __future__ import annotations

from pathlib import Path

from app.core.logger import setup_logger
from app.schemas.file import FileMeta, FileType
from app.services.pdf_service import PDFService
from app.services.vision_service import vision_chat

logger = setup_logger("pdf_loader")

# 扫描件每页视觉理解的 prompt
PAGE_VISION_PROMPT = """请仔细阅读这一页的内容，按以下结构输出：

## 页面摘要
（简要概括本页的核心内容）

## 文字内容
（完整提取本页中可读的文字，保持原文结构和层次）

## 表格/图表
（如果有表格或图表，提取关键数据；没有则写"无"）

## 关键词
（3-5 个检索关键词，逗号分隔）"""


class PDFLoader:
    """PDF 加载器 - 自动判断文本型/扫描件，分别处理"""

    def load(self, file_path: str | Path) -> tuple[FileMeta, str]:
        """
        加载 PDF 并生成文本化内容。

        - 可提取文本：按页提取文本，合并
        - 扫描件：每页渲染为图片，走视觉理解生成描述

        Args:
            file_path: PDF 文件路径
        Returns:
            (FileMeta, 合并后的文本内容)
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        meta = FileMeta(
            filename=p.name,
            file_type=FileType.PDF,
            file_path=str(p.resolve()),
            file_size=p.stat().st_size,
        )

        with PDFService(p) as pdf:
            meta.page_count = pdf.page_count
            result = pdf.extract_all()

            if result["is_scanned"]:
                # 扫描件：逐页走视觉理解
                logger.info("扫描件 PDF: 逐页视觉理解, 共 %d 页", result["page_count"])
                page_texts = []
                for page_info in result["pages"]:
                    page_num = page_info["page"]
                    page_b64 = page_info["image_base64"]
                    logger.info("视觉理解第 %d/%d 页...", page_num, result["page_count"])
                    page_text = vision_chat(
                        prompt=PAGE_VISION_PROMPT,
                        image_base64=page_b64,
                    )
                    # 添加页码标识
                    page_texts.append(f"--- 第 {page_num} 页 ---\n{page_text}")
                text = "\n\n".join(page_texts)
            else:
                # 可提取文本型：直接合并
                page_texts = []
                for page_info in result["pages"]:
                    page_num = page_info["page"]
                    page_text = page_info["text"]
                    page_texts.append(f"--- 第 {page_num} 页 ---\n{page_text}")
                text = "\n\n".join(page_texts)

        logger.info("PDF 加载完成: %s -> %d 页, %d 字符", p.name, meta.page_count, len(text))
        return meta, text
