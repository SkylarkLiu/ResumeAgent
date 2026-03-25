"""
PDF 服务 - PyMuPDF 文本提取与页面渲染
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF

from app.core.logger import setup_logger

logger = setup_logger("pdf_service")


class PDFService:
    """PDF 处理服务，提供文本提取和页面渲染能力"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.doc: fitz.Document = fitz.open(str(self.file_path))
        self.page_count: int = len(self.doc)

    def close(self):
        self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def is_scanned(self, text_threshold: int = 50) -> bool:
        """
        判断 PDF 是否为扫描件（无可提取文本）。
        采样前几页，如果平均每页提取字符数低于阈值则判定为扫描件。
        """
        total_chars = 0
        sample_pages = min(5, self.page_count)

        for i in range(sample_pages):
            page = self.doc[i]
            text = page.get_text().strip()
            total_chars += len(text)

        avg_chars = total_chars / max(sample_pages, 1)
        is_scan = avg_chars < text_threshold
        logger.info(
            "PDF 扫描件检测: %s (采样 %d 页, 平均 %d 字符/页, 阈值 %d)",
            "扫描件" if is_scan else "可提取文本",
            sample_pages,
            avg_chars,
            text_threshold,
        )
        return is_scan

    def extract_text_by_page(self) -> list[dict]:
        """
        按页提取文本（适用于可提取文本的 PDF）。
        返回: [{"page": 1, "text": "...", "char_count": 123}, ...]
        """
        pages = []
        for i in range(self.page_count):
            page = self.doc[i]
            text = page.get_text().strip()
            if text:  # 跳过空白页
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
        logger.info("PDF 文本提取: %d/%d 页有内容", len(pages), self.page_count)
        return pages

    def render_page_as_image(self, page_num: int, dpi: int = 150) -> bytes:
        """
        将指定页渲染为 PNG 图片字节流。

        Args:
            page_num: 页码（从 1 开始）
            dpi: 渲染分辨率
        Returns:
            PNG 图片的字节数据
        """
        if page_num < 1 or page_num > self.page_count:
            raise ValueError(f"页码超出范围: {page_num}, 共 {self.page_count} 页")

        page = self.doc[page_num - 1]
        # 缩放因子：72 是 PDF 默认 DPI
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        # 输出为 PNG
        img_data = pix.tobytes("png")
        logger.debug("页面 %d 渲染完成: %dx%d", page_num, pix.width, pix.height)
        return img_data

    def extract_all(self) -> dict:
        """
        智能提取 PDF 全部内容。
        - 可提取文本型：按页提取文本
        - 扫描件：返回每页的图片字节流列表

        返回:
            {
                "is_scanned": bool,
                "pages": [{"page": 1, "text": "..."}] 或
                         [{"page": 1, "image_bytes": b"...", "image_base64": "..."}],
                "page_count": int,
            }
        """
        is_scan = self.is_scanned()

        if is_scan:
            pages = []
            for i in range(1, self.page_count + 1):
                img_bytes = self.render_page_as_image(i)
                import base64
                pages.append({
                    "page": i,
                    "image_bytes": img_bytes,
                    "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
                })
        else:
            pages = self.extract_text_by_page()

        return {
            "is_scanned": is_scan,
            "pages": pages,
            "page_count": self.page_count,
        }
