"""
文本分块器 - LangChain RecursiveCharacterTextSplitter 封装

新版 LangChain (1.x) 使用 langchain-text-splitters 独立包，
配合 langchain-core 的 Document 对象，实现文本→分块→Document 的标准流水线。
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings

_settings = get_settings()


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """获取文本分块器实例"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or _settings.chunk_size,
        chunk_overlap=chunk_overlap or _settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """将文本切分为字符串块列表"""
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def split_into_documents(
    text: str,
    source: str = "",
    page: int | None = None,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    将文本切分为 LangChain Document 列表。

    每个 Document 携带 source / page / 自定义 metadata，
    方便后续嵌入和检索时追溯来源。

    Args:
        text: 原始文本
        source: 来源文件名
        page: PDF 页码（如有）
        metadata: 额外元数据
        chunk_size: 块大小
        chunk_overlap: 块重叠
    Returns:
        Document 对象列表
    """
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    base_meta: dict = {"source": source}
    if page is not None:
        base_meta["page"] = page
    if metadata:
        base_meta.update(metadata)

    return [Document(page_content=chunk, metadata=base_meta.copy()) for chunk in chunks]


def split_texts_into_documents(
    text_chunks: list[dict],
) -> list[Document]:
    """
    批量将已分页的文本块转为 Document 列表。

    适用于 PDF 按页提取后的批量转换。

    Args:
        text_chunks: [{"page": 1, "text": "..."}, ...]
    Returns:
        Document 对象列表
    """
    documents = []
    for chunk in text_chunks:
        page_num = chunk.get("page")
        text = chunk.get("text", "")
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": chunk.get("source", ""),
                    "page": page_num,
                },
            )
        )
    return documents
