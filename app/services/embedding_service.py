"""
Embedding 服务 - 智谱 embedding-3 文本向量化
"""
from __future__ import annotations

from zai import ZhipuAiClient

from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("embedding_service")

_settings = get_settings()


def get_embedding_client() -> ZhipuAiClient:
    return ZhipuAiClient(api_key=_settings.zhipuai_api_key)


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    批量文本嵌入。

    Args:
        texts: 文本列表（最多支持单次 64 条，超出会自动分批）
        model: 模型名，默认从配置读取
    Returns:
        嵌入向量列表，与输入一一对应
    """
    client = get_embedding_client()
    model = model or _settings.embedding_model_name

    all_embeddings: list[list[float]] = []

    # 智谱 API 单次上限，分批处理
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug("Embedding 批次 %d-%d/%d", i, i + len(batch), len(texts))

        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    logger.debug("Embedding 完成: 共 %d 条", len(all_embeddings))
    return all_embeddings


def embed_single(text: str, model: str | None = None) -> list[float]:
    """单条文本嵌入"""
    results = embed_texts([text], model=model)
    return results[0] if results else []
