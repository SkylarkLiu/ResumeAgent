"""
RAG 编排服务 - 检索 + Augment + LLM 回答 + 来源引用
"""
from __future__ import annotations

from app.core.logger import setup_logger
from app.repositories.vector_store import FAISSVectorStore
from app.services.llm_service import chat_completion
from app.services.retrieval_service import RetrievalService

logger = setup_logger("rag_service")

# RAG 系统 prompt
RAG_SYSTEM_PROMPT = """你是一个专业的文档问答助手。请基于用户提供的参考内容来回答问题。

规则：
1. 只根据参考内容回答，不要编造信息
2. 如果参考内容中没有相关信息，请明确告知用户
3. 回答时引用来源，标明出处
4. 回答要简洁清晰，有条理
5. 如果涉及数据、数字、日期等关键信息，请确保准确"""

RAG_USER_TEMPLATE = """参考内容：

{context}

---

用户问题：{question}

请基于上述参考内容回答用户问题。如果参考内容不足以回答，请说明。回答格式要求清晰，并在关键信息处标注来源。"""


class RAGService:
    """RAG 问答编排服务"""

    def __init__(self, vector_store: FAISSVectorStore):
        self.retrieval = RetrievalService(vector_store)
        self.vector_store = vector_store

    def answer(self, question: str, top_k: int | None = None) -> dict:
        """
        RAG 问答全链路：检索 → 拼装上下文 → LLM 回答 → 返回来源引用。

        Args:
            question: 用户问题
            top_k: 检索条数
        Returns:
            {"answer": str, "sources": [...]}
        """
        # 1. 检索
        sources = self.retrieval.retrieve(question, top_k=top_k)

        if not sources:
            logger.info("知识库为空或无相关结果，使用通用回答")
            return {
                "answer": "当前知识库为空或未找到相关内容。请先上传文档到知识库，然后再提问。",
                "sources": [],
            }

        # 2. 拼装上下文
        context_parts = []
        for i, src in enumerate(sources):
            source_label = f"【来源{i+1}】"
            if src.get("source"):
                source_label += f" {src['source']}"
            if src.get("page"):
                source_label += f" 第{src['page']}页"
            context_parts.append(f"{source_label}\n{src['content']}")

        context = "\n\n".join(context_parts)
        logger.info("RAG 上下文: %d 条来源, 共 %d 字符", len(sources), len(context))

        # 3. LLM 回答
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": RAG_USER_TEMPLATE.format(context=context, question=question)},
        ]

        answer = chat_completion(messages)
        logger.info("RAG 回答完成: %d 字符", len(answer))

        # 4. 返回结果（含来源引用）
        source_items = [
            {
                "content": src["content"],
                "source": src.get("source", ""),
                "page": src.get("page"),
                "score": src.get("score", 0.0),
            }
            for src in sources
        ]

        return {"answer": answer, "sources": source_items}
