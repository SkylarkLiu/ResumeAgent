"""
MVP 内存会话管理器（阶段 2 迁移到 LangGraph checkpointer/thread）

临时实现，仅用于阶段 1 快速验证多轮对话流程。
后续将迁移到 LangGraph checkpointer/thread 模式，
以 thread_id 为维度持久化 graph state。

分层裁剪改造：_trim_history 改为分层裁剪策略，
早期消息压缩为摘要文本，最近消息完整保留。
"""
from __future__ import annotations

from app.core.logger import setup_logger
from app.agent.utils.history_utils import build_layered_messages, estimate_tokens

logger = setup_logger("session_manager")


class SessionManager:
    """基于内存的简单会话管理器（MVP），支持分层裁剪"""

    def __init__(
        self,
        max_history: int = 20,
        recent_count: int = 10,
        summary_max_chars: int = 120,
        summary_token_budget: int = 800,
    ):
        self.max_history = max_history  # 兼容旧逻辑的 fallback
        self.recent_count = recent_count
        self.summary_max_chars = summary_max_chars
        self.summary_token_budget = summary_token_budget
        self._sessions: dict[str, dict] = {}

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话历史消息列表"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return list(session.get("messages", []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """向会话追加一条消息"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "resume_data": None,
                "conversation_summary": "",
            }

        self._sessions[session_id]["messages"].append(
            {"role": role, "content": content}
        )
        self._trim_history(session_id)

    def get_session(self, session_id: str) -> dict | None:
        """获取完整会话快照"""
        return self._sessions.get(session_id)

    def get_resume_data(self, session_id: str) -> dict | None:
        """获取跨轮次的简历提取结果"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.get("resume_data")

    def set_resume_data(self, session_id: str, data: dict) -> None:
        """保存简历提取结果（跨轮次保持）"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "resume_data": None,
                "conversation_summary": "",
            }
        self._sessions[session_id]["resume_data"] = data
        logger.info("session=%s 已保存 resume_data", session_id)

    def get_conversation_summary(self, session_id: str) -> str:
        """获取会话的对话摘要"""
        session = self._sessions.get(session_id)
        if not session:
            return ""
        return session.get("conversation_summary", "")

    def clear_session(self, session_id: str) -> None:
        """清空指定会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session=%s 已清除", session_id)

    def _trim_history(self, session_id: str) -> None:
        """
        分层裁剪历史消息：
        - 保留最近 recent_count 条完整消息
        - 更早的消息压缩为 conversation_summary
        - 物理删除已压缩的早期消息
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        messages = session["messages"]
        if len(messages) <= self.recent_count:
            return  # 消息不多，无需裁剪

        # 将 MVP dict 消息转为类似 BaseMessage 的对象以复用 build_layered_messages
        from langchain_core.messages import AIMessage as LC_AIMessage, HumanMessage as LC_HumanMessage

        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(LC_HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(LC_AIMessage(content=msg["content"]))

        _, new_summary = build_layered_messages(
            lc_messages,
            recent_count=self.recent_count,
            summary_max_chars=self.summary_max_chars,
            token_budget=self.summary_token_budget,
        )

        # 合并旧摘要
        old_summary = session.get("conversation_summary", "")
        if old_summary and new_summary:
            # 旧摘要已在之前被压缩过，直接用新摘要覆盖旧内容中的重叠部分
            # 策略：新摘要 + 旧摘要的非重叠部分（截断防膨胀）
            trimmed_old = old_summary
            if len(trimmed_old) > 400:
                trimmed_old = "…" + trimmed_old[-400:]
            merged_summary = f"{new_summary}\n[更早]{trimmed_old}"
        elif old_summary and not new_summary:
            merged_summary = old_summary
        else:
            merged_summary = new_summary

        # 物理删除早期消息，只保留最近的
        session["messages"] = messages[-self.recent_count:]
        session["conversation_summary"] = merged_summary

        summary_tokens = estimate_tokens(merged_summary) if merged_summary else 0
        logger.debug(
            "session=%s 分层裁剪: %d→%d条, 摘要=%d token",
            session_id,
            len(messages),
            len(session["messages"]),
            summary_tokens,
        )
