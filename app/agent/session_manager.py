"""
MVP 内存会话管理器（阶段 2 迁移到 LangGraph checkpointer/thread）

临时实现，仅用于阶段 1 快速验证多轮对话流程。
后续将迁移到 LangGraph checkpointer/thread 模式，
以 thread_id 为维度持久化 graph state。
"""
from __future__ import annotations

from app.core.logger import setup_logger

logger = setup_logger("session_manager")


class SessionManager:
    """基于内存的简单会话管理器（MVP）"""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
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
            self._sessions[session_id] = {"messages": [], "resume_data": None}

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
            self._sessions[session_id] = {"messages": [], "resume_data": None}
        self._sessions[session_id]["resume_data"] = data
        logger.info("session=%s 已保存 resume_data", session_id)

    def clear_session(self, session_id: str) -> None:
        """清空指定会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session=%s 已清除", session_id)

    def _trim_history(self, session_id: str) -> None:
        """裁剪历史消息到 max_history 条"""
        session = self._sessions.get(session_id)
        if not session:
            return
        messages = session["messages"]
        if len(messages) > self.max_history:
            session["messages"] = messages[-self.max_history:]
            logger.debug(
                "session=%s 历史已裁剪: %d -> %d",
                session_id,
                len(messages),
                self.max_history,
            )
