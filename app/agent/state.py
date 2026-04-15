"""
Agent 状态定义 - TypedDict + 枚举 + 结构化路由决策
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class RouteType(str, Enum):
    """数据来源路由"""
    RETRIEVE = "retrieve"
    WEB = "web"
    DIRECT = "direct"


class TaskType(str, Enum):
    """任务类型路由"""
    QA = "qa"
    RESUME_ANALYSIS = "resume_analysis"
    JD_ANALYSIS = "jd_analysis"
    SUMMARY_ASSESSMENT = "summary_assessment"
    INTERVIEW_SIMULATION = "interview_simulation"
    INTERVIEW_FOLLOWUP = "interview_followup"
    JD_FOLLOWUP = "jd_followup"
    RESUME_FOLLOWUP = "resume_followup"
    MATCH_FOLLOWUP = "match_followup"
    REACT_FALLBACK = "react_fallback"


class RouteDecision(BaseModel):
    """Router 结构化输出模型"""
    reasoning: str = Field(description="简要判断理由")
    route_type: RouteType = Field(description="数据来源路径")
    task_type: TaskType = Field(description="任务类型")


class AgentState(TypedDict, total=False):
    """
    Agent 全局状态，通过 StateGraph 共享。

    边界原则：
    - context_sources 是"结构化数据层"，各节点读写
    - working_context 是"文本化展示层"，仅在 generate 节点消费
    """
    # ---- 消息历史（LangGraph 内置 add_messages reducer）----
    # add_messages 语义：追加新消息 + 更新已有消息（按 message.id 匹配）
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- 结构化中间产物（各节点读写，不拍扁）----
    # 每个元素: {"content": str, "source": str, "score": float, "type": "kb"|"web"|"resume"}
    context_sources: list[dict]

    # ---- 文本化上下文（仅 generate 节点从 context_sources 拼装后消费）----
    working_context: str

    # ---- 路由决策（存储字符串值，避免 msgpack 序列化问题）----
    route_type: str  # "retrieve" | "web" | "direct"
    task_type: str   # "qa" | "resume_analysis" | "jd_analysis" | "summary_assessment" | "jd_followup" | "resume_followup" | "match_followup" | "react_fallback"
    question_signature: str
    planning_reason: str
    response_mode: str  # "full_report" | "followup_brief" | "match_brief" | "direct_answer"
    active_agent: str | None
    execution_plan: list[str]
    current_step: int
    max_steps: int
    final_response_ready: bool
    agent_outputs: dict[str, dict]
    expert_cache: dict[str, dict]
    tool_cache: dict[str, dict]
    tool_trace: list[dict]
    react_iterations: int
    react_handoff_agent: str | None

    # ---- 会话与跨轮次数据 ----
    session_id: str

    # ---- 简历分析子图 ----
    # 输入阶段: {"raw_text": str} 或 {"file_path": str} 或 {"file_base64": str}
    # 提取后: {"name", "skills", "experience", "education", "projects", "summary", ...}
    resume_data: dict | None

    # ---- JD 分析子图 ----
    # 输入阶段: {"raw_text": str}（JD 原文）
    # 提取后: {"position", "company", "skills_must", "tech_stack", "summary", ...}
    jd_data: dict | None
    interview_data: dict | None
    summary_data: dict | None

    # ---- 分层裁剪：对话摘要 ----
    # 早期消息的压缩摘要文本（由 generate 节点维护，持久化到 checkpointer）
    conversation_summary: str

    # ---- 检索降级标记 ----
    # 当 KB 检索结果质量不足时，标记为 True，触发降级到 web search
    retrieval_fallback: bool

    # ---- 最终输出 ----
    final_answer: str
