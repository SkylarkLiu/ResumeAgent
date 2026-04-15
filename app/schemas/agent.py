"""
Agent 相关数据模型
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="", description="会话 ID（前端生成并持久化）")


class AgentSourceItem(BaseModel):
    """Agent 回答的来源"""
    content: str = Field(default="", description="来源文本")
    source: str = Field(default="", description="来源 URL 或文件名")
    score: float = Field(default=0.0, description="相似度分数")
    type: str = Field(default="kb", description="来源类型: kb / web")


class AgentChatResponse(BaseModel):
    """Agent 对话响应"""
    answer: str = Field(..., description="模型回答")
    sources: list[AgentSourceItem] = Field(default_factory=list, description="来源列表")
    session_id: str = Field(..., description="会话 ID")
    route_type: str = Field(default="direct", description="实际路由路径")
    task_type: str = Field(default="qa", description="任务类型")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    message_count: int


class SessionListItem(BaseModel):
    """会话列表条目"""
    session_id: str
    title: str = Field(default="", description="会话标题")
    message_count: int = 0
    last_message_preview: str = Field(default="", description="最后一条消息预览")
    has_resume_data: bool = False
    has_jd_data: bool = False
    has_summary_data: bool = False
    task_type: str = Field(default="", description="最近一次任务类型")
    created_at: str = Field(default="", description="创建时间")
    updated_at: str = Field(default="", description="最近更新时间")


class SessionMessageItem(BaseModel):
    """会话消息条目"""
    role: str = Field(description="角色: user / assistant")
    content: str = Field(default="", description="消息内容")
    task_type: str = Field(default="", description="任务类型（仅 assistant 消息有）")
    route_type: str = Field(default="", description="路由类型（仅 assistant 消息有）")
    summary_data: dict = Field(default_factory=dict, description="综合评估结构化结果（仅总结消息）")


class SessionMessagesResponse(BaseModel):
    """会话消息列表响应"""
    session_id: str
    messages: list[SessionMessageItem] = Field(default_factory=list)


class ResumeAnalysisRequest(BaseModel):
    """简历分析请求（支持文本粘贴）"""
    resume_text: str = Field(default="", description="简历文本内容（直接粘贴）")
    question: str = Field(default="请对我的简历进行全面分析评估", description="用户问题或分析要求")
    session_id: str = Field(default="", description="会话 ID")
    target_position: str = Field(default="", description="目标岗位（可选，帮助更精准匹配）")


class ResumeAnalysisResponse(BaseModel):
    """简历分析响应"""
    answer: str = Field(..., description="分析报告（Markdown）")
    resume_data: dict = Field(default_factory=dict, description="提取的结构化简历信息")
    sources: list[AgentSourceItem] = Field(default_factory=list, description="参考的 JD 来源")
    session_id: str = Field(..., description="会话 ID")


class JDAnalysisRequest(BaseModel):
    """JD 分析请求（支持文本粘贴）"""
    jd_text: str = Field(default="", description="JD 岗位描述文本内容（直接粘贴）")
    question: str = Field(default="请分析该岗位的核心要求并给出简历写作建议", description="用户问题或分析要求")
    session_id: str = Field(default="", description="会话 ID")


class JDAnalysisResponse(BaseModel):
    """JD 分析响应"""
    answer: str = Field(..., description="分析报告（Markdown）")
    jd_data: dict = Field(default_factory=dict, description="提取的结构化 JD 信息")
    session_id: str = Field(..., description="会话 ID")
