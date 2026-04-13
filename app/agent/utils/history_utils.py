"""
分层裁剪工具 - token 估算 + 早期消息摘要 + 分层历史构建

方案 A 核心实现：
- System prompt（始终保留）
- 早期消息压缩为摘要
- 最近 N 条完整消息原样保留

分层裁剪相比纯条数裁剪的优势：
1. 简单对话（短消息）可以保留更多轮次上下文
2. 分析场景（长消息）自动压缩早期历史，避免 token 浪费
3. 不丢失对话连续性，早期上下文以摘要形式保留
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


# ---- Token 估算 ----

# 中文平均约 1.5 字符/token，英文约 4 字符/token，混合取中间值
_CHARS_PER_TOKEN: float = 2.0


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    使用简单的字符/token 比率估算，不引入外部分词器依赖。
    误差约 ±30%，但对于裁剪决策足够可靠——
    因为目标是控制 token 预算数量级，而非精确到个位。

    Args:
        text: 待估算文本
    Returns:
        估算的 token 数
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_message_tokens(msg: BaseMessage) -> int:
    """
    估算单条消息的 token 数量。

    包含内容文本 + 角色开销（约 4 token/条）。
    """
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        content = str(content) if content else ""
    return estimate_tokens(content) + 4


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """估算消息列表的总 token 数。"""
    return sum(estimate_message_tokens(msg) for msg in messages)


# ---- 早期消息摘要 ----

def summarize_early_messages(
    messages: list[BaseMessage],
    max_chars_per_msg: int = 120,
) -> str:
    """
    将早期消息压缩为一段文本摘要。

    策略：对每条消息截取前 max_chars_per_msg 个字符，拼接为对话摘要。
    不调用 LLM 生成摘要（避免额外延迟和 token 开销），
    而是采用规则化压缩：保留角色标识 + 内容截断。

    Args:
        messages: 需要压缩的消息列表
        max_chars_per_msg: 每条消息保留的最大字符数
    Returns:
        压缩后的对话摘要文本
    """
    if not messages:
        return ""

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            # 跳过路由决策等内部消息
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.startswith("[路由决策]"):
                continue
            role = "助手"
        else:
            continue

        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content) if content else ""
        content = content.strip()
        if not content:
            continue

        truncated = content[:max_chars_per_msg]
        if len(content) > max_chars_per_msg:
            truncated += "…"
        lines.append(f"{role}: {truncated}")

    return "\n".join(lines)


# ---- 分层历史构建 ----

def build_layered_messages(
    messages: list[BaseMessage],
    *,
    recent_count: int = 10,
    summary_max_chars: int = 600,
    token_budget: int = 0,
) -> tuple[list[dict[str, str]], str]:
    """
    分层裁剪核心函数：将消息列表分为「摘要层」+「完整层」。

    算法：
    1. 最近的 recent_count 条消息原样保留（完整层）
    2. 更早的消息压缩为文本摘要（摘要层）
    3. 如果设置了 token_budget，在摘要层内做进一步裁剪，
       从最早的消息开始丢弃，直到摘要 token 数在预算内

    Args:
        messages: 完整消息列表
        recent_count: 最近保留的完整消息条数
        summary_max_chars: 摘要层每条消息的最大字符数
        token_budget: 摘要层的 token 预算（0 表示不限制）
    Returns:
        (complete_messages, summary_text)
        - complete_messages: 最近 N 条的 OpenAI 格式消息列表
        - summary_text: 早期消息的压缩摘要文本
    """
    if not messages:
        return [], ""

    # 分割点
    split_idx = max(0, len(messages) - recent_count)
    early_messages = messages[:split_idx]
    recent_messages = messages[split_idx:]

    # 过滤路由决策消息后构建完整层
    complete_msgs: list[dict[str, str]] = []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            complete_msgs.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if not (isinstance(msg.content, str) and msg.content.startswith("[路由决策]")):
                complete_msgs.append({"role": "assistant", "content": msg.content})

    # 构建摘要层
    summary_text = summarize_early_messages(early_messages, max_chars_per_msg=summary_max_chars)

    # 如果设置了 token 预算，裁剪摘要
    if token_budget > 0 and summary_text:
        summary_tokens = estimate_tokens(summary_text)
        if summary_tokens > token_budget:
            # 从头部裁剪，保留摘要尾部（更近的上下文）
            # 简单策略：按字符比例截断
            ratio = token_budget / summary_tokens
            keep_chars = int(len(summary_text) * ratio * 0.9)  # 留 10% 余量
            summary_text = "…" + summary_text[-keep_chars:]

    return complete_msgs, summary_text
