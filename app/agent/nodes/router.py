"""
路由节点 - 结构化输出 + 枚举校验 + fallback + 重试
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.prompts import ROUTER_SYSTEM_PROMPT
from app.agent.state import AgentState, RouteDecision, RouteType, TaskType
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.router")

# 最大重试次数
_MAX_RETRIES = 1


def _parse_json_from_response(text: str) -> dict | None:
    """
    从 LLM 响应中提取 JSON，支持多种格式：
    1. 纯 JSON
    2. Markdown 代码块包裹的 JSON
    3. 嵌套在其他文本中的 JSON
    4. 包含中文引号的 JSON（兼容处理）
    5. 多余空白/换行的 JSON
    """
    text = text.strip()
    if not text:
        return None

    # 预处理：将中文引号替换为英文引号
    text_clean = text.replace('\u201c', '"').replace('\u201d', '"')
    text_clean = text_clean.replace('\u2018', '"').replace('\u2019', '"')

    # 尝试 1: 直接解析（纯 JSON）
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # 尝试 2: 提取 markdown 代码块中的 JSON
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试 3: 提取最外层的 { } 块（贪婪匹配后回退）
    match = re.search(r'\{.*\}', text_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 尝试 4: 逐字符扫描找到第一个匹配的 { } 对
    depth = 0
    start = -1
    for i, ch in enumerate(text_clean):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text_clean[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    # 尝试 5: 尝试修复常见的 JSON 格式问题
    match = re.search(r'\{.*\}', text_clean, re.DOTALL)
    if match:
        raw = match.group(0)
        # 修复尾逗号
        fixed = re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None


def route_query(
    state: AgentState,
    *,
    web_search_available: bool = False,
) -> dict:
    """
    意图路由节点：判断用户问题应走 KB / Web / Direct。
    支持 glm-5 等新模型的兼容性处理：重试 + 增强 JSON 解析。

    Args:
        state: 当前 Agent 状态
        web_search_available: Web 搜索是否可用（API key 是否配置）
    Returns:
        {"route_type": RouteType, "task_type": TaskType, "messages": [AIMessage]}
    """
    messages = state.get("messages", [])

    # 提取最近几条消息作为路由上下文
    history_for_router = messages[-6:] if len(messages) > 6 else messages

    # 构造路由请求
    router_messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for msg in history_for_router:
        if isinstance(msg, HumanMessage):
            router_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            router_messages.append({"role": "assistant", "content": msg.content})

    # 尝试路由（含重试）
    last_error = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            response = chat_completion(
                router_messages,
                temperature=0,
                max_tokens=512,  # 增大 token 上限，兼容 glm-5
                thinking={"type": "disabled"},  # 路由任务简单，关闭 glm-5 深度思考
            )

            # 调试日志：打印原始响应，方便排查格式问题
            logger.debug(
                "路由原始响应 (attempt=%d): %r",
                attempt + 1,
                response[:200] if response else "(空)",
            )

            if not response or not response.strip():
                raise ValueError("模型返回空内容")

            # 解析 JSON（增强版，兼容多种格式）
            data = _parse_json_from_response(response)
            if data is None:
                raise ValueError(f"无法从响应中提取 JSON，原始响应前200字符: {repr(response[:200])}")

            decision = RouteDecision.model_validate(data)
            route_type = decision.route_type
            task_type = decision.task_type
            reasoning = decision.reasoning

            # Web 搜索不可用时降级
            if route_type == RouteType.WEB and not web_search_available:
                logger.warning("路由到 WEB 但搜索不可用，降级为 RETRIEVE")
                route_type = RouteType.RETRIEVE

            logger.info(
                "路由决策: route=%s, task=%s, reasoning=%s",
                route_type.value,
                task_type.value,
                reasoning[:100],
            )

            return {
                "route_type": route_type.value,
                "task_type": task_type.value,
                "messages": [AIMessage(content=f"[路由决策] 走{route_type.value}路径，{reasoning}")],
            }

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "路由解析失败 (attempt=%d/%d), 重试中: %s",
                    attempt + 1,
                    1 + _MAX_RETRIES,
                    e,
                )
            else:
                logger.warning("路由解析所有尝试均失败: %s", e)

    # 所有尝试失败，fallback: DIRECT + QA
    logger.warning(
        "路由解析最终 fallback 到 DIRECT+QA, 最后错误: %s",
        last_error,
    )
    return {
        "route_type": RouteType.DIRECT.value,
        "task_type": TaskType.QA.value,
        "messages": [AIMessage(content=f"[路由决策] 路由解析失败，fallback为直接回答: {last_error}")],
    }
