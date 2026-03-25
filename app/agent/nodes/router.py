"""
路由节点 - 结构化输出 + 枚举校验 + fallback
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


def _parse_json_from_response(text: str) -> dict | None:
    """
    从 LLM 响应中提取 JSON，支持：
    1. 纯 JSON
    2. Markdown 代码块包裹的 JSON
    """
    text = text.strip()
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 markdown 代码块
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取第一个 { } 块
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
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

    try:
        response = chat_completion(
            router_messages,
            temperature=0,
            max_tokens=256,
        )

        # 解析 JSON（兼容 markdown 代码块）
        data = _parse_json_from_response(response)
        if data is None:
            raise ValueError(f"无法从响应中提取 JSON: {response[:100]}")
        
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

    except Exception as e:
        # fallback: DIRECT + QA
        logger.warning("路由解析失败，fallback 到 DIRECT+QA: %s", e)
        route_type = RouteType.DIRECT
        task_type = TaskType.QA
        reasoning = f"路由解析失败，fallback: {e}"

    return {
        "route_type": route_type.value,  # 枚举转字符串，避免 msgpack 序列化问题
        "task_type": task_type.value,
        "messages": [AIMessage(content=f"[路由决策] 走{route_type.value}路径，{reasoning}")],
    }
