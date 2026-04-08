"""
路由节点 - 结构化输出 + 枚举校验 + fallback + 重试
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.prompts import ROUTER_SYSTEM_PROMPT
from app.agent.state import AgentState, RouteDecision, RouteType, TaskType
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.router")

# 最大重试次数
_MAX_RETRIES = 1


def _latest_user_question(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content.strip() if isinstance(msg.content, str) else ""
            if content:
                return content
    return ""


def _build_state_summary(state: AgentState) -> str:
    jd_data = state.get("jd_data") or {}
    resume_data = state.get("resume_data") or {}

    parts: list[str] = []
    parts.append(f"has_jd_data={bool(jd_data)}")
    parts.append(f"has_resume_data={bool(resume_data)}")

    jd_position = jd_data.get("position") or ""
    jd_summary = jd_data.get("summary") or ""
    if jd_position:
        parts.append(f"jd_position={jd_position}")
    if jd_summary:
        parts.append(f"jd_summary={jd_summary[:120]}")

    resume_summary = resume_data.get("summary") or ""
    resume_target = resume_data.get("target_position") or ""
    if resume_target:
        parts.append(f"resume_target_position={resume_target}")
    if resume_summary:
        parts.append(f"resume_summary={resume_summary[:120]}")

    return "\n".join(parts)


def _is_resume_like_text(question: str) -> bool:
    q = question.lower()
    if len(question) >= 250:
        resume_markers = (
            "教育经历",
            "工作经历",
            "项目经历",
            "专业技能",
            "教育背景",
            "自我评价",
            "实习经历",
            "校内经历",
            "个人信息",
            "姓名",
            "电话",
            "邮箱",
            "毕业院校",
            "技能栈",
        )
        hits = sum(1 for marker in resume_markers if marker in question)
        if hits >= 2:
            return True

    compact_markers = ("简历如下", "这是我的简历", "下面是我的简历", "我的简历内容", "帮我看下简历")
    return any(marker in q for marker in compact_markers)


def _rule_based_followup_route(question: str, state: AgentState, web_search_available: bool) -> dict | None:
    q = question.lower()
    has_jd_data = bool(state.get("jd_data"))
    has_resume_data = bool(state.get("resume_data"))

    jd_followup_keywords = (
        "关于以上jd",
        "关于以上 jd",
        "关于这个jd",
        "关于这个 jd",
        "基于以上jd",
        "基于以上 jd",
        "根据这个jd",
        "根据这个 jd",
        "这个岗位",
        "该岗位",
        "这个jd",
        "这个 jd",
        "面试准备",
        "面试阶段",
        "岗位重点",
        "岗位要求",
        "技术栈",
    )
    resume_keywords = ("简历", "评估", "优化", "修改简历", "改简历", "简历建议", "润色", "重写")
    match_keywords = ("匹配度", "匹配", "是否匹配", "缺少什么", "差距", "缺口", "对比jd", "对比 jd", "改进什么", "补什么", "最该补", "最需要改进")
    latest_keywords = ("最新", "今天", "本周", "最近", "2026", "实时", "新闻")

    if has_jd_data and (_is_resume_like_text(question) or (has_resume_data and any(k in q for k in match_keywords))):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.MATCH_FOLLOWUP.value if has_resume_data else TaskType.RESUME_ANALYSIS.value,
            "messages": [AIMessage(content="[路由决策] 已有 JD 上下文，识别为简历评估/匹配场景，走匹配分析路径")],
        }

    if has_resume_data and any(k in q for k in resume_keywords):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.RESUME_FOLLOWUP.value,
            "messages": [AIMessage(content="[路由决策] 识别为基于已有简历的优化追问，走 resume_followup")],
        }

    if has_jd_data and any(k in q for k in jd_followup_keywords):
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.JD_FOLLOWUP.value,
            "messages": [AIMessage(content="[路由决策] 基于已存在 JD 上下文，走 jd_followup")],
        }

    if any(k in q for k in latest_keywords):
        route_type = RouteType.WEB.value if web_search_available else RouteType.RETRIEVE.value
        return {
            "route_type": route_type,
        "task_type": TaskType.QA.value,
        "messages": [AIMessage(content=f"[路由决策] 时效性问题，走 {route_type} 路径")],
    }

    return None


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
    question = _latest_user_question(messages)
    if not question:
        logger.warning("路由阶段未找到当前用户问题，fallback 到 DIRECT+QA")
        return {
            "route_type": RouteType.DIRECT.value,
            "task_type": TaskType.QA.value,
            "messages": [AIMessage(content="[路由决策] 未找到有效问题，fallback 为直接回答")],
        }

    rule_result = _rule_based_followup_route(question, state, web_search_available)
    if rule_result is not None:
        logger.info(
            "路由规则命中: route=%s, task=%s, question=%s",
            rule_result["route_type"],
            rule_result["task_type"],
            question[:80],
        )
        return rule_result

    state_summary = _build_state_summary(state)
    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"当前用户问题：\n{question}\n\n"
                f"结构化状态摘要：\n{state_summary}\n\n"
                "请只输出一行 JSON。"
            ),
        },
    ]

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
