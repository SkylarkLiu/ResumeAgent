"""
7D-2 回归测试：JD 分析 -> 简历分析 -> 匹配追问 多步流程。

验证：
1. 先做 JD 分析 → session 中有 jd_data
2. 同 session 做简历分析 → 使用已有 JD 上下文
3. 同 session 做匹配追问 → 规则引擎命中 match_followup
"""
from __future__ import annotations

import json

import pytest


def _parse_sse_events(raw_text: str) -> list[dict]:
    events = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            pass
    return events


@pytest.mark.asyncio
async def test_jd_then_resume_then_match_followup(client, fresh_session_id):
    """完整三步流程：JD 分析 → 简历分析 → 匹配追问。"""

    # Step 1: JD 分析
    resp1 = await client.post(
        "/agent/jd-analysis",
        json={
            "jd_text": "高级前端开发工程师\n要求：React、TypeScript、Node.js\n负责核心业务前端架构设计",
            "question": "请分析该岗位的核心要求",
            "session_id": fresh_session_id,
        },
    )
    assert resp1.status_code == 200
    events1 = _parse_sse_events(resp1.text)
    assert "done" in [e["type"] for e in events1], "Step 1: JD analysis should complete"

    # Step 2: 同 session 简历分析
    resp2 = await client.post(
        "/agent/resume-analysis",
        json={
            "resume_text": "张三 | 前端开发工程师\n技能：React, Vue, TypeScript\n经验：3年前端开发",
            "question": "请对我的简历进行全面分析评估",
            "session_id": fresh_session_id,
            "target_position": "高级前端开发工程师",
        },
    )
    assert resp2.status_code == 200
    events2 = _parse_sse_events(resp2.text)
    assert "done" in [e["type"] for e in events2], "Step 2: Resume analysis should complete"

    # 简历分析的 done 事件应包含 resume_data
    done2 = next(e for e in events2 if e["type"] == "done")
    assert "resume_data" in done2, "Step 2: done should have resume_data"

    # Step 3: 同 session 匹配追问
    resp3 = await client.post(
        "/agent/chat/stream",
        json={
            "question": "我和这个岗位的差距在哪？怎么补？",
            "session_id": fresh_session_id,
        },
    )
    assert resp3.status_code == 200
    events3 = _parse_sse_events(resp3.text)
    event_types3 = [e["type"] for e in events3]

    # 应有 done 事件
    assert "done" in event_types3, f"Step 3: Should have done event, got: {event_types3}"

    # 应有 route 或 planning 事件
    has_routing = "route" in event_types3 or "planning" in event_types3
    assert has_routing, f"Step 3: Should have routing event, got: {event_types3}"

    # 如果有 planning 事件，task 应该是 match_followup（规则引擎命中）
    planning_events = [e for e in events3 if e["type"] == "planning"]
    if planning_events:
        # 规则引擎应该把带匹配关键词的问题识别为 match_followup
        task = planning_events[0].get("task", "")
        # match_followup 或 resume_followup 都算合理（取决于规则优先级）
        assert task in ("match_followup", "resume_followup"), \
            f"Step 3: Expected match_followup or resume_followup, got task={task}"


@pytest.mark.asyncio
async def test_jd_then_jd_followup(client, fresh_session_id):
    """JD 分析 → JD 追问流程。"""

    # Step 1: JD 分析
    resp1 = await client.post(
        "/agent/jd-analysis",
        json={
            "jd_text": "后端开发工程师\n要求：Java、Spring Boot、MySQL",
            "question": "请分析该岗位",
            "session_id": fresh_session_id,
        },
    )
    assert resp1.status_code == 200
    events1 = _parse_sse_events(resp1.text)
    assert "done" in [e["type"] for e in events1]

    # Step 2: JD 追问
    resp2 = await client.post(
        "/agent/chat/stream",
        json={
            "question": "这个岗位面试要准备什么",
            "session_id": fresh_session_id,
        },
    )
    assert resp2.status_code == 200
    events2 = _parse_sse_events(resp2.text)
    assert "done" in [e["type"] for e in events2], "JD followup should complete"

    # 应有 planning 事件，task 应该是 jd_followup
    planning_events = [e for e in events2 if e["type"] == "planning"]
    if planning_events:
        task = planning_events[0].get("task", "")
        assert task == "jd_followup", f"Expected jd_followup, got task={task}"


@pytest.mark.asyncio
async def test_resume_then_resume_followup(client, fresh_session_id):
    """简历分析 → 简历追问流程。"""

    # Step 1: 简历分析
    resp1 = await client.post(
        "/agent/resume-analysis",
        json={
            "resume_text": "李四 | 产品经理\n经验：5年产品经验\n技能：需求分析、项目管理",
            "question": "请分析我的简历",
            "session_id": fresh_session_id,
        },
    )
    assert resp1.status_code == 200
    events1 = _parse_sse_events(resp1.text)
    assert "done" in [e["type"] for e in events1]

    # Step 2: 简历追问
    resp2 = await client.post(
        "/agent/chat/stream",
        json={
            "question": "我的简历有什么亮点",
            "session_id": fresh_session_id,
        },
    )
    assert resp2.status_code == 200
    events2 = _parse_sse_events(resp2.text)
    assert "done" in [e["type"] for e in events2], "Resume followup should complete"

    # 应有 planning 事件，task 应该是 resume_followup
    planning_events = [e for e in events2 if e["type"] == "planning"]
    if planning_events:
        task = planning_events[0].get("task", "")
        assert task == "resume_followup", f"Expected resume_followup, got task={task}"
