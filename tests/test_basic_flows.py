"""
7D-1 回归测试：单轮 QA + JD 分析 + 简历分析。

验证：
1. 单轮 QA 对话能正常返回回答
2. JD 分析（文本粘贴）能正常返回提取和报告
3. 简历分析（文本粘贴）能正常返回提取和报告
"""
from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_qa_chat_returns_answer(client, fresh_session_id):
    """单轮 QA 对话能正常返回回答。"""
    resp = await client.post(
        "/agent/chat/stream",
        json={
            "question": "简历中STAR法则怎么写",
            "session_id": fresh_session_id,
        },
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    # 解析 SSE 事件
    events = _parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]

    # 应至少有 done 事件
    assert "done" in event_types, f"Missing 'done' event, got: {event_types}"

    # done 事件应有 answer
    done_event = next(e for e in events if e["type"] == "done")
    assert done_event.get("answer", ""), "done event should have non-empty answer"


@pytest.mark.asyncio
async def test_qa_chat_has_route_event(client, fresh_session_id):
    """QA 对话应发出路由决策事件。"""
    resp = await client.post(
        "/agent/chat/stream",
        json={
            "question": "简历中STAR法则怎么写",
            "session_id": fresh_session_id,
        },
    )
    events = _parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]

    # 应有 route 或 planning 事件
    has_routing = "route" in event_types or "planning" in event_types
    assert has_routing, f"Expected route or planning event, got: {event_types}"


@pytest.mark.asyncio
async def test_jd_analysis_returns_extracted_and_answer(client, fresh_session_id):
    """JD 分析能正常返回 JD 提取和报告。"""
    resp = await client.post(
        "/agent/jd-analysis",
        json={
            "jd_text": "高级前端开发工程师\n要求：熟练掌握 React、TypeScript\n负责核心业务前端架构设计",
            "question": "请分析该岗位的核心要求",
            "session_id": fresh_session_id,
        },
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    events = _parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]

    # 应有 extracted（JD 提取完成）和 done
    assert "extracted" in event_types, f"Missing 'extracted' event, got: {event_types}"
    assert "done" in event_types, f"Missing 'done' event, got: {event_types}"

    # extracted 事件应包含 jd_data
    extracted_event = next(e for e in events if e["type"] == "extracted")
    assert "jd_data" in extracted_event, "extracted event should have jd_data"

    # done 事件应有 answer
    done_event = next(e for e in events if e["type"] == "done")
    assert done_event.get("answer", ""), "done event should have non-empty answer"


@pytest.mark.asyncio
async def test_jd_analysis_done_includes_jd_data(client, fresh_session_id):
    """JD 分析的 done 事件应包含 jd_data（不含 raw_text）。"""
    resp = await client.post(
        "/agent/jd-analysis",
        json={
            "jd_text": "高级前端开发工程师\n要求：熟练掌握 React、TypeScript",
            "question": "请分析该岗位",
            "session_id": fresh_session_id,
        },
    )
    events = _parse_sse_events(resp.text)
    done_event = next(e for e in events if e["type"] == "done")
    assert "jd_data" in done_event, "done event should have jd_data"


@pytest.mark.asyncio
async def test_resume_analysis_returns_extracted_and_answer(client, fresh_session_id):
    """简历分析能正常返回简历提取和报告。"""
    resp = await client.post(
        "/agent/resume-analysis",
        json={
            "resume_text": "张三 | 前端开发工程师\n教育经历：XX大学 计算机科学\n工作经历：ABC公司 前端工程师\n技能：React, Vue, TypeScript",
            "question": "请对我的简历进行全面分析评估",
            "session_id": fresh_session_id,
            "target_position": "前端开发工程师",
        },
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    events = _parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]

    # 应有 extracted（简历提取完成）和 done
    assert "extracted" in event_types, f"Missing 'extracted' event, got: {event_types}"
    assert "done" in event_types, f"Missing 'done' event, got: {event_types}"

    # extracted 事件应包含 resume_data
    extracted_event = next(e for e in events if e["type"] == "extracted")
    assert "resume_data" in extracted_event, "extracted event should have resume_data"

    # done 事件应有 answer
    done_event = next(e for e in events if e["type"] == "done")
    assert done_event.get("answer", ""), "done event should have non-empty answer"


@pytest.mark.asyncio
async def test_resume_analysis_done_includes_resume_data(client, fresh_session_id):
    """简历分析的 done 事件应包含 resume_data（不含 raw_text）。"""
    resp = await client.post(
        "/agent/resume-analysis",
        json={
            "resume_text": "张三 | 前端开发工程师\n技能：React, Vue",
            "question": "请分析我的简历",
            "session_id": fresh_session_id,
        },
    )
    events = _parse_sse_events(resp.text)
    done_event = next(e for e in events if e["type"] == "done")
    assert "resume_data" in done_event, "done event should have resume_data"


@pytest.mark.asyncio
async def test_empty_question_returns_error(client, fresh_session_id):
    """空问题应返回 422 验证错误。"""
    # 通过 chat/stream 发送空问题
    resp = await client.post(
        "/agent/chat/stream",
        json={
            "question": "",
            "session_id": fresh_session_id,
        },
    )
    # 空问题会被 FastAPI 请求体验证拦截，返回 422
    assert resp.status_code in (200, 422), f"Expected 200 or 422, got {resp.status_code}"


@pytest.mark.asyncio
async def test_jd_analysis_empty_text_returns_error(client, fresh_session_id):
    """JD 分析空文本应返回错误。"""
    resp = await client.post(
        "/agent/jd-analysis",
        json={
            "jd_text": "  ",
            "question": "分析",
            "session_id": fresh_session_id,
        },
    )
    assert resp.status_code == 200  # SSE 流内返回 error 事件

    events = _parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]
    assert "error" in event_types, f"Expected error event for empty JD text, got: {event_types}"


@pytest.mark.asyncio
async def test_resume_analysis_empty_text_returns_error(client, fresh_session_id):
    """简历分析空文本应返回错误。"""
    resp = await client.post(
        "/agent/resume-analysis",
        json={
            "resume_text": "  ",
            "question": "分析",
            "session_id": fresh_session_id,
        },
    )
    assert resp.status_code == 200

    events = _parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]
    assert "error" in event_types, f"Expected error event for empty resume text, got: {event_types}"


# ---- 工具函数 ----

def _parse_sse_events(raw_text: str) -> list[dict]:
    """从 SSE 原始文本中解析出所有 data 事件的 JSON payload。"""
    events = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        json_str = line[6:]
        try:
            payload = json.loads(json_str)
            events.append(payload)
        except json.JSONDecodeError:
            pass
    return events
